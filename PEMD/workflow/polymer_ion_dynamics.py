import os
import math
import numpy as np
import MDAnalysis as mda
from tqdm.auto import tqdm
from PEMD.analysis import msd
from MDAnalysis.analysis import rdf
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor, as_completed


class PolymerIonDynamics:

    def __init__(self, work_dir, tpr_file, xtc_wrap_file, xtc_unwrap_file, cation_atoms, anion_atoms, polymer_atoms,
                 run_start, run_end, dt, dt_collection, ):
        self.work_dir = work_dir
        self.tpr_file = tpr_file
        self.xtc_wrap_file = xtc_wrap_file
        self.xtc_unwrap_file = xtc_unwrap_file
        self.run_start = run_start
        self.run_end = run_end
        self.dt = dt
        self.dt_collection = dt_collection

        self.u_wrap = self.load_trajectory(self.xtc_wrap_file)
        self.u_unwrap = self.load_trajectory(self.xtc_unwrap_file)

        self.cation_atoms_wrap = self.u_wrap.select_atoms(cation_atoms)
        self.anion_atoms_wrap = self.u_wrap.select_atoms(anion_atoms)
        self.polymer_atoms_wrap = self.u_wrap.select_atoms(polymer_atoms)

        self.cation_atoms_unwrap = self.u_unwrap.select_atoms(cation_atoms)
        self.anion_atoms_unwrap = self.u_unwrap.select_atoms(anion_atoms)
        self.polymer_atoms_unwrap = self.u_unwrap.select_atoms(polymer_atoms)

        self.num_cations = len(self.cation_atoms_unwrap)
        self.num_o_polymer = len(self.polymer_atoms_unwrap)
        self.num_chain = len(np.unique(self.polymer_atoms_unwrap.resids))
        self.num_o_chain = int(self.num_o_polymer // self.num_chain)

        self.initialize_simulation()

    def load_trajectory(self, xtc_file):
        tpr_path = os.path.join(self.work_dir, self.tpr_file)
        xtc_path = os.path.join(self.work_dir, xtc_file)
        return mda.Universe(tpr_path, xtc_path)

    def initialize_simulation(self):
        self.box_size = self.u_unwrap.dimensions[0]
        self.cutoff_radius = self.calculate_cutoff_radius()
        self.poly_o_ave_n, self.poly_n, self.bound_o_n, self.poly_o_positions= self.process_traj()

    def calculate_cutoff_radius(self, nbins=200, range_rdf=(0.0, 10.0)):

        rdf_analysis = rdf.InterRDF(self.cation_atoms_wrap, self.polymer_atoms_wrap, nbins=nbins, range=range_rdf)
        rdf_analysis.run()

        bins = rdf_analysis.results.bins
        rdf_values = rdf_analysis.results.rdf

        # Identifying the first peak and the subsequent first minimum
        deriv_sign_changes = np.diff(np.sign(np.diff(rdf_values)))
        peak_indices = np.where(deriv_sign_changes < 0)[0] + 1
        if not peak_indices.size:
            raise ValueError("No peak found in RDF data. Please check the atom selection or simulation parameters.")
        first_peak_index = peak_indices[0]

        min_after_peak_indices = np.where(deriv_sign_changes[first_peak_index:] > 0)[0] + first_peak_index + 1
        if not min_after_peak_indices.size:
            raise ValueError("No minimum found after the first peak in RDF data.")
        first_min_index = min_after_peak_indices[0]

        return round(float(bins[first_min_index]), 3)

    def distance(self, x0, x1, box_length):
        delta = x1 - x0
        delta = np.where(delta > 0.5 * box_length, delta - box_length, delta)
        delta = np.where(delta < -0.5 * box_length, delta + box_length, delta)
        return delta

    def process_traj(self,):

        times = self.times_range(self.run_end - self.run_start)
        poly_o_ave_n = np.zeros((len(times), self.num_cations))  # Initialize array
        poly_n = np.zeros((len(times), self.num_cations))  # Initialize array
        bound_o_n = np.full((len(times), self.num_cations, 10), -1, dtype=int)  # 初始化bound氧的索引
        poly_o_positions = np.zeros((len(times), self.num_o_polymer, 3))  # 初始化氧坐标的数组

        for ts in tqdm(self.u_unwrap.trajectory[self.run_start: self.run_end], desc='Processing trajectory'):

            for n, li in enumerate(self.cation_atoms_unwrap):
                distances_oe_vec = self.distance(self.polymer_atoms_unwrap.positions, li.position,
                                                 self.box_size)
                distances_oe = np.linalg.norm(distances_oe_vec, axis=1)
                close_oe_index = np.where(distances_oe <= self.cutoff_radius)[0]

                if len(close_oe_index) > 0:
                    o_resids = self.polymer_atoms_unwrap[close_oe_index].resids

                    if np.all(o_resids == o_resids[0]):
                        poly_o_ave_n[ts.frame, n] = np.mean(close_oe_index)
                        poly_n[ts.frame, n] = o_resids[0]
                        bound_o_n[ts.frame, n, :len(close_oe_index)] = close_oe_index  # bound氧的索引
                    else:
                        poly_n[ts.frame, n] = -1

            for i in range(int(self.num_chain)):
                oe_in_onechain = self.u_unwrap.select_atoms(f'resid {i + 1}')   # 选择当前链中的原子
                # 计算当前链的原子相对于质心的位置
                poly_o_positions[ts.frame, self.num_o_chain * i:self.num_o_chain * (i + 1),
                :] = self.polymer_atoms_unwrap.positions[self.num_o_chain * i:self.num_o_chain * (i + 1),
                     :] - oe_in_onechain.center_of_mass()

        return poly_o_ave_n, poly_n, bound_o_n, poly_o_positions

    def calculate_tau3(self, ):
        t_max = (self.run_end - self.run_start) * self.dt_collection * self.dt / 1000  # Convert to ns
        backjump_threshold = 100 / (self.dt_collection * self.dt)  # 100 ps within jumps considered transient

        hopping_counts = [0] * self.num_cations  # Records hopping counts for each lithium-ion
        potential_hops = {}  # Records the last hopping time and chain for each lithium-ion
        last_bound_chains = [None] * self.num_cations  # Records the chain each lithium-ion was last bound to

        for i in range(self.num_cations):
            for t in range(self.run_start, self.run_end):
                li_bound_current_chain = self.poly_n[t, i]

                # First transition from unbound to bound
                if last_bound_chains[i] is None and li_bound_current_chain not in [0, -1]:
                    last_bound_chains[i] = li_bound_current_chain

                # Check for a valid hop
                elif last_bound_chains[i] is not None and li_bound_current_chain != last_bound_chains[i] and li_bound_current_chain not in [0, -1, -2]:
                    if i not in potential_hops or potential_hops[i]['chain'] != li_bound_current_chain:
                        potential_hops[i] = {'time': t, 'chain': li_bound_current_chain}

                    if i in potential_hops:
                        elapsed_time = t - potential_hops[i]['time']

                        if elapsed_time >= backjump_threshold:  # Confirm and count a hop
                            hopping_counts[i] += 1
                            last_bound_chains[i] = li_bound_current_chain
                            del potential_hops[i]

        total_hops = sum(hopping_counts)
        tau3 = t_max * self.num_cations / total_hops if total_hops > 0 else float('inf')  # Avoid division by zero

        return tau3

    def calculate_delta_n_square(self, dt, ):
        """Calculate mean squared displacement for the given time difference dt."""
        msd_in_dt = []
        if dt == 0:
            return 0  # MSD at dt=0 is 0 as Δn would be 0
        for t in range(self.run_start, self.run_end - dt):
            delta_n = self.poly_o_ave_n[t + dt] - self.poly_o_ave_n[t]
            delta_n_square = np.square(delta_n)

            mask_i = (self.poly_o_ave_n[t + dt] == 0) | (self.poly_o_ave_n[t + dt] == -1)
            mask_j = (self.poly_o_ave_n[t] == 0) | (self.poly_o_ave_n[t] == -1)
            mask_h = self.poly_n[t + dt] != self.poly_n[t]

            mask_unbound = (self.poly_n[t:t + dt] != self.poly_n[t, None])
            unbound_counts = np.sum(mask_unbound, axis=0)
            mask_k = (unbound_counts / dt) > 0.05

            full_mask = mask_i | mask_j | mask_h | mask_k
            delta_n_square_filtered = delta_n_square[~full_mask]
            if delta_n_square_filtered.size > 0:
                msd_in_dt.append(np.mean(delta_n_square_filtered))

        return np.mean(msd_in_dt) if msd_in_dt else 0

    def extrapolate_msd(self, tau3, times_M1, msd_M1,):
        valid_indices = (times_M1 > 0) & (msd_M1 > 0)
        times_filtered = times_M1[valid_indices]
        msd_filtered = msd_M1[valid_indices]

        log_times = np.log(times_filtered)
        log_msd = np.log(msd_filtered)

        intercept = np.mean(log_msd - 0.8 * log_times)

        t_extrap = tau3 * 1000

        slope_at_t_extrap = np.exp(intercept) * 0.8 * t_extrap ** (0.8 - 1)
        D1 = slope_at_t_extrap / 2
        tau1 = ((self.num_cations - 1) ** 2) / (math.pi ** 2 * D1) / 1000

        return tau1

    def ms_endtoend_distance(self):
        re_all = []

        for ts in tqdm(self.u_unwrap.trajectory[self.run_start: self.run_end], desc="Processing"):
            ts_vectors = []

            for mol_id in range(1, int(self.num_chain) + 1):  # Assuming 20 molecules
                chain_indices = np.where(self.polymer_atoms_unwrap.resids == mol_id)[0]  # 获得所有聚合物链醚氧的index，并每一条链单独储存一个index
                if len(chain_indices) > 1:  # Ensure there is more than one oxygen atom
                    chain_coor = self.polymer_atoms_unwrap.positions[chain_indices]  # 获得每条聚合物链醚氧的坐标
                    chain1_coor = chain_coor[1:]
                    chain2_coor = chain_coor[:-1]
                    b0_array = self.distance(chain1_coor, chain2_coor, self.box_size)  # 生成每个间隔醚氧的向量
                    re_vector = np.sum(b0_array, axis=0)  # 所有向量加和
                    re = np.linalg.norm(re_vector)  # 对向量进行模长的计算
                    ts_vectors.append(re)

            if ts_vectors:
                ts_vectors = np.square(ts_vectors)
                re_mean = np.mean(ts_vectors)
                re_all.append(re_mean)

        return re_all

    def times_range(self, t_end):
        times = np.arange(0, t_end * self.dt * self.dt_collection, self.dt * self.dt_collection, dtype=int)
        return times

    def calculate_oe_msd(self, times_MR):
        n_atoms = np.shape(self.poly_o_positions)[1]
        msd_oe = msd.calc_Lii_self(self.poly_o_positions, times_MR) / n_atoms
        return msd_oe

    def calculate_msd_M2(self, dt):
        msd_in_t = []
        if dt == 0:
            return 0
        for t in range(self.run_start, self.run_end - dt):
            delta_n = self.poly_o_positions[t + dt] - self.poly_o_positions[t]
            delta_n_square = np.sum(np.square(delta_n), axis=1)

            i = np.where((np.abs(self.poly_o_ave_n[dt + t] - self.poly_o_ave_n[t]) <= 1) & (self.poly_o_ave_n[t] != 0))[0]

            bound_counts = np.sum(np.abs(self.poly_o_ave_n[t:t + dt] - self.poly_o_ave_n[t]) <= 1, axis=0)
            j = np.where((bound_counts / dt) >= 0.85)[0]
            li_intersection = np.intersect1d(i, j)

            all_bound_oe_indices = []  # 创建一个列表来收集所有有效的索引
            for idx in li_intersection:
                valid_indices = self.bound_o_n[t, idx][self.bound_o_n[t, idx] != -1]
                if valid_indices.size > 0:
                    all_bound_oe_indices.extend(valid_indices)  # 收集所有有效的索引

            if all_bound_oe_indices:
                msd_in_t.append(np.mean(delta_n_square[all_bound_oe_indices]))

        return np.mean(msd_in_t) if msd_in_t else 0

    def calculate_msd_parallel(self, calculate_msd, time_window):
        """Calculate the parallel computation of MSD over multiple time deltas."""
        msd = []
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(calculate_msd, dt,): dt for dt in range(time_window)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculate MSD"):
                dt = futures[future]
                msd_result = future.result()
                msd.append((dt, msd_result))
        msd.sort(key=lambda x: x[0])
        return np.array([result for _, result in msd])

    def rouse_model(self, t, tau, Re_square,):
        """计算 Rouse 模型的理论值，用于拟合 MSD 数据。"""
        sum_part = sum([(1 - np.exp(-p ** 2 * t / tau)) / p ** 2 for p in range(1, self.num_o_chain - 1)])
        return (2 * Re_square / np.pi ** 2) * sum_part

    def fit_rouse_model(self, re_all, times, msd):
        """计算 Rouse 时间常数并拟合 MSD 数据。"""
        Re_square = np.mean(re_all)  # 平均平方端到端距离
        tau,_ = curve_fit(lambda t, tau: self.rouse_model(t, tau, Re_square), times, msd)
        return tau[0] / 1000





