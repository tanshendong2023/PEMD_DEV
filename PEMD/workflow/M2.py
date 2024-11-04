import os
import numpy as np
import MDAnalysis as mda
from tqdm.auto import tqdm
from PEMD.analysis import msd
from scipy.optimize import curve_fit
from multiprocessing import Pool
from PEMD.analysis import residence_time, coordination


work_dir = './'
data_tpr_file = 'nvt_prod.tpr'
dcd_xtc_file = 'nvt_prod_unwrap.xtc'

select_atoms = {
    'cation': 'resname LIP and name Li',
    'anion': 'resname NSC and name OBT',
    'polymer': 'resname MOL and name O',
}


# Load the trajectory
u = coordination.load_md_trajectory(work_dir, 'nvt_prod.tpr', 'nvt_prod.xtc')

# Select the atoms of interest
li_atoms = u.select_atoms('resname LIP and name Li')
peo_atoms = u.select_atoms('resname MOL and name O')
tfsi_atoms = u.select_atoms('resname NSC and name OBT')

# Perform RDF and coordination number calculation
bins_peo, rdf_peo, coord_num_peo = coordination.calculate_rdf_and_coordination(u, li_atoms, peo_atoms)
bins_tfsi, rdf_tfsi, coord_num_tfsi = coordination.calculate_rdf_and_coordination(u, li_atoms, tfsi_atoms)

# obtain the coordination number and first solvation shell distance
r_li_peo, y_rdf_peo, y_coord_peo = coordination.obtain_rdf_coord(bins_peo, rdf_peo, coord_num_peo)
r_li_tfsi, y_rdf_tfsi, y_coord_tfsi = coordination.obtain_rdf_coord(bins_tfsi, rdf_tfsi, coord_num_tfsi)

# setting the first solvation shell distance
cutoff_radii = {
    'PEO': r_li_peo,
    'TFSI': r_li_tfsi,
}


run_start = 0
run_end = 80001  # step
dt = 0.001
dt_collection = 5000 # step
num_li = 50
num_oe = 50
chains = 20
time_window = 501


def distance(x0, x1, box_length):
    delta = x1 - x0
    delta = np.where(delta > 0.5 * box_length, delta - box_length, delta)
    delta = np.where(delta < -0.5 * box_length, delta + box_length, delta)
    return delta

def load_data_traj(work_dir, data_tpr_file, dcd_xtc_file, select_atoms, run_start, run_end, dt, dt_collection,
                   cutoff_radii):

    # load trajectory
    data_tpr_file_path = os.path.join(work_dir, data_tpr_file)
    dcd_xtc_file_path = os.path.join(work_dir, dcd_xtc_file)

    u = mda.Universe(data_tpr_file_path, dcd_xtc_file_path)

    # obtain the time array
    t_total = run_end - run_start  # total simulation steps, minus equilibration time
    times = np.arange(0, t_total * dt * dt_collection, dt * dt_collection, dtype=int)

    li_atoms = u.select_atoms(select_atoms['cation'])
    oe_atoms = u.select_atoms(select_atoms['polymer'])

    # 初始化参数
    chain_n = len(np.unique(oe_atoms.resids))  # 链的条数
    oe_per_chain = len(oe_atoms) // chain_n  # 每链中的OE数目

    oe_ave_n = np.zeros((run_end - run_start, len(li_atoms)))  # 初始化锂周围氧的平均索引
    poly_n = np.zeros((run_end - run_start, len(li_atoms)))  # 初始化锂所在聚合物链索引的数组
    bound_o_n = np.full((run_end - run_start, len(li_atoms), 10), -1, dtype=int)  # 初始化bound氧的索引
    oe_positions = np.zeros((run_end - run_start, len(oe_atoms), 3))  # 初始化氧坐标的数组

    for ts in tqdm(u.trajectory[run_start: run_end], desc='processing'):

        box_size = ts.dimensions[0]

        for n, li in enumerate(li_atoms):

            distances_oe_vec = distance(oe_atoms.positions, li.position, box_size)
            distances_oe = np.linalg.norm(distances_oe_vec, axis=1)
            close_oe_index = np.where(distances_oe <= cutoff_radii['PEO'])[0]

            if len(close_oe_index) > 0:  # 确保选择的Li都和醚氧相互作用
                o_resids = oe_atoms[close_oe_index].resids  # 找到醚氧所在的链

                # 检查所有的醚氧和同一个聚合物相互作用
                if np.all(o_resids == o_resids[0]):
                    oe_ave_n[ts.frame, n] = np.mean(close_oe_index)  # 锂周围氧的平均索引
                    poly_n[ts.frame, n] = o_resids[0]  # 锂所在链的平均索引
                    bound_o_n[ts.frame, n, :len(close_oe_index)] = close_oe_index  # bound氧的索引

        for i in range(int(chain_n)):
            oe_in_onechain = u.select_atoms(f'resid {i + 1}')   # 选择当前链中的原子
            # 计算当前链的原子相对于质心的位置
            oe_positions[ts.frame, oe_per_chain * i:oe_per_chain * (i + 1), :] = oe_atoms.positions[oe_per_chain * i:oe_per_chain * (i + 1),:] - oe_in_onechain.center_of_mass()

    return oe_ave_n, poly_n, bound_o_n, oe_positions, times


def ms_endtoend_distance(work_dir, data_tpr_file, dcd_xtc_file, run_start, dt_collection, chains, select_atoms):
    # Load the trajectory
    data_tpr_file_path = os.path.join(work_dir, data_tpr_file)
    dcd_xtc_file_path = os.path.join(work_dir, dcd_xtc_file)

    u = mda.Universe(data_tpr_file_path, dcd_xtc_file_path)
    box_size = u.dimensions[0]

    # Preselecting atoms
    oe_atoms = u.select_atoms(select_atoms['polymer'])
    re_all = []  # inital the end to end distance list

    for ts in tqdm(u.trajectory[int(run_start / dt_collection):], desc="Processing"):
        ts_vectors = []

        for mol_id in range(1, int(chains) + 1):  # Assuming 20 molecules
            chain_indices = np.where(oe_atoms.resids == mol_id)[0]  # 获得所有聚合物链醚氧的index，并每一条链单独储存一个index
            if len(chain_indices) > 1:  # Ensure there is more than one oxygen atom
                chain_coor = oe_atoms.positions[chain_indices]  # 获得每条聚合物链醚氧的坐标
                chain1_coor = chain_coor[1:]
                chain2_coor = chain_coor[:-1]
                b0_array = distance(chain1_coor, chain2_coor, box_size)  # 生成每个间隔醚氧的向量
                re_vector = np.sum(b0_array, axis=0)  # 所有向量加和
                re = np.linalg.norm(re_vector)  # 对向量进行模长的计算
                ts_vectors.append(re)

        if ts_vectors:
            ts_vectors = np.square(ts_vectors)
            re_mean = np.mean(ts_vectors)
            re_all.append(re_mean)

    return re_all


def compute_msd_for_dt(dt, threshold=0.85):
    msd_in_t = []
    if dt == 0:
        return 0

    for t in range(run_start, run_end - dt):
        delta_n = oe_positions[t + dt] - oe_positions[t]
        delta_n_square = np.sum(np.square(delta_n), axis=1)

        i = np.where((np.abs(oe_ave_n[dt + t] - oe_ave_n[t]) <= 1) & (oe_ave_n[t] != 0))[0]

        bound_counts = np.sum(np.abs(oe_ave_n[t:t + dt] - oe_ave_n[t]) <= 1, axis=0)
        j = np.where((bound_counts / dt) >= threshold)[0]
        li_intersection = np.intersect1d(i, j)

        # msd_for_bound_li = []
        all_bound_oe_indices = []  # 创建一个列表来收集所有有效的索引
        for idx in li_intersection:
            valid_indices = bound_o_n[t, idx][bound_o_n[t, idx] != -1]
            if valid_indices.size > 0:
                all_bound_oe_indices.extend(valid_indices)  # 收集所有有效的索引

        if all_bound_oe_indices:
            msd_in_t.append(np.mean(delta_n_square[all_bound_oe_indices]))

    return np.mean(msd_in_t) if msd_in_t else 0

def compute_tR(re_all, times, num_oe, msd_oe):
    """计算 Rouse 时间常数并拟合 MSD 数据。"""
    Re_square = np.mean(re_all)  # 平均平方端到端距离
    popt, pcov = curve_fit(lambda t, tau_R: rouse_model(t, tau_R, Re_square, num_oe), times, msd_oe)
    tauR_fit = popt[0]  # 转换为纳秒

    fit_curve = rouse_model(times, tauR_fit, Re_square, num_oe)

    return tauR_fit / 1000, fit_curve

def rouse_model(t, tau_R, Re_square, N):
    """计算 Rouse 模型的理论值，用于拟合 MSD 数据。"""
    sum_part = sum([(1 - np.exp(-p ** 2 * t / tau_R)) / p ** 2 for p in range(1, N - 1)])
    return (2 * Re_square / np.pi ** 2) * sum_part


if __name__ == '__main__':
    #  loads and processes trajectory data
    oe_ave_n,poly_n, bound_o_n, oe_positions, times = load_data_traj(work_dir, data_tpr_file, dcd_xtc_file, \
                                                                     select_atoms, run_start, run_end, dt, \
                                                                     dt_collection, cutoff_radii,)

    # calculate the mean square end to end distance
    re_all = ms_endtoend_distance(work_dir, data_tpr_file, dcd_xtc_file, run_start, dt_collection, \
                                  chains, select_atoms)

    times_msd = np.arange(0, time_window * dt * dt_collection-1, dt * dt_collection, dtype=int)

    with Pool() as pool:
        msd = list(tqdm(pool.imap(compute_msd_for_dt, range(0, time_window)), total=time_window))

    # fit the tR by the rouse model
    tau2, fit_curve = compute_tR(re_all, times_msd, num_oe, msd)

    # Define the path for the results file
    results_file_path = os.path.join(work_dir, 'tau2.txt')

    # Write results to a text file
    with open(results_file_path, 'w') as file:
        file.write(f"Calculated τ2: {tau2:.2f} ns\n")

    print(f"Results saved to {results_file_path}")




