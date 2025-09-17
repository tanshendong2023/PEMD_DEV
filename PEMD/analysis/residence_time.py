# import os
# import numpy as np
# import MDAnalysis as mda
# from tqdm.auto import tqdm
# from PEMD.analysis import msd
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from MDAnalysis.analysis import distances
# from MDAnalysis.lib.distances import distance_array
#
#
# def distance(x0, x1, box_length):
#     delta = x1 - x0
#     delta = np.where(delta > 0.5 * box_length, delta - box_length, delta)
#     delta = np.where(delta < -0.5 * box_length, delta + box_length, delta)
#     return delta
#
#
# def load_data_traj(work_dir, data_tpr_file, dcd_xtc_file, select_atoms, run_start, run_end, dt, dt_collection,
#                    cutoff_radii):
#
#     # load trajectory
#     data_tpr_file_path = os.path.join(work_dir, data_tpr_file)
#     dcd_xtc_file_path = os.path.join(work_dir, dcd_xtc_file)
#
#     u = mda.Universe(data_tpr_file_path, dcd_xtc_file_path)
#
#     # obtain the time array
#     t_total = run_end - run_start  # total simulation steps, minus equilibration time
#     times = np.arange(0, t_total * dt * dt_collection, dt * dt_collection, dtype=int)
#
#     li_atoms = u.select_atoms(select_atoms['cation'])
#     tfsi_atoms = u.select_atoms(select_atoms['anion'])
#     oe_atoms = u.select_atoms(select_atoms['polymer'])
#
#     oe_ave_n = np.zeros((run_end - run_start, len(li_atoms)))  # 初始化锂周围氧的平均索引
#     poly_n = np.zeros((run_end - run_start, len(li_atoms)))  # 初始化锂所在聚合物链索引的数组
#     tfsi_n = np.zeros((run_end - run_start, len(li_atoms)))  # 初始化锂所在聚合物链索引的数组
#     bound_o_n = np.full((run_end - run_start, len(li_atoms), 10), -1, dtype=int)  # 初始化bound氧的索引
#     bound_o_positions = np.zeros((run_end - run_start, len(oe_atoms), 3))  # 初始化氧坐标的数组
#
#     for ts in tqdm(u.trajectory[run_start: run_end], desc='processing'):
#
#         box_size = ts.dimensions[0]
#
#         for n, li in enumerate(li_atoms):
#
#             distances_oe_vec = distance(oe_atoms.positions, li.position, box_size)
#             distances_oe = np.linalg.norm(distances_oe_vec, axis=1)
#             close_oe_index = np.where(distances_oe <= cutoff_radii['PEO'])[0]
#
#             distances_tfsi_vec = distance(tfsi_atoms.positions, li.position, box_size)
#             distances_tfsi = np.linalg.norm(distances_tfsi_vec, axis=1)
#             close_tfsi_index = np.where(distances_tfsi <= cutoff_radii['TFSI'])[0]
#
#             if len(close_oe_index) > 0:  # 确保选择的Li都和醚氧相互作用
#                 o_resids = oe_atoms[close_oe_index].resids  # 找到醚氧所在的链
#
#                 # 检查所有的醚氧和同一个聚合物相互作用
#                 if np.all(o_resids == o_resids[0]):
#                     oe_ave_n[ts.frame, n] = np.mean(close_oe_index)  # 锂周围氧的平均索引
#                     poly_n[ts.frame, n] = o_resids[0]  # 锂所在链的索引
#                     bound_o_n[ts.frame, n, :len(close_oe_index)] = close_oe_index  # bound氧的索引
#                     mol = u.select_atoms(f'resid {o_resids[0]}')
#                     bound_o_positions[ts.frame, close_oe_index, :] = (oe_atoms.positions[close_oe_index] - mol.center_of_mass())  # 醚氧的坐标
#
#                 else:
#                     poly_n[ts.frame, n] = -1
#
#             if len(close_tfsi_index) > 0:
#                 tfsi_n[ts.frame, n] = -2  # Ensure poly_n also gets -2 for consistency with tfsi interactions
#
#     return oe_ave_n, poly_n, bound_o_n, bound_o_positions, times
#
#
# def compute_tau3(poly_n, run_start, run_end, dt, dt_collection, num_li, ):
#
#     t_max = (run_end - run_start) * dt_collection * dt / 1000 # ns
#     backjump_threshold = 100 / (dt_collection * dt)  # 100 ps内的跳回算作短暂跳跃
#
#     hopping_counts = [0] * num_li   # 用于记录每个锂离子的跳跃次数
#     potential_hops = {}    # 用于记录每个锂离子的最后跳跃时间和链
#     last_bound_chains = [None] * num_li  # 用于记录每个锂离子在模拟开始时的绑定链，初始为None
#
#     for i in range(num_li):
#         for t in range(run_start, run_end):
#             li_bound_current_chain = poly_n[t, i]
#
#             # 首次从未绑定转为绑定
#             if last_bound_chains[i] is None and li_bound_current_chain not in [0, -1,]:
#                 last_bound_chains[i] = li_bound_current_chain
#
#             # 检查是否为有效跳跃
#             elif last_bound_chains[i] is not None and li_bound_current_chain != last_bound_chains[i] and li_bound_current_chain not in [0, -1, -2]:
#                 if i not in potential_hops or potential_hops[i]['chain'] != li_bound_current_chain:
#                     potential_hops[i] = {'time': t, 'chain': li_bound_current_chain}
#
#                 if i in potential_hops:
#                     elapsed_time = t - potential_hops[i]['time']
#
#                     if elapsed_time >= backjump_threshold:  # 100 ps内的跳回算作短暂跳跃
#                         # 确认跳跃并更新计数
#                         hopping_counts[i] += 1
#                         last_bound_chains[i] = li_bound_current_chain
#                         del potential_hops[i]
#
#     # 输出每个锂离子的跳跃次数和总跳跃次数
#     total_hops = sum(hopping_counts)
#     tau3 = t_max * num_li / total_hops
#
#     return total_hops, tau3
#
#
# def compute_dn_msd(oe_ave_n, poly_n, run_start, run_end, time_window, dt, dt_collection, threshold=0.05):
#
#     # obtain the time array
#     times = np.arange(0, time_window * dt_collection, dt * dt_collection, dtype=int)
#
#     msd = []
#     for dt in tqdm(range(time_window), desc="Calculate MSD"):
#         msd_in_dt = []
#         if dt == 0:
#             msd.append(0)  # Assuming MSD at dt=0 is 0 as Δn would be 0
#             continue
#
#         for t in range(run_start, run_end - dt):
#             delta_n = oe_ave_n[t + dt] - oe_ave_n[t]
#             delta_n_square = np.square(delta_n)
#
#             # 使用掩码来找出特定条件的索引，减少数组操作
#             mask_i = (oe_ave_n[t + dt] == 0) | (oe_ave_n[t + dt] == -1)
#             mask_j = (oe_ave_n[t] == 0) | (oe_ave_n[t] == -1)
#             mask_h = poly_n[t + dt] != poly_n[t]
#
#             # 使用向量化方法计算 unbound_counts
#             mask_unbound = (poly_n[t:t + dt] != poly_n[t, None])
#             unbound_counts = np.sum(mask_unbound, axis=0)
#             mask_k = (unbound_counts / dt) > threshold
#
#             # 合并掩码并过滤 delta_n_square
#             full_mask = mask_i | mask_j | mask_h | mask_k
#             delta_n_square_filtered = delta_n_square[~full_mask]
#             if delta_n_square_filtered.size > 0:
#                 dns_ensemble_avg = np.mean(delta_n_square_filtered)
#                 msd_in_dt.append(dns_ensemble_avg)
#
#         if dt > 0:
#             msd.append(np.mean(msd_in_dt))
#
#     return np.array(msd), times
#
#
# def ms_endtoend_distance(work_dir, data_tpr_file, dcd_xtc_file, run_start, dt_collection, chains, select_atoms):
#     # Load the trajectory
#     data_tpr_file_path = os.path.join(work_dir, data_tpr_file)
#     dcd_xtc_file_path = os.path.join(work_dir, dcd_xtc_file)
#
#     u = mda.Universe(data_tpr_file_path, dcd_xtc_file_path)
#     box_size = u.dimensions[0]
#
#     # Preselecting atoms
#     oe_atoms = u.select_atoms(select_atoms['polymer'])
#     re_all = []  # inital the end to end distance list
#
#     for ts in tqdm(u.trajectory[int(run_start / dt_collection):], desc="Processing"):
#         ts_vectors = []
#
#         for mol_id in range(1, int(chains) + 1):  # Assuming 20 molecules
#             chain_indices = np.where(oe_atoms.resids == mol_id)[0]  # 获得所有聚合物链醚氧的index，并每一条链单独储存一个index
#             if len(chain_indices) > 1:  # Ensure there is more than one oxygen atom
#                 chain_coor = oe_atoms.positions[chain_indices]  # 获得每条聚合物链醚氧的坐标
#                 chain1_coor = chain_coor[1:]
#                 chain2_coor = chain_coor[:-1]
#                 b0_array = distance(chain1_coor, chain2_coor, box_size)  # 生成每个间隔醚氧的向量
#                 re_vector = np.sum(b0_array, axis=0)  # 所有向量加和
#                 re = np.linalg.norm(re_vector)  # 对向量进行模长的计算
#                 ts_vectors.append(re)
#
#         if ts_vectors:
#             ts_vectors = np.square(ts_vectors)
#             re_mean = np.mean(ts_vectors)
#             re_all.append(re_mean)
#
#     return re_all
#
#
# def plot_ms_endtoend_distance(re_all, run_start, run_end, system="PEO-LiTFSI"):
#     print(f"Mean squared end-to-end distance: {np.mean(re_all)}")
#     print(f"stddev: {np.std(re_all)}")
#
#     font_list = {"title": 20, "label": 18, "legend": 16, "ticket": 18, "data": 14}
#     linewidth = 1.5
#     color_list = ["#DF543F", "#2286A9", "#FBBF7C", "#3C3846"]
#
#     t = np.arange(run_start, run_end)
#
#     fig, ax = plt.subplots()
#     ax.plot(t, re_all, '-', linewidth=linewidth, color=color_list[2], label=system)
#
#     ax.legend(fontsize=font_list["legend"], frameon=False)
#     ax.set_xlabel(r'step', fontsize=font_list["label"])
#     ax.set_ylabel(r'Average $R{_e}{^2}$', fontsize=font_list["label"])
#     ax.tick_params(axis='x', labelsize=font_list["ticket"], direction='in')
#     ax.tick_params(axis='y', labelsize=font_list["ticket"], direction='in')
#     ax.grid(True, linestyle='--')
#     fig.set_size_inches(6, 5)
#     plt.tight_layout()
#
#
# def get_ether_oxygen_position(work_dir, data_tpr_file, dcd_xtc_file, select_atoms, run_start, run_end, dt,
#                               dt_collection):
#     # load the trajectory
#     data_tpr_file_path = os.path.join(work_dir, data_tpr_file)
#     dcd_xtc_file_path = os.path.join(work_dir, dcd_xtc_file)
#
#     u = mda.Universe(data_tpr_file_path, dcd_xtc_file_path)
#
#     # Select the atoms of interest
#     oe_atoms = u.select_atoms(select_atoms['polymer'])
#     t_total = run_end - run_start  # total simulation steps, minus equilibration time
#     times = np.arange(0, t_total * dt * dt_collection, dt * dt_collection, dtype=int)
#
#     # 初始化参数
#     chain_n = len(np.unique(oe_atoms.resids))  # 链的条数
#     oe_per_chain = len(oe_atoms) // chain_n  # 每链中的OE数目
#
#     # 初始化用于存储位置的数组
#     atom_positions = np.zeros((len(times), len(oe_atoms), 3))
#
#     # 处理每个时间步
#     for ts in tqdm(u.trajectory[run_start: run_end], desc="Processing"):  # 遍历每个链
#         for i in range(int(chain_n)):
#             # 选择当前链中的原子
#             oe_in_onechain = u.select_atoms(f'resid {i + 1}')
#             # 计算当前链的原子相对于质心的位置
#             atom_positions[ts.frame, oe_per_chain * i:oe_per_chain * (i + 1), :] = oe_atoms.positions[oe_per_chain *
#                                                                                                i:oe_per_chain * (i + 1),:] - oe_in_onechain.center_of_mass()
#
#     return atom_positions, times
#
#
# def compute_oe_msd(atom_position, times):
#     n_atoms = np.shape(atom_position)[1]
#     msd_oe = msd.calc_Lii_self(atom_position, times) / n_atoms
#     return msd_oe
#
#
# def rouse_model(t, tau_R, Re_square, N):
#     """计算 Rouse 模型的理论值，用于拟合 MSD 数据。"""
#     sum_part = sum([(1 - np.exp(-p ** 2 * t / tau_R)) / p ** 2 for p in range(1, N - 1)])
#     return (2 * Re_square / np.pi ** 2) * sum_part
#
#
# def compute_tR(re_all, times, num_oe, msd_oe):
#     """计算 Rouse 时间常数并拟合 MSD 数据。"""
#     Re_square = np.mean(re_all)  # 平均平方端到端距离
#     popt, pcov = curve_fit(lambda t, tau_R: rouse_model(t, tau_R, Re_square, num_oe), times, msd_oe)
#     tauR_fit = popt[0]  # 转换为纳秒
#
#     fit_curve = rouse_model(times, tauR_fit, Re_square, num_oe)
#
#     return tauR_fit / 1000, fit_curve
#
#
#
#
#
# def store_bound_o(u, run_start, run_end, ):
#     rc = 3.575
#     li_atoms = u.select_atoms('resname LIP and name Li')
#     oe_atoms = u.select_atoms('resname MOL and name O')
#
#     for ts in tqdm(u.trajectory[run_start: run_end], desc='processing'):
#
#         for li in li_atoms:
#             distances_vec = distance(oe_atoms.positions, li.positions, u.dimensions[0])
#             distances = np.linalg.norm(distances_vec, axis=1)
#             close_o_indices = np.where(distances < rc)[0]


import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acovf

def times_array(run, run_start, run_end, time_step=5):
    times = []
    for step, _ts in enumerate(run.trajectory[run_start:run_end]):
        times.append(step * time_step)
    return np.array(times)

def calc_acf(a_values: dict[str, np.ndarray]) -> list[np.ndarray]:
    acfs = []
    for neighbors in a_values.values():  # for _atom_id, neighbors in a_values.items():
        acfs.append(acovf(neighbors, demean=False, adjusted=True, fft=True))
    return acfs


def calc_neigh_corr(run, distance_dict, select_dict, run_start, run_end):
    acf_avg = {}
    center_atoms = run.select_atoms('resname LIP and name Li')
    for kw in distance_dict:
        acf_all = []
        for atom in tqdm(center_atoms[::]):
            distance = distance_dict.get(kw)
            assert distance is not None
            bool_values = {}
            for time_count, _ts in enumerate(run.trajectory[run_start:run_end:]):

                if kw in select_dict:
                    selection = (
                            "("
                            + select_dict[kw]
                            + ") and (around "
                            + str(distance)
                            + " index "
                            + str(atom.id - 1)
                            + ")"
                    )
                    shell = run.select_atoms(selection)
                else:
                    raise ValueError("Invalid species selection")

                # 获取这些原子所属的分子ID
                # mols = set(i.residue for i in shell)

                # for mol in mols:
                #     if str(mol.resid) not in bool_values:
                #         bool_values[str(mol.resid)] = np.zeros(int((run_end - run_start) / 1))
                #     bool_values[str(mol.resid)][time_count] = 1
                for atom_ in shell.atoms:
                    if str(atom_.id) not in bool_values:
                        bool_values[str(atom_.id)] = np.zeros(int((run_end - run_start) / 1))
                    bool_values[str(atom_.id)][time_count] = 1

            acfs = calc_acf(bool_values)
            acf_all.extend(list(acfs))
        acf_avg[kw] = np.mean(acf_all, axis=0)
    return acf_avg

def biexponential_func(t, a, tau_res, beta, tau_short):
    return a * np.exp(-(t / tau_res) ** beta) + (1 - a) * np.exp(-t / tau_short)

def fit_residence_time(times, acf_avg_dict, cutoff_time, time_step):
    acf_avg_norm = {}
    popt = {}
    pcov = {}
    tau = {}
    species_list = list(acf_avg_dict.keys())

    # Exponential fit of solvent-Li ACF
    for kw in species_list:
        acf_avg_norm[kw] = acf_avg_dict[kw] / acf_avg_dict[kw][0]

        popt[kw], pcov[kw] = curve_fit(
            biexponential_func,
            times[:cutoff_time],
            acf_avg_norm[kw][:cutoff_time],
            p0=(0.5, 100, 1, 10)
        )
        tau[kw] = popt[kw][1]  # ps

    # Plot ACFs
    colors = ["b", "g", "r", "c", "m", "y"]
    line_styles = ["-", "--", "-.", ":"]
    for i, kw in enumerate(species_list):
        plt.plot(times, acf_avg_norm[kw], label=kw, color=colors[i])
        fitted_x = np.linspace(0, cutoff_time * time_step, cutoff_time)
        fitted_y = biexponential_func(np.linspace(0, cutoff_time * time_step, cutoff_time), *popt[kw])
        save_decay = np.vstack(
            (
                times[:cutoff_time],
                acf_avg_norm[kw][:cutoff_time],
                fitted_x,
                fitted_y,
            )
        )

        plt.plot(
            fitted_x,
            fitted_y,
            line_styles[i],
            color="k",
            label=kw + " Fit",
        )

    plt.xlabel("Time (ps)")
    plt.legend()
    plt.ylabel("Neighbor Auto-correlation Function")
    plt.ylim(0, 1)
    plt.xlim(0, cutoff_time * time_step)
    plt.show()

    return tau





















