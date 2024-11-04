#!/usr/bin/env python


import os
import math
import numpy as np
import MDAnalysis as mda
from tqdm.auto import tqdm
from PEMD.analysis import residence_time, coordination
from concurrent.futures import ProcessPoolExecutor, as_completed


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
time_window = 2001
dt = 0.001
dt_collection = 5000 # step
num_li = 50
num_oe = 50
chains = 20


def distance(x0, x1, box_length):
    delta = x1 - x0
    delta = np.where(delta > 0.5 * box_length, delta - box_length, delta)
    delta = np.where(delta < -0.5 * box_length, delta + box_length, delta)
    return delta


def load_data_traj(work_dir, data_tpr_file, dcd_xtc_file, select_atoms, run_start, run_end, cutoff_radii):

    # load trajectory
    data_tpr_file_path = os.path.join(work_dir, data_tpr_file)
    dcd_xtc_file_path = os.path.join(work_dir, dcd_xtc_file)

    u = mda.Universe(data_tpr_file_path, dcd_xtc_file_path)

    li_atoms = u.select_atoms(select_atoms['cation'])
    oe_atoms = u.select_atoms(select_atoms['polymer'])

    oe_ave_n = np.zeros((run_end - run_start, len(li_atoms)))  # 初始化锂周围氧的平均索引
    poly_n = np.zeros((run_end - run_start, len(li_atoms)))  # 初始化锂所在聚合物链索引的数组

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
                    poly_n[ts.frame, n] = o_resids[0]  # 锂所在链的索引

                else:
                    poly_n[ts.frame, n] = -1

    return oe_ave_n, poly_n


def compute_delta_n_square(dt, oe_ave_n, poly_n, run_start, run_end, threshold):
    msd_in_dt = []
    if dt == 0:
        return 0  # MSD at dt=0 is 0 as Δn would be 0
    for t in range(run_start, run_end - dt):
        delta_n = oe_ave_n[t + dt] - oe_ave_n[t]
        delta_n_square = np.square(delta_n)

        mask_i = (oe_ave_n[t + dt] == 0) | (oe_ave_n[t + dt] == -1)
        mask_j = (oe_ave_n[t] == 0) | (oe_ave_n[t] == -1)
        mask_h = poly_n[t + dt] != poly_n[t]

        mask_unbound = (poly_n[t:t + dt] != poly_n[t, None])
        unbound_counts = np.sum(mask_unbound, axis=0)
        mask_k = (unbound_counts / dt) > threshold

        full_mask = mask_i | mask_j | mask_h | mask_k
        delta_n_square_filtered = delta_n_square[~full_mask]
        if delta_n_square_filtered.size > 0:
            msd_in_dt.append(np.mean(delta_n_square_filtered))

    return np.mean(msd_in_dt) if msd_in_dt else 0

def compute_dn_msd_parallel(oe_ave_n, poly_n, run_start, run_end, time_window, dt, dt_collection, threshold=0.05):
    times = np.arange(0, time_window * dt * dt_collection, dt * dt_collection, dtype=int)
    msd = []
    with ProcessPoolExecutor() as executor:
        # 使用字典存储future以保持顺序
        futures = {executor.submit(compute_delta_n_square, dt, oe_ave_n, poly_n, run_start, run_end, threshold): dt for dt in range(time_window)}
        # 确保结果按照dt顺序处理
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculate MSD"):
            dt = futures[future]
            msd_result = future.result()
            msd.append((dt, msd_result))
    # 按dt排序并提取MSD值
    msd.sort(key=lambda x: x[0])
    return np.array([result for _, result in msd]), times


def power_law_model(t, slope, intercept):
    """模型函数：使用幂律关系计算MSD."""
    return np.exp(intercept) * t ** slope


def extrapolate_msd(times, msd, tau3, N):
    # 过滤非正值
    valid_indices = (times > 0) & (msd > 0)
    times_filtered = times[valid_indices]
    msd_filtered = msd[valid_indices]

    # 对数据进行对数变换
    log_times = np.log(times_filtered)
    log_msd = np.log(msd_filtered)

    # 直接计算截距 log(C)
    intercept = np.mean(log_msd - 0.8 * log_times)  # 采用固定的斜率 0.8

    # 计算外推时间点，将秒转换为毫秒
    t_extrap = tau3 * 1000

    # 使用幂律模型外推MSD
    msd_extrapolated = power_law_model(t_extrap, 0.8, intercept)

    # 计算外推点的斜率
    slope_at_t_extrap = np.exp(intercept) * 0.8 * t_extrap ** (0.8 - 1)

    # 计算D1和τ1
    D1 = slope_at_t_extrap / 2
    tau1 = ((N - 1) ** 2) / (math.pi ** 2 * D1) / 1000  # 将结果转换为ns

    return tau1


# Main function to load and process data
if __name__ == "__main__":
    #  loads and processes trajectory data
    oe_ave_n, poly_n = load_data_traj(work_dir, data_tpr_file, dcd_xtc_file, select_atoms, run_start, run_end,
                                      cutoff_radii,)

    total_hops, tau3 = residence_time.compute_tau3(poly_n, run_start, run_end, dt, dt_collection, num_li,)

    dn_msd, times = compute_dn_msd_parallel(oe_ave_n, poly_n, run_start, run_end, time_window, dt, dt_collection)

    tau1 = extrapolate_msd(times, dn_msd, tau3, num_oe)

    # Define the path for the results file
    results_file_path = os.path.join(work_dir, 'tau1-3.txt')

    # Write results to a text file
    with open(results_file_path, 'w') as file:
        file.write(f"Calculated τ1: {tau1:.2f} ns\n")
        file.write(f"Calculated τ3: {tau3:.2f} ns\n")

    print(f"Results saved to {results_file_path}")






