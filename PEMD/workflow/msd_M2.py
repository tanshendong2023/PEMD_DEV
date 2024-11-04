#!/usr/bin/env python


import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PEMD.analysis import coordination
from concurrent.futures import ProcessPoolExecutor, as_completed


def distance(x0, x1, box_length):
    delta = x1 - x0
    delta = np.where(delta > 0.5 * box_length, delta - box_length, delta)
    delta = np.where(delta < -0.5 * box_length, delta + box_length, delta)
    return delta

def load_data_traj(run, cations, polymers, num_cation, run_start, run_end, cutoff_radii):

    oe_ave_n = np.zeros((run_end - run_start, num_cation))  # 初始化锂周围氧的平均索引
    li_positions = np.zeros((run_end - run_start, num_cation, 3))  # 初始化锂离子坐标的数组

    for ts in tqdm(run.trajectory[run_start: run_end], desc='processing'):

        li_positions[ts.frame, :, :] = cations.positions[:,:]

        box_size = ts.dimensions[0]

        for n, li in enumerate(cations):

            distances_oe_vec = distance(polymers.positions, li.position, box_size)
            distances_oe = np.linalg.norm(distances_oe_vec, axis=1)
            close_oe_index = np.where(distances_oe < cutoff_radii)[0]

            if len(close_oe_index) > 0:  # 确保选择的Li都和醚氧相互作用
                o_resids = polymers[close_oe_index].resids  # 找到醚氧所在的链

                # 检查所有的醚氧和同一个聚合物相互作用
                if np.all(o_resids == o_resids[0]):
                    oe_ave_n[ts.frame, n] = np.mean(close_oe_index)  # 锂周围氧的平均索引

    return oe_ave_n, li_positions


def compute_delta_n_square(dt, li_positions, oe_ave_n, run_start, run_end, threshold):
    msd_in_dt = []
    count_li_dt = []
    if dt == 0:
        return 0, 0  # MSD at dt=0 is 0 as Δn would be 0
    for t in range(run_start, run_end - dt):
        delta_n = li_positions[t + dt] - li_positions[t]
        delta_n_square = np.sum(np.square(delta_n), axis=1)

        i = (np.abs(oe_ave_n[dt + t] - oe_ave_n[t]) <= 1) & (oe_ave_n[t] != 0)
        bound_counts = np.sum(np.abs(oe_ave_n[t:t + dt] - oe_ave_n[t]) <= 1, axis=0)
        j = (bound_counts / dt) >= threshold
        li_intersection = i & j

        count_li_dt.append(np.sum(li_intersection))

        delta_n_square_filtered = delta_n_square[li_intersection]

        if delta_n_square_filtered.size > 0:
            msd_in_dt.append(np.mean(delta_n_square_filtered))

    return np.mean(msd_in_dt) if msd_in_dt else 0, np.mean(count_li_dt) if count_li_dt else 0


def compute_dn_msd_parallel(li_positions, oe_ave_n, run_start, run_end, time_window, dt, dt_collection, threshold=0.85):
    times = np.arange(0, time_window * dt * dt_collection, dt * dt_collection, dtype=int)
    msd = []
    count_li = []
    with ProcessPoolExecutor() as executor:
        # 使用字典存储future以保持顺序
        futures = {
            executor.submit(compute_delta_n_square, dt, li_positions, oe_ave_n, run_start, run_end, threshold): dt for
            dt in range(time_window)}
        # 确保结果按照dt顺序处理
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculate MSD"):
            dt = futures[future]
            msd_result, count_li_result = future.result()
            msd.append((dt, msd_result))
            count_li.append((dt, count_li_result))
    # 按dt排序并提取MSD值
    msd.sort(key=lambda x: x[0])
    count_li.sort(key=lambda x: x[0])
    return np.array([result for _, result in msd]), np.array([result for _, result in count_li]), times


if __name__ == '__main__':
    #  loads and processes trajectory data
    work_dir = './'
    tpr_file = 'nvt_prod.tpr'
    xtc_wrap_file = 'nvt_prod.xtc'
    xtc_unwrap_file = 'nvt_prod_unwrap.xtc'
    run_start = 0
    run_end = 80001  # step
    dt = 0.001
    dt_collection = 5000  # step
    time_window = 4001

    # Load the trajectory
    run_wrap = coordination.load_md_trajectory(work_dir, tpr_file, xtc_wrap_file)
    run_unwrap = coordination.load_md_trajectory(work_dir, tpr_file, xtc_unwrap_file)

    # Select the atoms of interest
    cations_wrap = run_wrap.select_atoms('resname LIP and name Li')
    polymers_wrap = run_wrap.select_atoms('resname MOL and name O')
    anions_wrap = run_wrap.select_atoms('resname NSC and name OBT')

    cations_unwrap = run_unwrap.select_atoms('resname LIP and name Li')
    polymers_unwrap = run_unwrap.select_atoms('resname MOL and name O')
    anions_unwrap = run_unwrap.select_atoms('resname NSC and name OBT')

    volume = run_unwrap.coord.volume
    num_cation = len(cations_unwrap)

    # Perform RDF and coordination number calculation
    bins_peo, rdf_peo, coord_num_peo = coordination.calc_rdf_coord(cations_wrap, polymers_wrap, volume)
    r_li_peo, y_coord_peo = coordination.obtain_rdf_coord(bins_peo, rdf_peo, coord_num_peo)

    oe_ave_n, li_positions = load_data_traj(run_unwrap, cations_unwrap, polymers_unwrap, num_cation, run_start,
                                            run_end, r_li_peo)

    msd, count_li, times = compute_dn_msd_parallel(li_positions, oe_ave_n, run_start, run_end, time_window, dt,
                                                   dt_collection)

    # 创建包含三个列的DataFrame
    df = pd.DataFrame({
        'Time': times,
        'MSD': msd,
        'Count_Li': count_li
    })

    # 写入CSV文件
    df.to_csv('msd_M2.csv', index=False)

    # Print success message
    print("CSV file has been successfully saved!")






