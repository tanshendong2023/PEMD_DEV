# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import numpy as np
from tqdm.auto import tqdm
from PEMD.analysis import coordination

def distance(x0, x1, box_length):
    delta = x1 - x0
    delta = np.where(delta > 0.5 * box_length, delta - box_length, delta)
    delta = np.where(delta < -0.5 * box_length, delta + box_length, delta)
    return delta

def load_data_traj(run, cations, polymers, anions, run_start, run_end, cutoff_peo, cutoff_tfsi, plasticizers=None, cutoff_sn=None):

    poly_n = np.zeros((run_end - run_start, len(cations)), dtype=int)  # 初始化数组存储环境编码
    for t, ts in enumerate(tqdm(run.trajectory[run_start:run_end], desc='Processing trajectory')):
        box_size = ts.dimensions[0]
        for n, li in enumerate(cations):
            # 计算锂离子与聚合物的距离
            distances_oe_vec = distance(polymers.positions, li.position, box_size)
            distances_oe = np.linalg.norm(distances_oe_vec, axis=1)
            close_oe_index = np.where(distances_oe <= cutoff_peo)[0]

            distances_anion_vec = distance(anions.positions, li.position, box_size)
            distances_anion = np.linalg.norm(distances_anion_vec, axis=1)

            # 根据聚合物的接近情况分类处理
            if close_oe_index.size > 0:
                o_resids = polymers[close_oe_index].resids
                poly_n[ts.frame, n] = o_resids[0] if np.all(o_resids == o_resids[0]) else -1
            else:
                # 当塑化剂信息可用时，考虑塑化剂的距离
                if plasticizers is not None and cutoff_sn is not None:
                    distances_plasticizer_vec = distance(plasticizers.positions, li.position, box_size)
                    distances_plasticizer = np.linalg.norm(distances_plasticizer_vec, axis=1)

                    if np.any(distances_anion <= cutoff_tfsi) and np.all(distances_plasticizer > cutoff_sn):
                        poly_n[ts.frame, n] = -2
                    elif np.any(distances_plasticizer <= cutoff_sn) and np.all(distances_anion > cutoff_tfsi):
                        poly_n[ts.frame, n] = -3
                    elif np.any(distances_anion <= cutoff_tfsi) and np.any(distances_plasticizer <= cutoff_sn):
                        poly_n[ts.frame, n] = -4
                elif np.any(distances_anion <= cutoff_tfsi):
                    poly_n[ts.frame, n] = -2  # 无塑化剂时仅考虑阴离子距离

    return poly_n

def count_jumps(data, jump_values, threshold=20):
    results = []
    for ion_index, sequence in enumerate(data.T):  # Iterate over each cation's time series
        state = 'seeking_start'
        positive_count = 0
        negative_count = 0
        jump_counts = {value: 0 for value in jump_values}
        jump_start_index = None
        last_positive_value = None  # Store the last positive value before the jump
        current_positive_value = None  # Store the positive value after the jump

        for index, value in enumerate(sequence):
            if state == 'seeking_start':
                if value > 0:
                    positive_count += 1
                    last_positive_value = value  # Update the last seen positive value
                else:
                    if positive_count >= threshold:
                        state = 'recording_negative'
                        negative_count = 1  # Reset for the upcoming negative sequence
                    positive_count = 0  # Reset counter if a non-positive is found

            elif state == 'recording_negative':
                if value <= 0:
                    negative_count += 1
                    if value in jump_values:
                        jump_counts[value] += 1
                else:
                    if negative_count >= threshold:
                        state = 'confirming_end'
                        positive_count = 1  # Start counting the final positive sequence
                        jump_start_index = index - negative_count
                    else:
                        negative_count = 0  # Reset negative count, not a valid jump yet

            elif state == 'confirming_end':
                if value > 0:
                    positive_count += 1
                    current_positive_value = value  # Track current positive value
                else:
                    if positive_count >= threshold:
                        if last_positive_value != current_positive_value:
                            # Only confirm the jump if the positive values are different
                            results.append((ion_index, jump_start_index, index - positive_count, jump_counts.copy()))
                        state = 'seeking_start'  # Reset to start looking for the next jump
                        jump_counts = {value: 0 for value in jump_values}  # Reset counts
                    else:
                        positive_count = 0  # Reset if interrupted by a non-positive

    return results

if __name__ == '__main__':
    work_dir = './'
    tpr_file = 'nvt_prod.tpr'
    xtc_wrap_file = 'nvt_prod.xtc'
    xtc_unwrap_file = 'nvt_prod_unwrap.xtc'
    run_start = 0
    run_end = 80001

    run_wrap = coordination.load_md_trajectory(work_dir, tpr_file, xtc_wrap_file)
    run_unwrap = coordination.load_md_trajectory(work_dir, tpr_file, xtc_unwrap_file)
    volume = run_wrap.coord.volume

    cations_wrap = run_wrap.select_atoms('resname LIP and name Li')
    polymers_wrap = run_wrap.select_atoms('resname MOL and name O')
    anions_wrap = run_wrap.select_atoms('resname NSC and name OBT')
    plasticizers_wrap = run_wrap.select_atoms('resname SN and name N')

    cations_unwrap = run_unwrap.select_atoms('resname LIP and name Li')
    polymers_unwrap = run_unwrap.select_atoms('resname MOL and name O')
    anions_unwrap = run_unwrap.select_atoms('resname NSC and name OBT')
    plasticizers_unwrap = run_unwrap.select_atoms('resname SN and name N')

    # polymer
    bins_peo, rdf_peo, coord_num_peo = coordination.calc_rdf_coord(cations_wrap, polymers_wrap, volume)
    cutoff_peo = coordination.obtain_rdf_coord(bins_peo, rdf_peo, coord_num_peo)[0]

    # anion
    bins_tfsi, rdf_tfsi, coord_num_tfsi = coordination.calc_rdf_coord(cations_wrap, anions_wrap, volume)
    cutoff_tfsi = coordination.obtain_rdf_coord(bins_tfsi, rdf_tfsi, coord_num_tfsi)[0]

    # plasticizer
    bins_sn, rdf_sn, coord_num_sn = coordination.calc_rdf_coord(cations_wrap, plasticizers_wrap, volume)
    cutoff_sn = coordination.obtain_rdf_coord(bins_sn, rdf_sn, coord_num_sn)[0]

    # 假设 jump_results 已经由 count_jumps 函数生成
    jump_values = [0, -1, -2, -3, -4]
    poly_n = load_data_traj(run_wrap, cations_wrap, polymers_wrap, anions_wrap, run_start, run_end,
                            cutoff_peo, cutoff_tfsi, plasticizers_wrap, cutoff_sn)

    jump_results = count_jumps(poly_n, jump_values)
    total_counts = {value: 0 for value in jump_values}

    # Collect results
    for jump in jump_results:
        for value in jump_values:
            total_counts[value] += jump[3].get(value, 0)

    # Output to text file
    with open('jump_analysis_results.txt', 'w') as file:
        file.write("Total jump counts for each value:\n")
        for value, count in total_counts.items():
            file.write(f"Value {value}: {count}\n")

        # Optionally print the results to console as well
        print("Results are also written to 'jump_analysis_results.txt'")








