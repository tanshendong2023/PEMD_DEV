# Description: This script is used to calculate the polymer-ion dynamics in the PEMD simulations.

import math
import numpy as np
from tqdm.auto import tqdm
from scipy.optimize import curve_fit

from PEMD.analysis.utils import minimum_image_displacement

def process_traj(
    run,
    times,
    run_start,
    run_end,
    num_cation,
    num_o_polymer,
    cutoff_radius,
    cations,
    polymers,
    box_size,
    num_chain,
    num_o_chain
):

    poly_o_n = np.zeros((len(times), num_cation))  # Initialize array
    poly_n = np.zeros((len(times), num_cation))  # Initialize array
    bound_o_n = np.full((len(times), num_cation, 10), -1, dtype=int)  # 初始化bound氧的索引
    poly_o_positions = np.zeros((len(times), num_o_polymer, 3))  # 初始化氧坐标的数组

    for ts in tqdm(run.trajectory[run_start: run_end], desc='Processing trajectory'):

        for n, li in enumerate(cations):
            distances_oe_vec = minimum_image_displacement(
                polymers.positions,
                li.position,
                box_size,
            )
            distances_oe = np.linalg.norm(distances_oe_vec, axis=1)
            close_oe_index = np.where(distances_oe <= cutoff_radius)[0]

            if len(close_oe_index) > 0:
                o_resids = polymers[close_oe_index].resids
                if np.all(o_resids == o_resids[0]):
                    poly_o_n[ts.frame, n] = np.mean(close_oe_index)
                    poly_n[ts.frame, n] = o_resids[0]
                    bound_o_n[ts.frame, n, :len(close_oe_index)] = close_oe_index  # bound氧的索引
                else:
                    poly_n[ts.frame, n] = -1

        for i in range(int(num_chain)):
            oe_in_onechain = run.select_atoms(f'resid {i + 1}')  # 选择当前链中的原子
            start_idx = num_o_chain * i
            end_idx = num_o_chain * (i + 1)
            poly_o_positions[ts.frame, start_idx:end_idx, :] = (
                    polymers.positions[start_idx:end_idx, :]
                    - oe_in_onechain.center_of_mass()
            )

    return poly_o_n, poly_n, bound_o_n, poly_o_positions

def calc_tau3(dt, dt_collection, num_cation, run_start, run_end, poly_n):

    backjump_threshold = 100 / (dt * dt_collection)  # 100 ps within jumps considered transient

    hopping_counts = [0] * num_cation  # Records hopping counts for each lithium-ion
    potential_hops = {}  # Records the last hopping time and chain for each lithium-ion
    last_bound_chains = [None] * num_cation  # Records the chain each lithium-ion was last bound to

    for i in range(num_cation):
        for t in range(run_start, run_end):
            li_bound_current_chain = poly_n[t, i]

            # First transition from unbound to bound
            if last_bound_chains[i] is None and li_bound_current_chain not in [0, -1]:
                last_bound_chains[i] = li_bound_current_chain

            # Check for a valid hop
            elif last_bound_chains[i] is not None and li_bound_current_chain != last_bound_chains[
                i] and li_bound_current_chain not in [0, -1]:

                if i not in potential_hops or potential_hops[i]['chain'] != li_bound_current_chain:
                    potential_hops[i] = {'time': t, 'chain': li_bound_current_chain}

                if i in potential_hops:
                    elapsed_time = t - potential_hops[i]['time']

                    if elapsed_time >= backjump_threshold:  # Confirm and count a hop
                        hopping_counts[i] += 1
                        last_bound_chains[i] = li_bound_current_chain
                        del potential_hops[i]

    total_hops = sum(hopping_counts)
    t_max = (run_end - run_start) * dt_collection * dt / 1000  # Convert to ns
    tau3 = t_max * num_cation / total_hops if total_hops > 0 else float('inf')  # Avoid division by zero

    return tau3

def calc_delta_n_square(dt, poly_o_n, poly_n, run_start, run_end):
    """Calculate mean squared displacement for the given time difference dt."""
    msd_in_dt = []
    if dt == 0:
        return 0  # MSD at dt=0 is 0 as Δn would be 0
    for t in range(run_start, run_end - dt):
        delta_n = poly_o_n[t + dt] - poly_o_n[t]
        delta_n_square = np.square(delta_n)

        mask_i = (poly_o_n[t + dt] == 0) | (poly_o_n[t + dt] == -1)
        mask_j = (poly_o_n[t] == 0) | (poly_o_n[t] == -1)
        mask_h = poly_n[t + dt] != poly_n[t]

        mask_unbound = (poly_n[t:t + dt] != poly_n[t, None])
        unbound_counts = np.sum(mask_unbound, axis=0)
        mask_k = (unbound_counts / dt) > 0.05

        full_mask = mask_i | mask_j | mask_h | mask_k
        delta_n_square_filtered = delta_n_square[~full_mask]
        if delta_n_square_filtered.size > 0:
            msd_in_dt.append(np.mean(delta_n_square_filtered))

    return np.mean(msd_in_dt) if msd_in_dt else 0

def calc_tau1(tau3, times_M1, msd_M1, num_o_chain):
    valid_indices = (times_M1 > 0) & (msd_M1 > 0)
    times_filtered = times_M1[valid_indices]
    msd_filtered = msd_M1[valid_indices]

    log_times = np.log(times_filtered)
    log_msd = np.log(msd_filtered)

    intercept = np.mean(log_msd - 0.8 * log_times)

    t_extrap = tau3 * 1000

    slope_at_t_extrap = np.exp(intercept) * 0.8 * t_extrap ** (0.8 - 1)
    D1 = slope_at_t_extrap / 2
    tau1 = ((num_o_chain - 1) ** 2) / (math.pi ** 2 * D1) / 1000

    return tau1

def ms_endtoend_distance(run, num_chain, polymers_unwrap, box_size, run_start, run_end,):

    re_all = []
    for ts in tqdm(run.trajectory[run_start: run_end], desc="Calculating end-to-end distance"):
        ts_vectors = []

        for mol_id in range(1, int(num_chain) + 1):  # Assuming 20 molecules
            chain_indices = np.where(polymers_unwrap.resids == mol_id)[0]  # 获得所有聚合物链醚氧的index，并每一条链单独储存一个index
            if len(chain_indices) > 1:  # Ensure there is more than one oxygen atom
                chain_coor = polymers_unwrap.positions[chain_indices]  # 获得每条聚合物链醚氧的坐标
                chain1_coor = chain_coor[1:]
                chain2_coor = chain_coor[:-1]
                b0_array = minimum_image_displacement(
                    chain1_coor,
                    chain2_coor,
                    box_size,
                )  # 生成每个间隔醚氧的向量
                re_vector = np.sum(b0_array, axis=0)  # 所有向量加和
                re = np.linalg.norm(re_vector)  # 对向量进行模长的计算
                ts_vectors.append(re)

        if ts_vectors:
            ts_vectors = np.square(ts_vectors)
            re_mean = np.mean(ts_vectors)
            re_all.append(re_mean)

    return re_all

def calc_msd_M2(dt, poly_o_positions, poly_o_n, bound_o_n, run_start, run_end,):
    msd_in_t = []
    if dt == 0:
        return 0
    for t in range(run_start, run_end - dt):
        delta_n = poly_o_positions[t + dt] - poly_o_positions[t]
        delta_n_square = np.sum(np.square(delta_n), axis=1)

        i = np.where((np.abs(poly_o_n[dt + t] - poly_o_n[t]) <= 1) & (poly_o_n[t] != 0))[0]

        bound_counts = np.sum(np.abs(poly_o_n[t:t + dt] - poly_o_n[t]) <= 1, axis=0)
        j = np.where((bound_counts / dt) >= 0.85)[0]
        li_intersection = np.intersect1d(i, j)

        all_bound_oe_indices = []  # 创建一个列表来收集所有有效的索引
        for idx in li_intersection:
            valid_indices = bound_o_n[t, idx][bound_o_n[t, idx] != -1]
            if valid_indices.size > 0:
                all_bound_oe_indices.extend(valid_indices)  # 收集所有有效的索引

        if all_bound_oe_indices:
            msd_in_t.append(np.mean(delta_n_square[all_bound_oe_indices]))

    return np.mean(msd_in_t) if msd_in_t else 0

def rouse_model(t, tau, Re_square, num_o_chain):
    """计算 Rouse 模型的理论值，用于拟合 MSD 数据。"""
    sum_part = sum([(1 - np.exp(-p ** 2 * t / tau)) / p ** 2 for p in range(1, num_o_chain - 1)])
    return (2 * Re_square / np.pi ** 2) * sum_part

def fit_rouse_model(re_all, times, msd, num_o_chain):
    """计算 Rouse 时间常数并拟合 MSD 数据。"""
    Re_square = np.mean(re_all)  # 平均平方端到端距离
    tau,_ = curve_fit(lambda t, tau: rouse_model(t, tau, Re_square, num_o_chain), times, msd)
    return tau[0] / 1000




