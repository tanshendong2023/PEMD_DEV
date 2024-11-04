#!/usr/bin/env python

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from tqdm import tqdm
from multiprocessing import Pool


def process_frame_range(args):
    """处理指定范围的帧，并返回结果"""
    universe, start, end, cutoff = args
    li_atoms = universe.select_atoms('name Li')
    tfsi_atoms = universe.select_atoms('name NBT or name OBT or name F1')
    cluster_type_counts = {}

    for ts in universe.trajectory[start:end]:
        dist_matrix = distance_array(li_atoms.positions, tfsi_atoms.positions, box=universe.dimensions)
        cluster_idx = {atom.index: {atom.index} for atom in li_atoms + tfsi_atoms}

        for i, atom_i in enumerate(li_atoms):
            for j, atom_j in enumerate(tfsi_atoms):
                if dist_matrix[i, j] <= cutoff:
                    connected_cluster = cluster_idx[atom_i.index].union(cluster_idx[atom_j.index])
                    for k in connected_cluster:
                        cluster_idx[k] = connected_cluster

        clusters = list({frozenset(v) for v in cluster_idx.values()})
        clusters = merge_clusters(clusters, tfsi_atoms)
        update_cluster_counts(clusters, li_atoms, tfsi_atoms, cluster_type_counts)

    return cluster_type_counts

def merge_clusters(clusters, ion_atoms):
    # 创建一个离子索引到团簇索引的映射
    ion_to_clusters = {}
    for atom in ion_atoms:
        for i, cluster in enumerate(clusters):
            if atom.index in cluster:
                if atom.resid not in ion_to_clusters:
                    ion_to_clusters[atom.resid] = set()
                ion_to_clusters[atom.resid].add(i)

    # 合并包含同一离子的团簇
    merged_clusters = []
    merged = set()
    for clusters_indices in ion_to_clusters.values():
        new_cluster = set()
        for idx in clusters_indices:
            if idx not in merged:
                new_cluster.update(clusters[idx])
                merged.add(idx)
        if new_cluster:
            merged_clusters.append(frozenset(new_cluster))

    # 添加未改变的团簇
    for i, cluster in enumerate(clusters):
        if i not in merged:
            merged_clusters.append(cluster)

    return merged_clusters

def update_cluster_counts(clusters, li_atoms, tfsi_atoms, cluster_type_counts):
    # 将每个TFSI原子索引映射到其对应的残基（离子）
    tfsi_index_to_residue = {atom.index: atom.resid for atom in tfsi_atoms}

    for cluster in clusters:
        li_count = len(cluster.intersection(set(li_atoms.indices)))
        # 查找团簇中的所有TFSI原子
        tfsi_atoms_in_cluster = cluster.intersection(set(tfsi_atoms.indices))
        # 将这些原子映射到它们的残基，并计数唯一残基
        tfsi_residues_in_cluster = {tfsi_index_to_residue[idx] for idx in tfsi_atoms_in_cluster}

        # # 调试：输出当前团簇中发现的TFSI残基
        # print(f"团簇: {cluster}")
        # print(f"团簇中的TFSI原子: {tfsi_atoms_in_cluster}")
        # print(f"团簇中的TFSI残基: {tfsi_residues_in_cluster}")

        tfsi_count = len(tfsi_residues_in_cluster)
        cluster_type = (li_count, tfsi_count)
        # print(cluster_type)

        if cluster_type not in cluster_type_counts:
            cluster_type_counts[cluster_type] = 0
        cluster_type_counts[cluster_type] += 1

    return cluster_type_counts


def combine_counts(counts_list):
    """合并多个计数结果"""
    combined_counts = {}
    for counts in counts_list:
        for key, value in counts.items():
            combined_counts[key] = combined_counts.get(key, 0) + value
    return combined_counts


# def calculate_cluster_frequencies(cluster_counts, total_frames):
#     """计算团簇的频率"""
#     return {k: v / total_frames for k, v in cluster_counts.items()}

def calculate_cluster_frequencies(cluster_counts):
    """计算基于所有团簇出现总次数的团簇频率"""
    total_clusters = sum(cluster_counts.values())  # 计算所有团簇的出现总次数
    return {k: v / total_clusters for k, v in cluster_counts.items()}  # 计算每种团簇类型的频率


def process_trajectory_parallel(universe, run_start, run_end, cutoff=3.25, num_workers=52):
    num_frames = run_end - run_start
    frame_ranges = [
        (universe, run_start + i * num_frames // num_workers, run_start + (i + 1) * num_frames // num_workers, cutoff)
        for i in range(num_workers)]

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_frame_range, frame_ranges), total=num_workers))

    combined_counts = combine_counts(results)
    cluster_frequencies = calculate_cluster_frequencies(combined_counts,)

    return cluster_frequencies


# Usage
u = mda.Universe('nvt_prod.tpr', 'nvt_prod.xtc')
cluster_frequencies = process_trajectory_parallel(u, 0, 80001)
print(cluster_frequencies)

# 使用repr()将字典转换为字符串
frequencies_str = repr(cluster_frequencies)

# 写入文件
file_path = 'cluster_type_frequencies.txt'
with open(file_path, 'w') as file:
    file.write(frequencies_str)