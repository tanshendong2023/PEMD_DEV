
# ****************************************************************************** #
#        The module implements functions to calculate the coordination           #
# ****************************************************************************** #


import os
import re
import glob
import logging
import warnings
import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt

from rdkit import Chem
from tqdm.auto import tqdm
from collections import deque
from rdkit.Geometry import Point3D
from MDAnalysis.analysis import rdf
from PEMD.model import polymer, model_lib


warnings.filterwarnings("ignore", category=UserWarning, module='MDAnalysis.coordinates.PDB')
logging.getLogger('MDAnalysis').setLevel(logging.WARNING)


def calc_rdf_coord(group1, group2, v, nbins=200, range_rdf=(0.0, 10.0)):
    # Initialize RDF analysis
    rdf_analysis = rdf.InterRDF(group1, group2, nbins=nbins, range=range_rdf)
    rdf_analysis.run()

    # Calculate coordination numbers
    rho = group2.n_atoms / v  # Density of the second group
    bins = rdf_analysis.results.bins
    rdf_values = rdf_analysis.results.rdf
    coord_numbers = np.cumsum(4 * np.pi * bins**2 * rdf_values * np.diff(np.append(0, bins)) * rho)

    return bins, rdf_values, coord_numbers

def obtain_rdf_coord(bins, rdf, coord_numbers):

    deriv_sign_changes = np.diff(np.sign(np.diff(rdf)))
    peak_index = np.where(deriv_sign_changes < 0)[0] + 1
    if len(peak_index) == 0:
        raise ValueError("No peak found in RDF data.")
    first_peak_index = peak_index[0]

    min_after_peak_index = np.where(deriv_sign_changes[first_peak_index:] > 0)[0] + first_peak_index + 1
    if len(min_after_peak_index) == 0:
        raise ValueError("No minimum found after the first peak in RDF data.")
    first_min_index = min_after_peak_index[0]

    x_val = round(float(bins[first_min_index]), 3)
    y_coord = round(float(np.interp(x_val, bins, coord_numbers)), 3)

    return x_val, y_coord

def load_md_trajectory(work_dir, tpr_filename='nvt_prod.tpr', xtc_filename='nvt_prod.xtc'):
    data_tpr_file = os.path.join(work_dir, tpr_filename)
    data_xtc_file = os.path.join(work_dir, xtc_filename)
    u = mda.Universe(data_tpr_file, data_xtc_file)
    return u

def distance(x0, x1, box_length):
    """Calculate minimum image distance accounting for periodic boundary conditions."""
    delta = x1 - x0
    delta = np.where(delta > 0.5 * box_length, delta - box_length, delta)
    delta = np.where(delta < -0.5 * box_length, delta + box_length, delta)
    return delta

def analyze_coordination(universe, li_atoms, molecule_groups, cutoff_radii, run_start, run_end):
    num_timesteps = run_end - run_start
    num_li_atoms = len(li_atoms)
    coordination = np.zeros((num_timesteps, num_li_atoms), dtype=int)

    for ts_index, ts in enumerate(tqdm(universe.trajectory[run_start:run_end], desc='Processing')):
        box_size = ts.dimensions[0:3]
        for li_index, li in enumerate(li_atoms):
            encoded_coordination = 0
            factor = 10**(len(molecule_groups) - 1)  # Factor for encoding counts at different decimal places
            for group_name, group_atoms in molecule_groups.items():
                d_vec = distance(group_atoms.positions, li.position, box_size)
                d = np.linalg.norm(d_vec, axis=1)
                close_atoms_index = np.where(d < cutoff_radii[group_name])[0]
                unique_resids = len(np.unique(group_atoms[close_atoms_index].resids))
                encoded_coordination += unique_resids * factor
                factor //= 10  # Increment factor for the next group encoding
            coordination[ts_index, li_index] = encoded_coordination

    return coordination

def plot_rdf_coordination(
    bins,
    rdf,
    coord_numbers,
):
    # 字体和配色
    font_list = {"label": 18, "ticket": 18, "legend": 16}
    color_list = ["#DF543F", "#2286A9", "#FBBF7C", "#3C3846"]

    # 创建画布
    fig, ax1 = plt.subplots()
    fig.set_size_inches(5.5, 4)

    # 绘制 RDF 曲线
    ax1.plot(
        bins,
        rdf,
        '-',
        linewidth=1.5,
        color=color_list[0],
        label='g(r)'
    )
    ax1.set_xlabel('Distance (Å)', fontsize=font_list["label"])
    ax1.set_ylabel('g(r)', fontsize=font_list["label"])
    ax1.tick_params(
        axis='both',
        which='both',
        direction='in',
        labelsize=font_list["ticket"]
    )

    # 创建第二个 y 轴，绘制配位数
    ax2 = ax1.twinx()
    ax2.plot(
        bins,
        coord_numbers,
        '--',
        linewidth=2,
        color="grey",
        label='Coord. Number'
    )
    ax2.set_ylabel('Coordination Number', fontsize=font_list["label"])
    ax2.tick_params(
        axis='y',
        which='both',
        direction='in',
        labelsize=font_list["ticket"]
    )

    # 坐标轴范围 & 网格
    ax1.set_xlim(0, 10)
    ax1.grid(True, linestyle='--')

    plt.tight_layout()
    plt.show()


def num_of_neighbor(
        work_dir,
        nvt_run,
        center_atom_name,
        distance_dict,
        select_dict,
        run_start,
        run_end,
        write,
        structure_code,
        write_freq,
        max_number
):
    # build the dir to store the cluster file
    write_path = os.path.join(work_dir, 'cluster_dir')
    os.makedirs(write_path, exist_ok=True)

    center_atoms = nvt_run.select_atoms(center_atom_name)
    trj_analysis = nvt_run.trajectory[run_start:run_end:10]
    cn_values = {}
    species = list(distance_dict.keys())

    for kw in species:
        cn_values[kw] = np.zeros(int(len(trj_analysis)))
    cn_values["total"] = np.zeros(int(len(trj_analysis)))

    written_structures = 0
    max_reached = False  # Initialize the flag variable

    for time_count, ts in enumerate(trj_analysis):
        if max_reached:
            break  # Exit the loop if max_number is reached
        for center_atom in center_atoms:
            digit_of_species = len(species) - 1
            for kw in species:
                selection = select_shell(select_dict, distance_dict, center_atom, kw)
                shell = nvt_run.select_atoms(selection, periodic=True)
                for _ in shell.atoms:
                    cn_values[kw][time_count] += 1
                    cn_values["total"][time_count] += 10 ** digit_of_species
                digit_of_species -= 1  # Simplify decrement

            if write and cn_values["total"][time_count] == structure_code:
                a = np.random.random()
                if a > 1 - write_freq:
                    selection_write = " or ".join(
                        "(same resid as (" + select_shell(select_dict, distance_dict, center_atom, kw) + "))"
                        for kw in species
                    )
                    center_resid_selection = "same resid as index " + str(center_atom.index)
                    selection_write = "((" + selection_write + ") or (" + center_resid_selection + "))"
                    structure = nvt_run.select_atoms(selection_write, periodic=True)
                    center_pos = ts[center_atom.index]
                    # path = write_path + 'num' + "_" + str(int(written_structures)) + ".xyz"
                    pdb_filename = 'num' + "_" + str(int(written_structures)) + ".pdb"
                    path = os.path.join(write_path, pdb_filename)
                    write_out(center_pos, structure, path)

                    written_structures += 1
                    if written_structures >= max_number:
                        print(f"{max_number} structures have been written out in {write_path}!!!.")
                        max_reached = True  # Set the flag to True
                        break  # Break out of the innermost loop

                if max_reached:
                    break  # Check the flag and break if needed
        if max_reached:
            break  # Check the flag and break if needed
    else:
        print(f"Target of {max_number} structures not reached; only {written_structures} were written to {write_path}!!!.")

    return cn_values


def write_out(center_pos, neighbors, path):
    # 计算相对坐标，处理周期性边界条件
    box = neighbors.dimensions
    half_box = box[:3] / 2.0
    new_positions = neighbors.positions - center_pos
    # 应用最小镜像原理
    new_positions = np.where(new_positions > half_box, new_positions - box[:3], new_positions)
    new_positions = np.where(new_positions < -half_box, new_positions + box[:3], new_positions)
    neighbors.positions = new_positions

    # 写入 PDB 文件
    neighbors.write(path)


def select_shell(
        select,
        distance,
        center_atom,
        kw
):
    if isinstance(select, dict):
        species_selection = select[kw]
        if species_selection is None:
            raise ValueError("Species specified does not match entries in the select dict.")
    else:
        species_selection = select
    if isinstance(distance, dict):
        distance_value = distance[kw]
        if distance_value is None:
            raise ValueError("Species specified does not match entries in the distance dict.")
        distance_str = str(distance_value)
    else:
        distance_str = distance
    return "(" + species_selection + ") and (around " + distance_str + " index " + str(center_atom.index) + ")"

def pdb2mol(work_dir, pdb_filename):
    label_to_element = {
        'N': 'N',
        'S': 'S',
        'O': 'O',
        'C': 'C',
        'F': 'F',
        'CL': 'Cl',
        'BR': 'Br',
        'LI': 'Li',
    }

    pdb_filepath = os.path.join(work_dir, pdb_filename)

    # 列表用于存储原子标签、坐标和残基名称
    atoms_data = []

    # 读取 PDB 文件
    with open(pdb_filepath, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                # PDB 格式列：
                # 列 13-16: 原子名称
                # 列 17: 替代位置指示符
                # 列 18-20: 残基名称
                # 列 22: 链标识符
                # 列 23-26: 残基序列号
                # 列 31-38: X 坐标
                # 列 39-46: Y 坐标
                # 列 47-54: Z 坐标
                atom_name = line[12:17].strip()
                res_name = line[17:20].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                # 从原子名称中推断元素符号，如果元素符号列为空
                element = line[76:78].strip()
                if not element:
                    # 从原子名称中提取首字母作为元素符号
                    element = ''.join([char for char in atom_name if char.isalpha()])[0]

                element = label_to_element.get(element, element)

                atoms_data.append((element, res_name, atom_name, (x, y, z)))

    num_atoms = len(atoms_data)
    if num_atoms == 0:
        print("PDB 文件中未找到原子信息。")
        return None

    # 创建一个新的 RDKit 分子
    mol = Chem.RWMol()

    # 添加原子到分子，并存储 resname 作为原子属性
    for atom_info in atoms_data:
        label, res_name, atom_name, coords = atom_info
        atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(label)
        if atomic_num == 0:
            print(f"无法识别的元素符号 '{label}'，跳过该原子。")
            continue
        atom = Chem.Atom(atomic_num)
        # 设置原子属性 'resname' 为残基名称
        atom.SetProp("resname", res_name)
        atom.SetProp("name", atom_name)
        mol.AddAtom(atom)

    # 生成 3D 构象
    conf = Chem.Conformer(num_atoms)
    for i, atom_info in enumerate(atoms_data):
        _, _, _, (x, y, z) = atom_info
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))
    mol.AddConformer(conf)

    # 添加键，基于距离和共价半径
    tolerance = 0.4  # Å
    pt = Chem.GetPeriodicTable()
    for i in range(num_atoms):
        atom_i = mol.GetAtomWithIdx(i)
        for j in range(i + 1, num_atoms):
            atom_j = mol.GetAtomWithIdx(j)
            # 计算原子之间的距离
            pos_i = np.array([conf.GetAtomPosition(i).x,
                              conf.GetAtomPosition(i).y,
                              conf.GetAtomPosition(i).z])
            pos_j = np.array([conf.GetAtomPosition(j).x,
                              conf.GetAtomPosition(j).y,
                              conf.GetAtomPosition(j).z])
            distance = np.linalg.norm(pos_i - pos_j)
            # 获取原子的共价半径
            radius_i = pt.GetRcovalent(atom_i.GetSymbol())
            radius_j = pt.GetRcovalent(atom_j.GetSymbol())
            if radius_i == 0 or radius_j == 0:
                continue  # 如果半径不可用，则跳过
            # 检查距离是否在共价半径之和加上容差范围内
            if distance <= (radius_i + radius_j + tolerance):
                try:
                    mol.AddBond(i, j, order=Chem.rdchem.BondType.SINGLE)
                except Exception as e:
                    print(f"添加键失败：{e}")

    # 转换为 Mol 对象并进行净化
    mol = mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.KekulizeException as e:
        print(f"分子净化失败：{e}")
        return None

    return mol


def parse_selection_string(selection_str):

    # 使用正则表达式分割 'and'，并提取键值对
    conditions = re.split(r'\s+and\s+', selection_str.strip(), flags=re.IGNORECASE)
    criteria = {}
    for condition in conditions:
        match = re.match(r'(\w+)\s+(\S+)', condition)
        if match:
            key, value = match.groups()
            criteria[key.lower()] = value
        else:
            raise ValueError(f"无法解析的选择条件: '{condition}'")
    return criteria


def bfs_traverse(mol, start_idx, max_steps=None, ignore_H=False):
    """
    从 start_idx 出发做广度优先遍历，返回访问到的原子索引集合。
    - max_steps: 最大深度（None 表示不限制）
    - ignore_H: 是否跳过氢原子
    """
    visited = {start_idx}
    queue = deque([(start_idx, 0)])
    while queue:
        idx, depth = queue.popleft()
        if max_steps is not None and depth >= max_steps:
            continue
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            nid = nbr.GetIdx()
            if ignore_H and nbr.GetSymbol() == 'H':
                continue
            if nid not in visited:
                visited.add(nid)
                queue.append((nid, depth + 1))
    return visited


def select_atoms_by_criteria(mol, criteria):
    """
    根据给定的属性字典 criteria（如 {'atom_name': 'C1', 'resname': 'PEO'}）
    从 mol 中选出所有满足的 Atom.
    """
    return [
        atom for atom in mol.GetAtoms()
        if all(atom.HasProp(k) and atom.GetProp(k) == v for k, v in criteria.items())
    ]


def get_cluster_index(
    mol,
    center_atom_name,
    select_dict,
    distance_dict,
    poly_name,
    repeating_unit,
    length
):

    # 将重复单元中的 '*' 替换为 '[H]'，并创建相应的 RDKit 分子
    unit_smi_with_h1 = repeating_unit.replace('*', '[H]')
    unit_mol_with_h1 = Chem.MolFromSmiles(unit_smi_with_h1)
    unit_mol = Chem.RemoveHs(unit_mol_with_h1)
    num_unit = unit_mol.GetNumAtoms()
    max_steps = num_unit * (length - 1)

    # 获取分子的构象
    conf = mol.GetConformer()
    center_criteria = parse_selection_string(center_atom_name)
    select_criteria = parse_selection_string(select_dict[poly_name])

    # Step 1: 根据选择条件选择中心原子（假设只有一个）
    center_atoms_list = select_atoms_by_criteria(mol, center_criteria)
    center_atom = center_atoms_list[0]
    n_idx = center_atom.GetIdx()

    # 获取选定原子的列表和坐标，基于选择条件
    select_atoms_list = select_atoms_by_criteria(mol, select_criteria)
    select_positions = np.array([conf.GetAtomPosition(atom.GetIdx()) for atom in select_atoms_list])

    # Step 3: 找到与中心原子最近的选定原子，并确认键连接
    n_pos = np.array([conf.GetAtomPosition(n_idx).x,
                      conf.GetAtomPosition(n_idx).y,
                      conf.GetAtomPosition(n_idx).z])
    distances = np.linalg.norm(select_positions - n_pos, axis=1)

    if center_atom.GetSymbol() != 'Li':
        center_atom_indices = bfs_traverse(mol, n_idx, max_steps=None, ignore_H=False)
        min_dist_idx = np.argmin(distances)
        closest_select_atom = select_atoms_list[min_dist_idx]

        # 获取与最近的选定原子相连的特定原子
        bonded_c = [nbr for nbr in closest_select_atom.GetNeighbors() if nbr.GetSymbol() != 'H']
        c_idx_list = [bonded_c[0].GetIdx()]
    else:
        center_atom_indices = {n_idx}
        threshold = distance_dict[poly_name]
        mask = distances < threshold
        idxs = np.where(mask)[0]
        c_idx_list = [select_atoms_list[i].GetIdx() for i in idxs]

    # 6. BFS 限定步数找聚合物片段（忽略 H）
    selected_atom_indices = set()
    for start_idx in c_idx_list:
        selected_atom_indices |= bfs_traverse(
            mol,
            start_idx,
            max_steps=max_steps,
            ignore_H=True
        )
    selected_atom_indices = sorted(selected_atom_indices)

    # 7. 找到其他原子索引
    other_atom_indices = []
    for name, sel_str in select_dict.items():
        if name != poly_name:
            other_criteria = parse_selection_string(select_dict[name])
            comp_idxs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetProp('resname') == other_criteria['resname']]
            other_atom_indices.extend(comp_idxs)

    return sorted(c_idx_list), sorted(center_atom_indices), sorted(selected_atom_indices), sorted(other_atom_indices)


def find_poly_match_subindex(poly_name, repeating_unit, length, mol, selected_atom_idxs, c_idx_list, ):

    (
        dum1,
        dum2,
        atom1,
        atom2,
    ) = polymer.Init_info(
        poly_name,
        repeating_unit,
    )

    (
        inti_mol3,
        monomer_mol,
        start_atom,
        end_atom,
    ) = model_lib.gen_smiles_nocap(
        dum1,
        dum2,
        atom1,
        atom2,
        repeating_unit,
        length,
    )

    main_smi = Chem.MolToSmiles(inti_mol3, canonical=False)
    main_smi_with_h1 = main_smi.replace('*', '[H]')
    main_mol_with_h1 = Chem.MolFromSmiles(main_smi_with_h1)

    main_mol = Chem.RemoveHs(main_mol_with_h1)
    main_smi = Chem.MolToSmiles(main_mol, canonical=False)

    mol1 = Chem.MolFromSmiles(main_smi)
    rw_mol = Chem.RWMol(mol1)
    for bond in rw_mol.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)

    mol2 = rw_mol.GetMol()
    smi2 = Chem.MolToSmiles(mol2, canonical=False)
    pattern = Chem.MolFromSmiles(smi2)
    matches = mol.GetSubstructMatches(pattern, uniquify=True)

    best = pick_most_central_match(matches, c_idx_list, selected_atom_idxs)
    if best is not None:
        return list(best), start_atom, end_atom


def pick_most_central_match(matches, c_idx_list, selected_atom_idxs):
    best_match = None
    best_score = float("inf")
    # 这里假设所有 match 长度都相同
    L = len(matches[0]) if matches else 0
    center = (L - 1) / 2

    for match in matches:
        # 基本筛选：c_idx_list 中的每个元素都在 match 里，
        # 且 match 里的每个 idx 都在 selected_atom_idxs 里
        if not all(c in match for c in c_idx_list):
            continue
        if not all(idx in selected_atom_idxs for idx in match):
            continue

        # 因为 c_idx_list 这里只含一个元素 c
        p_c = len(c_idx_list)//2
        c = c_idx_list[p_c]
        pos = match.index(c)

        # 距离中心的“偏离度”
        score = abs(pos - center)
        if score < best_score:
            best_score = score
            best_match = match

    return best_match


def get_cluster_withcap(work_dir, mol, match_list, center_atom_idx, other_atom_indices, start_atom, end_atom, out_xyz_filename):

    # 1. 先把 center_atom_idx 规范成列表
    if isinstance(center_atom_idx, int):
        center_idxs = [center_atom_idx]
    else:
        center_idxs = list(center_atom_idx)

    # 2. 如果 other_atom_indices 是 dict，就 flatten
    if isinstance(other_atom_indices, dict):
        other_idxs = []
        for lst in other_atom_indices.values():
            other_idxs.extend(lst)
    else:
        other_idxs = list(other_atom_indices)

    h_atom_idx = set()
    for idx in match_list:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:  # H atom
                h_atom_idx.add(neighbor.GetIdx())

    all_idxs = set(match_list) | h_atom_idx | set(center_idxs) | set(other_idxs)
    select_atom_idx_with_h = sorted(all_idxs)

    # select_atom_idx_with_h = sorted(set(match_list + list(h_atom_idx) + center_atom_idx + other_atom_indices))

    new_mol = Chem.RWMol()
    index_map = {}

    for old_idx in select_atom_idx_with_h:
        atom = mol.GetAtomWithIdx(old_idx)
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_idx = new_mol.AddAtom(new_atom)
        index_map[old_idx] = new_idx

    # 创建一个反向索引映射（新分子索引到原始原子索引）
    # reverse_index_map = {v: k for k, v in index_map.items()}
    reverse_index_map = {new_idx: old_idx for old_idx, new_idx in index_map.items()}

    # 添加原子间的键（如果两个原子都在要提取的列表中）
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in select_atom_idx_with_h and end in select_atom_idx_with_h:
            new_mol.AddBond(index_map[begin], index_map[end], bond.GetBondType())

    # terminal_atoms = [start_atom, end_atom]
    start_atom_old = match_list[start_atom]
    end_atom_old = match_list[end_atom]
    start_atom_new = index_map[start_atom_old]
    end_atom_new = index_map[end_atom_old]
    terminal_atoms = [start_atom_new, end_atom_new]

    # 定义要添加的封端基团信息
    capping_info = []
    for terminal_idx in terminal_atoms:
        atom = new_mol.GetAtomWithIdx(terminal_idx)
        h_count = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1)
        if atom.GetAtomicNum() == 6 and h_count == 2:
            capping_info.append({'type': 'H', 'atom_idx': terminal_idx})
        else:
            capping_info.append({'type': 'CH3', 'atom_idx': terminal_idx})

    # 定义键长（以 Å 为单位）
    C_H_bond_length = 1.09
    C_C_bond_length = 1.50

    # 获取原始分子的构象
    orig_conf = mol.GetConformer()
    new_atom_positions = {}

    for cap in capping_info:
        terminal_idx = cap['atom_idx']
        terminal_atom = new_mol.GetAtomWithIdx(terminal_idx)

        # 获取对应的原始分子中的原子索引
        terminal_old_idx = reverse_index_map[terminal_idx]
        terminal_pos = np.array(orig_conf.GetAtomPosition(terminal_old_idx))

        # 获取连接到端基原子的非氢原子，用于计算方向
        neighbor_indices = [nbr.GetIdx() for nbr in terminal_atom.GetNeighbors()
                            if nbr.GetAtomicNum() > 1 and nbr.GetIdx() != terminal_idx]

        if neighbor_indices:
            neighbor_idx = neighbor_indices[0]
            neighbor_old_idx = reverse_index_map[neighbor_idx]
            neighbor_pos = np.array(orig_conf.GetAtomPosition(neighbor_old_idx))
            bond_vec = terminal_pos - neighbor_pos
        else:
            # 如果没有非氢原子邻居，使用默认方向
            bond_vec = np.array([0.0, 0.0, 1.0])

        # 归一化方向向量
        bond_vec = bond_vec / np.linalg.norm(bond_vec)

        # 计算垂直于 bond_vec 的两个正交向量
        perp_vec1 = np.cross(bond_vec, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp_vec1) < 1e-3:
            perp_vec1 = np.cross(bond_vec, np.array([0.0, 1.0, 0.0]))
        perp_vec1 = perp_vec1 / np.linalg.norm(perp_vec1)
        perp_vec2 = np.cross(bond_vec, perp_vec1)
        perp_vec2 = perp_vec2 / np.linalg.norm(perp_vec2)

        if cap['type'] == 'H':
            connected_h_indices = [
                nbr.GetIdx() for nbr in terminal_atom.GetNeighbors()
                if nbr.GetAtomicNum() == 1
            ]

            if len(connected_h_indices) < 2:
                raise ValueError(f"Expected at least 2 hydrogen atoms connected to atom index {terminal_idx}, found {len(connected_h_indices)}.")

            h1_old_idx = reverse_index_map.get(connected_h_indices[0])
            h2_old_idx = reverse_index_map.get(connected_h_indices[1])
            if h1_old_idx is None or h2_old_idx is None:
                raise KeyError(f"One of the hydrogen atoms ({connected_h_indices[0]}, {connected_h_indices[1]}) is missing in reverse_index_map.")

            h1_pos = np.array(orig_conf.GetAtomPosition(h1_old_idx))
            h2_pos = np.array(orig_conf.GetAtomPosition(h2_old_idx))
            h1_vec = h1_pos - terminal_pos
            h2_vec = h2_pos - terminal_pos
            mid_h_vec = (h1_vec + h2_vec) / 2
            norm_mid_h_vec = np.linalg.norm(mid_h_vec)
            if norm_mid_h_vec < 1e-6:
                # 如果 mid_h_vec 过小，使用 perp_vec1
                mid_h_vec = perp_vec1
            else:
                mid_h_vec /= norm_mid_h_vec

            # 投影 mid_h_vec 到垂直于 bond_vec 的平面上
            proj_mid_h_vec = mid_h_vec - np.dot(mid_h_vec, bond_vec) * bond_vec
            norm_proj_mid_h_vec = np.linalg.norm(proj_mid_h_vec)
            if norm_proj_mid_h_vec < 1e-6:
                # 如果投影后的向量太小，使用 perp_vec1
                proj_mid_h_vec = perp_vec1
            else:
                proj_mid_h_vec /= norm_proj_mid_h_vec

            # 计算新氢原子的方向，确保与 bond_vec 形成四面体角度
            theta = np.deg2rad(360-109.5)  # 四面体键角
            direction = (
                np.cos(theta) * (-bond_vec) +
                np.sin(theta) * proj_mid_h_vec
            )
            direction /= np.linalg.norm(direction)

            # 计算新氢原子的位置
            H_pos = terminal_pos + direction * C_H_bond_length

            # 添加新氢原子到分子中
            new_H = Chem.Atom(1)
            new_H_idx = new_mol.AddAtom(new_H)
            new_atom_positions[new_H_idx] = H_pos
            new_mol.AddBond(terminal_idx, new_H_idx, Chem.BondType.SINGLE)

        elif cap['type'] == 'CH3':
            terminal_old_idx = reverse_index_map[terminal_idx]
            terminal_old_atom = mol.GetAtomWithIdx(terminal_old_idx)
            neighbor_old_idx = [nbr.GetIdx() for nbr in terminal_old_atom.GetNeighbors()
                                if nbr.GetIdx() not in match_list]

            neighbor_old_pos = np.array(orig_conf.GetAtomPosition(neighbor_old_idx[0]))
            if neighbor_old_idx[0] > terminal_old_idx:
                bond_vec = neighbor_old_pos - terminal_pos
            else:
                bond_vec = - terminal_pos + neighbor_old_pos
            direction = bond_vec / np.linalg.norm(bond_vec)
            C_pos = terminal_pos + direction * C_C_bond_length

            new_C = Chem.Atom(6)
            new_C_idx = new_mol.AddAtom(new_C)
            new_atom_positions[new_C_idx] = C_pos
            new_mol.AddBond(terminal_idx, new_C_idx, Chem.BondType.SINGLE)

            # 添加三个氢原子，排列为正四面体
            # 计算新的键向量
            bond_vec_new = C_pos - terminal_pos
            bond_vec_new = bond_vec_new / np.linalg.norm(bond_vec_new)

            # 重新计算垂直于 bond_vec_new 的两个正交向量
            perp_vec1_new = np.cross(bond_vec_new, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(perp_vec1_new) < 1e-3:
                perp_vec1_new = np.cross(bond_vec_new, np.array([0.0, 1.0, 0.0]))
            perp_vec1_new = perp_vec1_new / np.linalg.norm(perp_vec1_new)
            perp_vec2_new = np.cross(bond_vec_new, perp_vec1_new)
            perp_vec2_new = perp_vec2_new / np.linalg.norm(perp_vec2_new)

            # 定义三个氢原子的角度（以度为单位）
            angles = [0, 120, 240]
            for angle in angles:
                theta = np.deg2rad(109.5)  # 四面体键角
                phi = np.deg2rad(angle)
                direction = (
                        np.cos(theta) * (-bond_vec_new) +
                        np.sin(theta) * (np.cos(phi) * perp_vec1_new + np.sin(phi) * perp_vec2_new)
                )
                H_pos = C_pos + direction * C_H_bond_length
                new_H = Chem.Atom(1)
                new_H_idx = new_mol.AddAtom(new_H)
                new_atom_positions[new_H_idx] = H_pos
                new_mol.AddBond(new_C_idx, new_H_idx, Chem.BondType.SINGLE)

    # 创建新的构象（Conformer）
    num_atoms_new = new_mol.GetNumAtoms()
    conf = Chem.Conformer(num_atoms_new)

    # 设置原始原子的坐标
    for old_idx, new_idx in index_map.items():
        pos = orig_conf.GetAtomPosition(old_idx)
        conf.SetAtomPosition(new_idx, pos)

    # 设置新添加的原子的坐标
    for idx, pos in new_atom_positions.items():
        conf.SetAtomPosition(idx, Point3D(*pos))

    # 将构象添加到分子中
    new_mol.AddConformer(conf)

    # 更新属性缓存并进行分子规范化
    # new_mol.UpdatePropertyCache(strict=False)
    # Chem.SanitizeMol(new_mol)

    # save to xyz file
    out_xyz_filepath = os.path.join(work_dir, out_xyz_filename)
    Chem.MolToXYZFile(new_mol, out_xyz_filepath)


def rotate_vector_around_axis(v, axis, theta):
    """
    使用Rodrigues旋转公式将向量v围绕单位向量axis旋转theta弧度。
    """
    axis = axis / np.linalg.norm(axis)
    v_parallel = np.dot(v, axis) * axis
    v_perp = v - v_parallel
    w = np.cross(axis, v)
    return v_parallel + v_perp * np.cos(theta) + w * np.sin(theta)


def merge_xyz_files(xyz_dir: str,
                            pattern: str = "num_*_frag.xyz",
                            out_name: str = "merged.xyz") -> str:

    xyz_paths = sorted(glob.glob(os.path.join(xyz_dir, pattern)))
    if not xyz_paths:
        raise FileNotFoundError(f"No files matched pattern {pattern} in {xyz_dir}")

    traj_path = os.path.join(xyz_dir, out_name)
    with open(traj_path, 'w') as fw:
        for idx, path in enumerate(xyz_paths, 1):
            with open(path, 'r') as fr:
                lines = fr.readlines()
                # 可选：覆盖注释行，写上帧编号
                lines[1] = f"Frame {idx}: {os.path.basename(path)}\n"
                fw.writelines(lines)

    print(f"Merged {len(xyz_paths)} frames into trajectory: {traj_path}")
    return traj_path