
import os
import re
import numpy as np
from collections import deque
from rdkit import Chem
from rdkit.Geometry import Point3D
from PEMD.model import model_lib


def num_of_neighbor(
        nvt_run,
        center_atom_name,
        distance_dict,
        select_dict,
        run_start,
        run_end,
        write,
        structure_code,
        write_freq,
        write_path,
        max_number
):
    center_atoms = nvt_run.select_atoms(center_atom_name)
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
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
                    path = os.path.join(write_path, 'num' + "_" + str(int(written_structures)) + ".pdb")
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

    return cn_values

# def write_out(
#         center_pos,
#         neighbors,
#         path
# ):
#     lines = []
#     lines.append(str(len(neighbors)))
#     lines.append("")
#     box = neighbors.dimensions
#     half_box = np.array([box[0], box[1], box[2]]) / 2
#     for atom in neighbors:
#         locs = []
#         for i in range(3):
#             loc = atom.position[i] - center_pos[i]
#             if loc > half_box[i]:
#                 loc = loc - box[i]
#             elif loc < -half_box[i]:
#                 loc = loc + box[i]
#             else:
#                 pass
#             locs.append(loc)
#         element_name = atom.name
#         assert element_name is not None
#         line = element_name + " " + " ".join(str(loc) for loc in locs)
#         lines.append(line)
#     with open(path, "w") as xyz_file:
#         xyz_file.write("\n".join(lines))


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
        # 根据需要添加更多元素
    }

    # PDB 文件的路径
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
        print(atom_name)
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
    print(criteria)
    return criteria

def get_cluster_index(mol,
                      center_atoms,
                      select_atoms,
                      repeating_unit,
                      length):

    # 将重复单元中的 '*' 替换为 '[H]'，并创建相应的 RDKit 分子
    unit_smi_with_h1 = repeating_unit.replace('*', '[H]')
    unit_mol_with_h1 = Chem.MolFromSmiles(unit_smi_with_h1)
    if unit_mol_with_h1 is None:
        raise ValueError("Invalid SMILES string for repeating unit.")
    unit_mol = Chem.RemoveHs(unit_mol_with_h1)
    num_unit = unit_mol.GetNumAtoms()
    max_steps = num_unit * (length - 1)

    # 获取分子的构象
    conf = mol.GetConformer()

    # 解析选择字符串
    center_criteria = parse_selection_string(center_atoms)
    select_criteria = parse_selection_string(select_atoms)
    print()

    # Step 1: 根据选择条件选择中心原子（假设只有一个）
    center_atoms_list = [
        atom for atom in mol.GetAtoms()
        if all(
            atom.HasProp(key) and atom.GetProp(key) == value
            for key, value in center_criteria.items()
        )
    ]

    if not center_atoms_list:
        raise ValueError(f"No center atoms found with criteria '{center_atoms}'.")
    if len(center_atoms_list) > 1:
        print(f"Warning: Multiple center atoms found with criteria '{center_atoms}'. Using the first one.")

    center_atom = center_atoms_list[0]
    n_idx = center_atom.GetIdx()

    # 获取选定原子的列表和坐标，基于选择条件
    select_atoms_list = [
        atom for atom in mol.GetAtoms()
        if all(
            atom.HasProp(key) and atom.GetProp(key) == value
            for key, value in select_criteria.items()
        )
    ]

    if not select_atoms_list:
        raise ValueError(f"No select atoms found with criteria '{select_atoms}'.")

    select_positions = np.array([conf.GetAtomPosition(atom.GetIdx()) for atom in select_atoms_list])

    # Step 2: 从中心原子开始，找到其所在的连通组分
    center_atom_indices = set()
    queue = deque()

    center_atom_indices.add(n_idx)
    queue.append(n_idx)
    while queue:
        current_idx = queue.popleft()
        current_atom = mol.GetAtomWithIdx(current_idx)
        for neighbor in current_atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in center_atom_indices:
                center_atom_indices.add(neighbor_idx)
                queue.append(neighbor_idx)

    # Step 3: 找到与中心原子最近的选定原子，并确认键连接
    n_pos = np.array([conf.GetAtomPosition(n_idx).x,
                      conf.GetAtomPosition(n_idx).y,
                      conf.GetAtomPosition(n_idx).z])
    distances = np.linalg.norm(select_positions - n_pos, axis=1)
    min_dist_idx = np.argmin(distances)
    closest_select_atom = select_atoms_list[min_dist_idx]

    # 获取与最近的选定原子相连的特定原子
    bonded_c_atoms = [nbr for nbr in closest_select_atom.GetNeighbors() if nbr.GetSymbol() != 'H']
    if not bonded_c_atoms:
        raise ValueError(
            f"No non-hydrogen neighbors found for the closest select atom (index {closest_select_atom.GetIdx()}).")

    c_atom = bonded_c_atoms[0]
    c_idx = c_atom.GetIdx()

    selected_atom_indices = set()
    selected_atom_indices.add(c_idx)

    # Step 4: 从该原子开始，限定步数的广度优先搜索
    visited = set()
    queue = deque()
    queue.append((c_idx, 0))
    visited.add(c_idx)

    while queue:
        current_idx, step = queue.popleft()
        if step >= max_steps:
            continue

        current_atom = mol.GetAtomWithIdx(current_idx)
        for neighbor in current_atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            neighbor_symbol = neighbor.GetSymbol()

            if neighbor_symbol == 'H':
                continue

            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                selected_atom_indices.add(neighbor_idx)
                queue.append((neighbor_idx, step + 1))

    return c_idx, sorted(center_atom_indices), sorted(selected_atom_indices)

def find_poly_match_subindex(poly_name, repeating_unit, length, mol, selected_atom_idx, center_atom_idx, ):

    (
        dum1,
        dum2,
        atom1,
        atom2,
    ) = model_lib.Init_info(
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
    print(matches)

    print(start_atom, end_atom)

    for match in matches:
        if center_atom_idx in match:
            if all(idx in selected_atom_idx for idx in match):
                return list(match), start_atom, end_atom

def get_cluster_withcap(work_dir, mol, match_list, center_atom_idx, start_atom, end_atom, out_xyz_filename):
    print(match_list)
    print(center_atom_idx)
    h_atom_idx = set()
    for idx in match_list:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:  # H atom
                h_atom_idx.add(neighbor.GetIdx())

    select_atom_idx_with_h = sorted(set(match_list + list(h_atom_idx) + center_atom_idx))

    new_mol = Chem.RWMol()
    index_map = {}

    for old_idx in select_atom_idx_with_h:
        atom = mol.GetAtomWithIdx(old_idx)
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_idx = new_mol.AddAtom(new_atom)
        index_map[old_idx] = new_idx

    # 创建一个反向索引映射（新分子索引到原始原子索引）
    reverse_index_map = {v: k for k, v in index_map.items()}
    print(reverse_index_map)

    # 添加原子间的键（如果两个原子都在要提取的列表中）
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in select_atom_idx_with_h and end in select_atom_idx_with_h:
            new_mol.AddBond(index_map[begin], index_map[end], bond.GetBondType())

    terminal_atoms = [start_atom, end_atom]

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
                bond_vec = terminal_pos - neighbor_old_pos
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
    new_mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(new_mol)

    # save to xyz file
    out_xyz_filepath = os.path.join(work_dir, out_xyz_filename)
    Chem.MolToXYZFile(new_mol, out_xyz_filepath)



# def xyz2mol(work_dir, xyz_filename):
#
#     label_to_element = {
#         'NBT': 'N',
#         'SBT': 'S',
#         'OBT': 'O',
#         'CBT': 'C',
#         'F1': 'F',
#     }
#
#     # read the xyz file
#     xyz_filepath = os.path.join(work_dir, xyz_filename)
#     with open(xyz_filepath, 'r') as f:
#         lines = f.readlines()
#
#     num_atoms = int(lines[0].strip())
#
#     # create a new RDKit molecule
#     mol = Chem.RWMol()
#     # add atoms to the molecule
#     for i in range(2, 2 + num_atoms):
#         parts = lines[i].strip().split()
#         if len(parts) < 4:
#             print(f"Invalid line: {lines[i]}")
#             continue
#         label, x, y, z = parts[:4]
#
#         element = label_to_element.get(label, label)
#         atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(element)
#         if atomic_num == 0:
#             print(f"Unrecognized element symbols '{element}'。")
#             continue
#         atom = Chem.Atom(atomic_num)
#         mol.AddAtom(atom)
#
#     # generate a 3D conformer
#     conf = Chem.Conformer(num_atoms)
#     for i in range(num_atoms):
#         parts = lines[2 + i].strip().split()
#         if len(parts) < 4:
#             continue
#         x, y, z = map(float, parts[1:4])
#         conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))
#     mol.AddConformer(conf)
#
#     # add bonds
#     tolerance = 0.4  # Å
#     pt = Chem.GetPeriodicTable()
#     for i in range(num_atoms):
#         atom_i = mol.GetAtomWithIdx(i)
#         for j in range(i + 1, num_atoms):
#             atom_j = mol.GetAtomWithIdx(j)
#             # calculate the distance between atoms
#             pos_i = np.array(conf.GetAtomPosition(i))
#             pos_j = np.array(conf.GetAtomPosition(j))
#             distance = np.linalg.norm(pos_i - pos_j)
#             # get the covalent radii of the atoms
#             radius_i = pt.GetRcovalent(atom_i.GetSymbol())
#             radius_j = pt.GetRcovalent(atom_j.GetSymbol())
#             if radius_i == 0 or radius_j == 0:
#                 continue  # skip if the radius is not available
#             # check if the distance is within the sum of the covalent radii
#             if distance <= (radius_i + radius_j + tolerance):
#                 try:
#                     mol.AddBond(i, j, order=Chem.rdchem.BondType.SINGLE)
#                 except Exception as e:
#                     print(f"Adding key failed：{e}")
#
#     # convert to Mol object and clean up
#     mol = mol.GetMol()
#     Chem.SanitizeMol(mol)
#
#     return mol

# def get_cluster_index(mol, center_atom_symbol, select_atom_symbol, repeating_unit, length,):
#
#     unit_smi_with_h1 = repeating_unit.replace('*', '[H]')
#     unit_mol_with_h1 = Chem.MolFromSmiles(unit_smi_with_h1)
#     unit_mol = Chem.RemoveHs(unit_mol_with_h1)
#     num_unit = unit_mol.GetNumAtoms()
#     max_steps = num_unit * (length - 1)
#
#     conf = mol.GetConformer()
#     # Step 1: 找到中心原子（假设只有一个）
#     center_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == center_atom_symbol]
#     center_atom = center_atoms[0]
#     n_idx = center_atom.GetIdx()
#
#     # 获取选定原子的列表和坐标
#     select_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == select_atom_symbol]
#     select_positions = np.array([conf.GetAtomPosition(atom.GetIdx()) for atom in select_atoms])
#
#     # Step 2: 从中心原子开始，找到其所在的连通组分
#     center_atom_indices = set()
#     queue = deque()
#
#     center_atom_indices.add(n_idx)
#     queue.append(n_idx)
#     while queue:
#         current_idx = queue.popleft()
#         current_atom = mol.GetAtomWithIdx(current_idx)
#         for neighbor in current_atom.GetNeighbors():
#             neighbor_idx = neighbor.GetIdx()
#             if neighbor_idx not in center_atom_indices:
#                 center_atom_indices.add(neighbor_idx)
#                 queue.append(neighbor_idx)
#
#     # Step 3: 找到与中心原子最近的选定原子，并确认键连接
#     n_pos = np.array(conf.GetAtomPosition(n_idx))
#     distances = np.linalg.norm(select_positions - n_pos, axis=1)
#     min_dist_idx = np.argmin(distances)
#     closest_select_atom = select_atoms[min_dist_idx]
#
#     # 获取与最近的选定原子相连的特定原子
#     bonded_c_atoms = [nbr for nbr in closest_select_atom.GetNeighbors()]
#     c_atom = bonded_c_atoms[0]
#     c_idx = c_atom.GetIdx()
#
#     selected_atom_indices = set()
#     selected_atom_indices.add(c_idx)
#
#     # Step 4: 从该原子开始，限定步数的广度优先搜索
#     visited = set()
#     queue = deque()
#     queue.append((c_idx, 0))
#     visited.add(c_idx)
#
#     while queue:
#         current_idx, step = queue.popleft()
#         if step >= max_steps:
#             continue
#
#         current_atom = mol.GetAtomWithIdx(current_idx)
#         for neighbor in current_atom.GetNeighbors():
#             neighbor_idx = neighbor.GetIdx()
#             neighbor_symbol = neighbor.GetSymbol()
#
#             if neighbor_symbol == 'H':
#                 continue
#
#             if neighbor_idx not in visited:
#                 visited.add(neighbor_idx)
#                 selected_atom_indices.add(neighbor_idx)
#                 queue.append((neighbor_idx, step + 1))
#
#     return c_idx, sorted(center_atom_indices), sorted(selected_atom_indices)