import os
import shutil
import pandas as pd
import parmed as pmd
import importlib.resources as pkg_resources

from rdkit import Chem
from foyer import Forcefield
from collections import defaultdict

from PEMD.model import model_lib
from PEMD.model.build import gen_poly_smiles
from PEMD.forcefields.xml import XMLGenerator
from PEMD.forcefields.ligpargen import PEMDLigpargen

from PEMD.model.build import (
    gen_poly_smiles,
    gen_copoly_smiles,
    gen_poly_3D,
)


def get_xml_ligpargen(work_dir, name, resname, repeating_unit, length, chg, chg_model, ):

    smiles = gen_poly_smiles(
        name,
        repeating_unit,
        length,
        leftcap='',
        rightcap='',
    )

    ligpargen_dir = os.path.join(work_dir, f'ligpargen_{name}')
    os.makedirs(ligpargen_dir, exist_ok=True)

    pdb_filename = f'{name}.pdb'
    pdb_filepath = os.path.join(work_dir, pdb_filename)
    model_lib.smiles_to_pdb(smiles, pdb_filepath, name, resname)

    PEMDLigpargen(
        ligpargen_dir,
        name,
        resname,
        chg,
        chg_model,
        filename = pdb_filepath,
    ).run_local()

    gmx_itp_file = os.path.join(ligpargen_dir, f"{name}.gmx.itp")
    xml_filename = os.path.join(work_dir, f"{name}.xml")
    generator = XMLGenerator(
        gmx_itp_file,
        smiles,
        xml_filename
    )
    generator.run()

    os.remove(f'{work_dir}/temp.sdf')
    os.remove(pdb_filepath)

    chg_file = os.path.join(ligpargen_dir, f'{name}.csv')
    resp_chg_df = pd.read_csv(chg_file)
    return resp_chg_df

def get_oplsaa_xml(
        work_dir,
        poly_name,
        pdb_file,
        xml = 'ligpargen',  # ligpargen or database
):

    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    untyped_str = pmd.load_file(pdb_file, structure=True)
    if xml == 'database':
        with pkg_resources.path("PEMD.forcefields", "oplsaa.xml") as oplsaa_path:
            oplsaa = Forcefield(forcefield_files = str(oplsaa_path))
        typed_str = oplsaa.apply(untyped_str, verbose=True, use_residue_map=True)
    else:
        xml_filename = os.path.join(work_dir, f"{poly_name}.xml")
        oplsaa = Forcefield(forcefield_files = xml_filename)
        typed_str = oplsaa.apply(untyped_str, verbose=True, use_residue_map=True)

    top_filename = os.path.join(MD_dir, f"{poly_name}.top")
    gro_filename = os.path.join(MD_dir, f"{poly_name}.gro")
    typed_str.save(top_filename, overwrite=True)
    typed_str.save(gro_filename, overwrite=True)

    nonbonditp_filename = os.path.join(MD_dir, f'{poly_name}_nonbonded.itp')
    bonditp_filename = os.path.join(MD_dir, f'{poly_name}_bonded.itp')

    model_lib.extract_from_top(top_filename, nonbonditp_filename, nonbonded=True, bonded=False)
    model_lib.extract_from_top(top_filename, bonditp_filename, nonbonded=False, bonded=True)

    os.remove(top_filename)
    # os.remove(gro_filename)

    return f'{poly_name}_bonded.itp', f"{poly_name}.gro"


def get_oplsaa_ligpargen(work_dir, name, resname, chg, chg_model, smiles, ):

    ligpargen_dir = os.path.join(work_dir, f'ligpargen_{name}')
    os.makedirs(ligpargen_dir, exist_ok=True)

    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    xyz_filename = f'{name}.xyz'
    model_lib.smiles_to_xyz(smiles, os.path.join(work_dir,  xyz_filename))

    PEMDLigpargen(
        ligpargen_dir,
        name,
        resname,
        chg,
        chg_model,
        filename = xyz_filename,
    ).run_local()

    nonbonditp_filename = os.path.join(MD_dir, f'{name}_nonbonded.itp')
    bonditp_filename = os.path.join(MD_dir, f'{name}_bonded.itp')
    pdb_filename = os.path.join(MD_dir, f'{name}.pdb')

    top_filename = os.path.join(ligpargen_dir, f"{name}.gmx.itp")
    gro_filename = os.path.join(ligpargen_dir, f'{name}.gmx.gro')
    model_lib.extract_from_top(top_filename, nonbonditp_filename, nonbonded=True, bonded=False)
    model_lib.extract_from_top(top_filename, bonditp_filename, nonbonded=False, bonded=True)
    model_lib.convert_gro_to_pdb(gro_filename, pdb_filename,)

    return f'{name}_bonded.itp'


def gen_ff_from_data(work_dir, compound_name, corr_factor, target_sum_chg):

    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    files_to_copy = [
        f"pdb/{compound_name}.pdb",
        f"itp/{compound_name}_bonded.itp",
        f"itp/{compound_name}_nonbonded.itp"
    ]
    for file_path in files_to_copy:
        try:
            resource_dir = pkg_resources.files('PEMD.forcefields')
            resource_path = resource_dir.joinpath(file_path)
            shutil.copy(str(resource_path), MD_dir)
            print(f"Copied {file_path} to {MD_dir} successfully.")
        except Exception as e:
            print(f"Failed to copy {file_path}: {e}")

    filename = f"{compound_name}_bonded.itp"
    scale_chg_itp(MD_dir, filename, corr_factor, target_sum_chg)
    print(f"scale charge successfully.")


def assign_partial_charges(mol_poly, sub_mol, matches):
    """
    将sub_mol的部分电荷赋值到mol_poly中的对应原子
    """
    for match in matches:
        for sub_atom_idx, poly_atom_idx in enumerate(match):
            sub_atom = sub_mol.GetAtomWithIdx(sub_atom_idx)
            charge = float(sub_atom.GetProp('partial_charge'))
            mol_poly.GetAtomWithIdx(poly_atom_idx).SetDoubleProp('partial_charge', charge)

def mol_to_charge_df(mol):

    data = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        # 检查原子是否具有 'partial_charge' 属性
        if atom.HasProp('partial_charge'):
            charge = float(atom.GetProp('partial_charge'))
        else:
            charge = None  # 或者设置为 0.0 或其他默认值
        data.append({'atom_index': atom_idx, 'atom': atom_symbol, 'charge': charge})

    # 创建 DataFrame 并按原子索引排序
    df = pd.DataFrame(data)
    df = df.sort_values('atom_index').reset_index(drop=True)

    return df


def apply_chg_to_poly(work_dir, smiles_short, mol_long, itp_file, resp_chg_df, repeating_unit,
                          end_repeating, corr_factor, target_sum_chg, ):

    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    left_mol, right_mol, mid_mol = apply_chg2mol(
        resp_chg_df,
        smiles_short,
        repeating_unit,
        end_repeating
    )

    # print(left_mol.GetNumAtoms())

    # mol_poly = Chem.MolFromSmiles(smiles_long)
    Chem.SanitizeMol(mol_long)
    mol_poly = Chem.AddHs(mol_long)

    # 将left_mol匹配到mol_poly中
    left_matches = []
    rw_mol = Chem.RWMol(mol_poly)
    used_atoms = set()
    for match in rw_mol.GetSubstructMatches(left_mol, uniquify=True, useChirality=False):
        if any(atom_idx in used_atoms for atom_idx in match):
            continue  # 跳过有重叠的匹配
        left_matches.append(match)
        used_atoms.update(match)  # 标记已使用的原子
    # print(f"left_mol 的匹配位置: {left_matches}")

    # 将right_mol匹配到mol_poly中
    right_matches = []
    rw_mol = Chem.RWMol(mol_poly)
    for match in rw_mol.GetSubstructMatches(right_mol, uniquify=True, useChirality=False):
        if any(atom_idx in used_atoms for atom_idx in match):
            continue  # 跳过有重叠的匹配
        right_matches.append(match)
        used_atoms.update(match)  # 标记已使用的原子
    # print(f"right_mol 的匹配位置: {right_matches}")

    # 赋值部分电荷给mol_poly中的对应原子
    assign_partial_charges(mol_poly, left_mol, left_matches)
    assign_partial_charges(mol_poly, right_mol, right_matches)

    # 将 mid_mol 匹配到 mol_poly 中的重复单元并赋值电荷
    mid_matches = []
    for match in rw_mol.GetSubstructMatches(mid_mol, uniquify=True, useChirality=False):
        if any(atom_idx in used_atoms for atom_idx in match):
            continue  # 跳过有重叠的匹配
        mid_matches.append(match)
        used_atoms.update(match)  # 标记已使用的原子
    # print(f"mid_mol 的匹配位置: {mid_matches}")

    # 赋值部分电荷给 mol_poly 中的 mid_mol 匹配位置
    assign_partial_charges(mol_poly, mid_mol, mid_matches)

    # 提取电荷为 DataFrame
    charge_update_df = mol_to_charge_df(mol_poly)

    # charge neutralize and scale
    charge_update_df_cor = charge_neutralize_scale(charge_update_df, corr_factor, target_sum_chg, )

    # update the itp file
    update_itp_file(MD_dir, itp_file, charge_update_df_cor)


def apply_chg2mol(resp_chg_df, poly_smi, repeating_unit, end_repeating):
    # 1. 从多聚物 SMILES 生成分子并加氢
    mol_poly = Chem.MolFromSmiles(poly_smi)
    mol_poly = Chem.AddHs(mol_poly)

    # 2. 将 RESP 电荷写入到 mol_poly 中
    for _, row in resp_chg_df.iterrows():
        pos = int(row['position'])
        charge = float(row['charge'])
        atom = mol_poly.GetAtomWithIdx(pos)
        atom.SetDoubleProp('partial_charge', charge)  # 使用 SetDoubleProp 存储电荷

    partial_charges = [
        float(row['charge'])
        for _, row in resp_chg_df.sort_values('position').iterrows()
    ]
    mol_poly.SetProp("partial_charges", ','.join(map(str, partial_charges)))

    # ==========  生成正向 mol_unit_fwd  ==========
    mol_unit_fwd = Chem.MolFromSmiles(repeating_unit)
    mol_unit_fwd = Chem.AddHs(mol_unit_fwd)
    # 移除星号原子
    edit_fwd = Chem.EditableMol(mol_unit_fwd)
    for atom in reversed(list(mol_unit_fwd.GetAtoms())):
        if atom.GetSymbol() == '*':
            edit_fwd.RemoveAtom(atom.GetIdx())
    mol_unit_fwd = edit_fwd.GetMol()

    # ==========  生成逆向 mol_unit_rev  ==========
    # 在 mol_unit_fwd 的基础上进行原子重排
    num_atoms_fwd = mol_unit_fwd.GetNumAtoms()
    new_order = list(range(num_atoms_fwd - 1, -1, -1))
    mol_unit_rev = Chem.RenumberAtoms(mol_unit_fwd, new_order)

    # ==========  对 mol_poly 分别进行子结构匹配  ==========
    rw_mol = Chem.RWMol(mol_poly)

    # 正向匹配
    fwd_used_atoms = set()
    fwd_matches = []
    for match in rw_mol.GetSubstructMatches(mol_unit_fwd, uniquify=True, useChirality=False):
        if any(atom_idx in fwd_used_atoms for atom_idx in match):
            continue
        fwd_matches.append(match)
        fwd_used_atoms.update(match)

    # 逆向匹配
    rev_used_atoms = set()
    rev_matches = []
    for match in rw_mol.GetSubstructMatches(mol_unit_rev, uniquify=True, useChirality=False):
        if any(atom_idx in rev_used_atoms for atom_idx in match):
            continue
        rev_matches.append(match)
        rev_used_atoms.update(match)

    # print("fwd_matches:", fwd_matches)
    # print("rev_matches:", rev_matches)

    # ==========  选择最佳匹配  ==========
    # 这里以“哪个匹配数更多”作为衡量标准，实际可根据需要调整
    if len(fwd_matches) >= len(rev_matches):
        best_matches = fwd_matches
        best_mol_unit = mol_unit_fwd
        best_direction = "forward"
    else:
        best_matches = rev_matches
        best_mol_unit = mol_unit_rev
        best_direction = "reverse"

    # print(f"Best direction: {best_direction}, matches found: {len(best_matches)}")

    # 如果没有匹配到，就直接返回
    if not best_matches:
        print("No matches found in either direction!")
        return None, None, None

    # ==========  后续操作就基于 best_matches 和 best_mol_unit  ==========
    # 下面基本沿用你原本的逻辑，但要注意把变量替换为 “最佳匹配” 相关的
    matched_atoms = set()
    for match in best_matches:
        matched_atoms.update(match)

    no_matched_atoms = [
        atom.GetIdx() for atom in mol_poly.GetAtoms()
        if atom.GetIdx() not in matched_atoms
    ]

    # 初始化端原子列表
    left_end_atoms = []
    right_end_atoms = []

    # 确保 best_matches 不为空
    if best_matches:
        left_neighbor = set(best_matches[0])
        right_neighbor = set(best_matches[-1])

        # 使用列表副本迭代，以避免修改原列表
        for idx in no_matched_atoms[:]:
            atom = mol_poly.GetAtomWithIdx(idx)
            neighbors = atom.GetNeighbors()
            for neighbor in neighbors:
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in left_neighbor:
                    left_end_atoms.append(idx)
                    left_neighbor.add(idx)
                    no_matched_atoms.remove(idx)
                    break
                elif neighbor_idx in right_neighbor:
                    right_end_atoms.append(idx)
                    right_neighbor.add(idx)
                    no_matched_atoms.remove(idx)
                    break
    # print(f"Left end atoms: {left_end_atoms}")
    # print(f"Right end atoms: {right_end_atoms}")

    # 组装左右端原子
    left_atoms = left_end_atoms + [
        atom_idx for match in best_matches[:end_repeating] for atom_idx in match
    ]
    right_atoms = right_end_atoms + [
        atom_idx for match in best_matches[-end_repeating:] for atom_idx in match
    ]
    # print("left_end_atoms:", left_end_atoms, "right_end_atoms:", right_end_atoms)

    # 这里假设你有一个函数 `gen_molfromindex(mol, index_list)` 用于通过原子索引提取子分子
    left_mol = gen_molfromindex(mol_poly, left_atoms)
    right_mol = gen_molfromindex(mol_poly, right_atoms)

    # 打印并获取部分电荷，赋值给 left_mol / right_mol
    for i, atom_idx in enumerate(left_atoms):
        charge = mol_poly.GetAtomWithIdx(atom_idx).GetDoubleProp('partial_charge')
        if i < left_mol.GetNumAtoms():
            left_mol.GetAtomWithIdx(i).SetDoubleProp('partial_charge', charge)
        # print(f"left位置 {atom_idx} 的平均电荷: {charge}")

    for i, atom_idx in enumerate(right_atoms):
        charge = mol_poly.GetAtomWithIdx(atom_idx).GetDoubleProp('partial_charge')
        if i < right_mol.GetNumAtoms():
            right_mol.GetAtomWithIdx(i).SetDoubleProp('partial_charge', charge)
        # print(f"right位置 {atom_idx} 的平均电荷: {charge}")

    # 假设每个 match 在中间（best_matches 的中段）代表一个重复单元
    mid_atoms = best_matches[1:-1]
    num_repeats = len(mid_atoms)
    # print(f"中间重复单元数量: {num_repeats}")

    # 确定每个重复单元中的原子数量
    if num_repeats > 0:
        atoms_per_unit = len(mid_atoms[0])
        # print(f"每个重复单元的原子数量: {atoms_per_unit}")
    else:
        atoms_per_unit = 0

    # 计算中间重复单元各位置的平均电荷
    charge_dict = defaultdict(list)
    for repeat_idx, match in enumerate(mid_atoms):
        for pos, atom_idx in enumerate(match):
            charge = mol_poly.GetAtomWithIdx(atom_idx).GetDoubleProp('partial_charge')
            charge_dict[pos].append(charge)

    avg_charges = {}
    for pos, charges in charge_dict.items():
        avg_charge = sum(charges) / len(charges)
        avg_charges[pos] = avg_charge
        # print(f"位置 {pos} 的平均电荷: {avg_charge}")

    # 选取第一个重复单元作为模板
    if num_repeats > 0:
        template_match = mid_atoms[0]
        template_atoms = list(template_match)
        mid_mol = gen_molfromindex(mol_poly, template_atoms)

        # 重新分配部分电荷到 mid_mol
        for pos, atom in enumerate(mid_mol.GetAtoms()):
            atom.SetDoubleProp('partial_charge', avg_charges[pos])
    else:
        mid_mol = None

    return left_mol, right_mol, mid_mol


def ave_chg_to_df(resp_chg_df, N_mid, N_mid_h, N_left, N_right, N_left_h, N_right_h, ):
    # deal with non-H atoms
    nonH_df = resp_chg_df[resp_chg_df['atom'] != 'H']

    top_N_noH_df = nonH_df.head(N_left)
    tail_N_noH_df = nonH_df.tail(N_right)
    mid_df_noH_df = nonH_df.drop(nonH_df.head(N_left).index.union(nonH_df.tail(N_right).index)).reset_index(drop=True)
    mid_ave_chg_noH_df = ave_mid_chg(mid_df_noH_df, N_mid)

    # deal with H atoms
    H_df = resp_chg_df[resp_chg_df['atom'] == 'H']

    top_N_H_df = H_df.head(N_left_h)
    tail_N_H_df = H_df.tail(N_right_h)
    mid_df_H_df = H_df.drop(H_df.head(N_left_h).index.union(H_df.tail(N_right_h).index)).reset_index(drop=True)
    mid_ave_chg_H_df = ave_mid_chg(mid_df_H_df, N_mid_h)

    return top_N_noH_df, tail_N_noH_df, mid_ave_chg_noH_df, top_N_H_df, tail_N_H_df, mid_ave_chg_H_df


def update_itp_file(MD_dir, itp_file, charge_update_df_cor):
    itp_filepath = os.path.join(MD_dir, itp_file)
    with open(itp_filepath, 'r') as file:
        lines = file.readlines()

    # 找到[ atoms ]部分的开始和结束
    in_section = False  # 标记是否处于指定部分
    start_index = end_index = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("[") and 'atoms' in line.split():
            in_section = True
            continue
        if in_section:
            if line.strip().startswith(";"):
                start_index = i + 1  # 跳过部分标题和列标题
                continue
            if line.strip() == "":
                end_index = i
                break

    # update the charge value in the [ atoms ] section
    charge_index = 0  # 用于跟踪DataFrame中当前的电荷索引
    for i in range(start_index, end_index):
        parts = lines[i].split()
        if charge_index < len(charge_update_df_cor):
            new_charge = charge_update_df_cor.iloc[charge_index]['charge']
            parts[6] = f'{new_charge:.8f}'  # 更新电荷值，假设电荷值在第7个字段
            lines[i] = ' '.join(parts) + '\n'
            charge_index += 1

    # save the updated itp file
    new_itp_filepath = os.path.join(MD_dir, itp_file)
    with open(new_itp_filepath, 'w') as file:
        file.writelines(lines)


def ave_end_chg(df, N):
    # 处理端部原子的电荷平均值
    top_N_df = df.head(N)
    tail_N_df = df.tail(N).iloc[::-1].reset_index(drop=True)
    average_charge = (top_N_df['charge'].reset_index(drop=True) + tail_N_df['charge']) / 2
    average_df = pd.DataFrame({
        'atom': top_N_df['atom'].reset_index(drop=True),  # 保持原子名称
        'charge': average_charge
    })
    return average_df


def ave_mid_chg(df, atom_count):
    # 处理中间原子的电荷平均值
    average_charges = []
    for i in range(atom_count):
        same_atoms = df[df.index % atom_count == i]
        avg_charge = same_atoms['charge'].mean()
        average_charges.append({'atom': same_atoms['atom'].iloc[0], 'charge': avg_charge})
    return pd.DataFrame(average_charges)


def smiles_to_df(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无效的SMILES字符串")

    mol = Chem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

    df = pd.DataFrame(atoms, columns=['atom'])
    df['charge'] = None
    return df



def gen_molfromindex(mol, idx):
    editable_mol = Chem.EditableMol(Chem.Mol())

    atom_map = {}  # 原始索引到新索引的映射
    for old_idx in idx:
        atom = mol.GetAtomWithIdx(old_idx)
        new_idx = editable_mol.AddAtom(atom)
        atom_map[old_idx] = new_idx

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in idx  and end_idx in idx:
            new_begin = atom_map[begin_idx]
            new_end = atom_map[end_idx]
            bond_type = bond.GetBondType()
            editable_mol.AddBond(new_begin, new_end, bond_type)

    return editable_mol.GetMol()

def charge_neutralize_scale(df, correction_factor=1, target_total_charge=0, ):
    current_total_charge = df['charge'].sum()
    charge_difference = target_total_charge - current_total_charge
    charge_adjustment_per_atom = charge_difference / len(df)
    df['charge'] = (df['charge'] + charge_adjustment_per_atom) * correction_factor

    return df

def apply_chg_to_molecule(work_dir, itp_file, resp_chg_df, corr_factor, target_sum_chg, ):
    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    # charge neutralize and scale
    charge_update_df_cor = charge_neutralize_scale(resp_chg_df, corr_factor, target_sum_chg, )

    # update the itp file
    update_itp_file(MD_dir, itp_file, charge_update_df_cor)


# work_dir, filename, corr_factor, target_sum_chg
def scale_chg_itp(work_dir, filename, corr_factor, target_sum_chg):
    filename = os.path.join(work_dir, filename)
    start_reading = False
    atoms = []

    with open(filename, 'r') as file:
        for line in file:
            if line.strip().startswith("[") and 'atoms' in line.split():  # 找到原子信息开始的地方
                start_reading = True
                continue
            if start_reading:
                if line.strip() == "":  # 遇到空行，停止读取
                    break
                if line.strip().startswith(";"):  # 忽略注释行
                    continue
                parts = line.split()
                if len(parts) >= 7:  # 确保行包含足够的数据
                    atom_id = parts[4]  # 假设原子类型在第5列
                    charge = float(parts[6])  # 假设电荷在第7列
                    atoms.append([atom_id, charge])

    # create DataFrame
    df = pd.DataFrame(atoms, columns=['atom', 'charge'])
    # charge neutralize and scale
    charge_update_df_cor = charge_neutralize_scale(df, corr_factor, target_sum_chg, )

    # reas itp file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 找到[ atoms ]部分的开始和结束
    in_section = False  # 标记是否处于指定部分
    start_index = end_index = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("[") and 'atoms' in line.split():
            in_section = True
            continue
        if in_section:
            if line.strip().startswith(";"):
                start_index = i + 1  # 跳过部分标题和列标题
                continue
            if line.strip() == "":
                end_index = i
                break

    # update the charge value in the [ atoms ] section
    charge_index = 0  # 用于跟踪DataFrame中当前的电荷索引
    for i in range(start_index, end_index):
        parts = lines[i].split()
        if charge_index < len(charge_update_df_cor):
            new_charge = charge_update_df_cor.iloc[charge_index]['charge']
            parts[6] = f'{new_charge:.8f}'  # 更新电荷值，假设电荷值在第7个字段
            lines[i] = ' '.join(parts) + '\n'
            charge_index += 1

    # save the updated itp file
    with open(filename, 'w') as file:
        file.writelines(lines)





