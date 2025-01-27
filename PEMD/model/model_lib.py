"""
PEMD code library.

Developed by: Tan Shendong
Date: 2024.03.15
"""

import re
import subprocess
import numpy as np
import pandas as pd
import networkx as nx
from mpmath import angerj

from rdkit import Chem
from openbabel import pybel
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem import Descriptors
from collections import defaultdict
from openbabel import openbabel as ob
from networkx.algorithms import isomorphism
from rdkit.Chem.rdmolfiles import MolToPDBFile
from scipy.spatial.transform import Rotation as R


# OpenBabel setup
obConversion = ob.OBConversion()
ff = ob.OBForceField.FindForceField('UFF')
mol = ob.OBMol()
np.set_printoptions(precision=20)

def rdkitmol2xyz(unit_name, m, out_dir, IDNum):
    try:
        Chem.MolToXYZFile(m, out_dir + '/' + unit_name + '.xyz', confId=IDNum)
    except Exception:
        obConversion.SetInAndOutFormats("mol", "xyz")
        Chem.MolToMolFile(m, out_dir + '/' + unit_name + '.mol', confId=IDNum)
        mol = ob.OBMol()
        obConversion.ReadFile(mol, out_dir + '/' + unit_name + '.mol')
        obConversion.WriteFile(mol, out_dir + '/' + unit_name + '.xyz')

def smile_toxyz(name, SMILES, ):
    # Generate XYZ file from SMILES
    m1 = Chem.MolFromSmiles(SMILES)    # Get mol(m1) from smiles
    m2 = Chem.AddHs(m1)   # Add H
    AllChem.Compute2DCoords(m2)    # Get 2D coordinates
    AllChem.EmbedMolecule(m2)    # Make 3D mol
    m2.SetProp("_Name", name + '   ' + SMILES)    # Change title
    AllChem.UFFOptimizeMolecule(m2, maxIters=200)    # Optimize 3D str
    rdkitmol2xyz(name, m2, '.', -1)
    file_name = name + '.xyz'
    return file_name

def FetchDum(smiles):
    # Get index of dummy atoms and bond type associated with it
    m = Chem.MolFromSmiles(smiles)
    dummy_index = []
    bond_type = None
    if m is not None:
        for atom in m.GetAtoms():
            if atom.GetSymbol() == '*':
                dummy_index.append(atom.GetIdx())
        for bond in m.GetBonds():
            if (
                bond.GetBeginAtom().GetSymbol() == '*'
                or bond.GetEndAtom().GetSymbol() == '*'
            ):
                bond_type = bond.GetBondType()
                break
    return dummy_index, str(bond_type)

def connec_info(name):
    # Collect valency and connecting information for each atom according to XYZ coordinates
    obConversion = ob.OBConversion()
    obConversion.SetInFormat("xyz")
    mol = ob.OBMol()
    obConversion.ReadFile(mol, name)
    neigh_atoms_info = []

    for atom in ob.OBMolAtomIter(mol):
        neigh_atoms = []
        bond_orders = []
        for allatom in ob.OBAtomAtomIter(atom):
            neigh_atoms.append(allatom.GetIndex())
            bond_orders.append(atom.GetBond(allatom).GetBondOrder())
        neigh_atoms_info.append([neigh_atoms, bond_orders])
    neigh_atoms_info = pd.DataFrame(neigh_atoms_info, columns=['NeiAtom', 'BO'])
    return neigh_atoms_info


# Initializes and retrieves information about dummy atoms in a molecule.
def Init_info(poly_name, smiles_mid ):
    # Get index of dummy atoms and atoms associated with them
    dum_index, bond_type = FetchDum(smiles_mid)
    dum1 = dum_index[0]
    dum2 = dum_index[1]

    # Assign dummy atom according to bond type
    dum = None
    if bond_type == 'SINGLE':
        dum = 'Cl'

    # Replace '*' with dummy atom
    smiles_each = smiles_mid.replace(r'*', dum)

    # Convert SMILES to XYZ coordinates
    xyz_filename = smile_toxyz(
        poly_name,
        smiles_each,       # Replace '*' with dummy atom
    )

    # Collect valency and connecting information for each atom according to XYZ coordinates
    neigh_atoms_info = connec_info(xyz_filename)

    # Find connecting atoms associated with dummy atoms.
    # Dum1 and dum2 are connected to atom1 and atom2, respectively.
    atom1 = neigh_atoms_info['NeiAtom'][dum1].copy()[0]
    atom2 = neigh_atoms_info['NeiAtom'][dum2].copy()[0]

    return dum1, dum2, atom1, atom2,

def gen_oligomer_smiles(poly_name, dum1, dum2, atom1, atom2, smiles_each, length, smiles_LCap_, smiles_RCap_, ):

    (
        inti_mol3,
        monomer_mol,
        start_atom,
        end_atom,
    ) = gen_smiles_nocap(
        dum1,
        dum2,
        atom1,
        atom2,
        smiles_each,
        length,
    )

    # Obtain the SMILES with cap
    main_mol_noDum = gen_smiles_with_cap(
        poly_name,
        inti_mol3,
        atom1,
        atom2 + (length -1) * monomer_mol.GetNumAtoms(),
        smiles_LCap_,
        smiles_RCap_,
    )

    return Chem.MolToSmiles(main_mol_noDum)

def gen_smiles_nocap(dum1, dum2, atom1, atom2, smiles_each, length, ):
    # Connect the units and caps to obtain SMILES structure
    input_mol = Chem.MolFromSmiles(smiles_each)
    edit_m1 = Chem.EditableMol(input_mol)
    edit_m2 = Chem.EditableMol(input_mol)
    edit_m3 = Chem.EditableMol(input_mol)

    edit_m1.RemoveAtom(dum2) # Delete dum2
    mol_without_dum1 = Chem.Mol(edit_m1.GetMol())

    edit_m2.RemoveAtom(dum1) # Delete dum1
    mol_without_dum2 = Chem.Mol(edit_m2.GetMol())

    edit_m3.RemoveAtom(dum1) # Delete dum1 and dum2
    if dum1 < dum2: # If dum1 < dum2, then the index of dum2 is dum2 - 1
        edit_m3.RemoveAtom(dum2 - 1)
    else:
        edit_m3.RemoveAtom(dum2)
    monomer_mol = edit_m3.GetMol()
    inti_mol = monomer_mol

    # After removing dummy atoms,adjust the order and set first_atom and second_atom to represent connecting atoms
    if atom1 > atom2:
        atom1, atom2 = atom2, atom1

    if dum1 < atom1 and dum2 < atom1:
        first_atom = atom1 - 2
    elif (dum1 < atom1 < dum2) or (dum1 > atom1 > dum2):
        first_atom = atom1 - 1
    else:
        first_atom = atom1

    if dum1 < atom2 and dum2 < atom2:
        second_atom = atom2 - 2
    elif (dum1 < atom2 < dum2) or (dum1 > atom2 > dum2):
        second_atom = atom2 - 1
    else:
        second_atom = atom2

    inti_mol3 = None

    if length == 1:
        inti_mol3 = input_mol

    # Connect the units
    elif length > 1:

        if length > 2:
            for i in range(1, length - 2):      # Fist connect middle n-2 units
                combo = Chem.CombineMols(inti_mol, monomer_mol)
                edcombo = Chem.EditableMol(combo)
                edcombo.AddBond(
                    second_atom + (i - 1) * monomer_mol.GetNumAtoms(),
                    first_atom + i * monomer_mol.GetNumAtoms(),
                    order=Chem.rdchem.BondType.SINGLE,
                )
                # Add bond according to the index of atoms to be connected

                inti_mol = edcombo.GetMol()

            inti_mol1 = inti_mol

            combo = Chem.CombineMols(mol_without_dum1, inti_mol1)   # Connect the leftmost unit
            edcombo = Chem.EditableMol(combo)
            edcombo.AddBond(
                atom2,
                first_atom + mol_without_dum1.GetNumAtoms(),
                order=Chem.rdchem.BondType.SINGLE
            )
            # print(atom2)
            # print(first_atom + mol_without_dum1.GetNumAtoms())
            inti_mol2 = edcombo.GetMol()

            combo = Chem.CombineMols(inti_mol2, mol_without_dum2)   # Connect the rightmost unit
            edcombo = Chem.EditableMol(combo)
            # print(atom2 + inti_mol1.GetNumAtoms())
            # print(first_atom + inti_mol2.GetNumAtoms())
            edcombo.AddBond(
                atom2 + inti_mol1.GetNumAtoms(),
                first_atom + inti_mol2.GetNumAtoms(),
                order=Chem.rdchem.BondType.SINGLE
            )
            inti_mol3 = edcombo.GetMol()

        else:
            combo = Chem.CombineMols(mol_without_dum1, mol_without_dum2)
            edcombo = Chem.EditableMol(combo)
            edcombo.AddBond(
                atom2,
                first_atom + mol_without_dum1.GetNumAtoms(),
                order=Chem.rdchem.BondType.SINGLE
            )
            inti_mol3 = edcombo.GetMol()

    return inti_mol3, monomer_mol, first_atom, second_atom + (length-1)*monomer_mol.GetNumAtoms()


def place_h_in_tetrahedral(mol, atom_idx, h_indices):
    """
    强行将 atom_idx 这个中心原子上的氢 (列表 h_indices) 与其余非氢配体
    一起摆成近似的sp3(正四面体)几何。
    假设总配体数=4（或小于4但要补成4）。这里演示核心几何思路。
    """
    conf = mol.GetConformer()
    center_atom = mol.GetAtomWithIdx(atom_idx)
    center_pos = np.array(conf.GetAtomPosition(atom_idx))

    # 拿到非氢配体
    neighbors = [nbr.GetIdx() for nbr in center_atom.GetNeighbors()]
    heavy_neighbors = [idx for idx in neighbors if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1]

    # 如果中心原子与非氢配体之和 + 氢原子总计不到4，可以看作要凑够 4
    # 比如有2个非氢配体，那就需要2个氢；有3个非氢配体，需要1个氢，等等
    # 这里假设加起来正好=4，或者你自己做检查/补氢操作
    # -------------------------------------------------------------------
    # 1. 计算已有非氢配体的向量
    heavy_vecs = []
    for hv in heavy_neighbors:
        hv_pos = np.array(conf.GetAtomPosition(hv))
        heavy_vecs.append(hv_pos - center_pos)
    # 2. 我们想得到理想四面体的 4 根单位向量 directions_tet
    directions_tet = _get_ideal_tetrahedral_vectors()
    # 这里 directions_tet 是形如 [v1, v2, v3, v4], 每个 v_i 是 3D 向量，互相夹角 ~109.47°，
    # 但它们在全局坐标系里都是固定的参考，要与当前 heavy_vecs 对齐。

    # 3. 将实际的 heavy_vecs 对齐到 ideal tetra 的前 n 个方向
    #    （最简单的思路：如果 n=1，就直接把 heavy_vecs[0] 对齐到 directions_tet[0]，
    #      如果 n=2，就让 heavy_vecs[0], heavy_vecs[1] 分别对齐 directions_tet[0], directions_tet[1]，等等）
    #    实际中需要构造一个最小 RMSD 的最佳旋转矩阵，这里演示用 rdkit 自带的对齐功能或numpy的Kabsch算法等。
    if len(heavy_vecs) > 0:
        # 先把 heavy_vecs 标准化
        heavy_vecs_array = np.array([v / np.linalg.norm(v) for v in heavy_vecs])
        target_vecs_array = np.array(directions_tet[:len(heavy_vecs)])
        # Kabsch 算法求旋转矩阵 R, 使 heavy_vecs_array * R ~ target_vecs_array
        R = _kabsch_rotation(heavy_vecs_array, target_vecs_array)
    else:
        # 如果没有非氢配体(很罕见)，那就啥也不用对齐，直接用 directions_tet 即可
        R = np.eye(3)

    # 4. 把对齐后的 directions_tet 应用这个旋转 R，得到与实际分子布局匹配的 4 个方向
    directions_aligned = []
    for v in directions_tet:
        directions_aligned.append(R @ v)  # numpy 矩阵乘法

    # 5. 分配哪些方向给非氢，哪些方向给氢
    #    heavy_neighbors 已经有 len(heavy_vecs) 个，把 directions_aligned[:len(heavy_vecs)] 给它们
    #    剩下 directions_aligned[len(heavy_vecs):] 就是氢的方向
    #    但实际上非氢配体原子不需要我们去改它坐标（它们已经有了真实坐标），只有氢需要调整
    #    这里演示一种策略，即只重定位氢原子的方向
    needed_h_count = 4 - len(heavy_neighbors)
    h_directions = directions_aligned[-needed_h_count:]

    # 6. 设置氢的坐标
    #    如果 h_indices 里面的氢数量 == needed_h_count，则一一对应
    #    否则你要根据实际情况进行映射/删除/补氢
    if len(h_indices) != needed_h_count:
        print("警告：氢原子数量和理论需要数量不一致，需要自行处理！")

    # 以典型 C—H 距离 1.09 Å 为例
    CH_BOND = 1.09

    for i, h_idx in enumerate(h_indices):
        h_dir = h_directions[i] / np.linalg.norm(h_directions[i])  # 归一化
        new_h_pos = center_pos + CH_BOND * h_dir
        conf.SetAtomPosition(h_idx, new_h_pos)


def _get_ideal_tetrahedral_vectors():
    """
    返回正四面体4个顶点相对于中心的归一化参考向量 (numpy array)。
    常用的一个简单表示是让它们尽量等距分布在单位球面上。
    下面仅给出一种常见 hard-coded 版本，互相夹角约 109.47 度。
    """
    # 这里给出一种坐标：在立方体对角空间中挑选4个顶点即可
    # 例如：
    vs = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]
    # 归一化
    vs = [np.array(v) / np.linalg.norm(v) for v in vs]
    return vs


def _kabsch_rotation(P, Q):
    """
    给定两个点集/向量集 P, Q (形状相同)，
    用 Kabsch 算法求最优旋转矩阵 R，使得 P*R ~ Q。
    P, Q 大小为 (n, 3)，n>=1。
    返回 3x3 的旋转矩阵 R (numpy)。
    """
    # 去质心 (不过我们只是方向向量，无需质心修正)
    # 直接算协方差矩阵:
    C = np.dot(P.T, Q)
    # 奇异值分解
    V, S, Wt = np.linalg.svd(C)
    # 检查行列式，避免反射
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
    if d:
        # 修正
        V[:, -1] = -V[:, -1]
    # 旋转矩阵
    R = np.dot(V, Wt)
    return R

def rotate_mol_around_axis(mol, axis, anchor, angle_rad):
    """
    对 mol 整体绕给定 axis（应为单位向量），
    以 anchor 为中心，旋转 angle_rad 弧度（右手法则）。
    当你只想旋转某个子集也可以针对性写循环，这里简单对 mol 的所有原子做旋转。
    """
    conf = mol.GetConformer()
    for atom_idx in range(mol.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(atom_idx))
        # 平移到 anchor，使 anchor 成为原点
        pos_shifted = pos - anchor
        # 绕 axis 旋转
        rot = R.from_rotvec(axis * angle_rad)
        pos_rot = rot.apply(pos_shifted)
        # 平移回去
        pos_new = pos_rot + anchor
        conf.SetAtomPosition(atom_idx, pos_new)

def gen_3D_info(monomer_mol, length, first_atom, second_atom, bond_length):

    new_mol = Chem.RWMol(monomer_mol)

    for i in range(1, length):
        # ---- 1) 计算单体原本的“参考轴” (unit_vector) ----
        conf_monomer_mol = monomer_mol.GetConformer()
        pos_first = np.array(conf_monomer_mol.GetAtomPosition(first_atom))
        pos_second = np.array(conf_monomer_mol.GetAtomPosition(second_atom))

        initial_vector = pos_second - pos_first
        initial_distance = np.linalg.norm(initial_vector)
        unit_vector = initial_vector / initial_distance  # 归一化

        # ---- 2) 找到当前链条“尾巴”信息 ----
        last_atom_index = second_atom + (i - 1) * monomer_mol.GetNumAtoms()
        pos_tail, direction_vector = get_vector(new_mol, last_atom_index, angle=120)
        pos_new_first_adjusted = pos_tail + direction_vector * bond_length

        # ---- 3) 将 monomer_mol 平移到正确的位置 ----
        translated_monomer = Chem.Mol(monomer_mol)
        translated_conf = translated_monomer.GetConformer()

        pos_new_first = np.array(conf_monomer_mol.GetAtomPosition(first_atom))
        translation = pos_new_first_adjusted - pos_new_first

        for atom_idx in range(translated_monomer.GetNumAtoms()):
            pos_old = np.array(translated_conf.GetAtomPosition(atom_idx))
            pos_new = pos_old + translation
            translated_conf.SetAtomPosition(atom_idx, pos_new)

        # ---- 4) 首先不旋转，直接尝试合并到聚合物 ----
        combo = Chem.CombineMols(new_mol, translated_monomer)
        editable_combo = Chem.EditableMol(combo)
        # 建立新键
        editable_combo.AddBond(
            second_atom + (i - 1) * monomer_mol.GetNumAtoms(),
            first_atom + i * monomer_mol.GetNumAtoms(),
            order=Chem.rdchem.BondType.SINGLE
        )
        final_mol = editable_combo.GetMol()

        # ---- 5) 检查是否重叠 ----
        check_mol = Chem.RWMol(final_mol)
        overlap = has_overlapping_atoms(check_mol)

        # 先令 final_mol2 = final_mol，表示“假设不需要旋转”
        final_mol2 = final_mol

        if overlap:
            print(f'[Step {i}] Overlap detected, trying 90-degree rotation...')
            # 撤销当前合并，用“未合并”的 new_mol (上一步的聚合物) 回退
            final_mol = new_mol.GetMol()

            # 对已经平移过的 translated_monomer 围绕 unit_vector 旋转 90°
            anchor = pos_new_first_adjusted
            rotate_mol_around_axis(translated_monomer, axis=unit_vector, anchor=anchor, angle_rad=np.pi/2)

            # 再次合并
            combo2 = Chem.CombineMols(new_mol, translated_monomer)
            editable_combo2 = Chem.EditableMol(combo2)
            editable_combo2.AddBond(
                second_atom + (i - 1) * monomer_mol.GetNumAtoms(),
                first_atom + i * monomer_mol.GetNumAtoms(),
                order=Chem.rdchem.BondType.SINGLE
            )
            final_mol2 = editable_combo2.GetMol()  # 用旋转后的单体得到的新分子

        # ---- 6) 无论是否 overlap，都对最终的 final_mol2 做氢的四面体化 ----
        check_mol2 = Chem.RWMol(final_mol2)
        atom_idx = first_atom + i * monomer_mol.GetNumAtoms()
        # 找到该中心原子的氢
        h_indices = [
            nbr.GetIdx() for nbr in check_mol2.GetAtomWithIdx(atom_idx).GetNeighbors()
            if nbr.GetAtomicNum() == 1
        ]
        place_h_in_tetrahedral(check_mol2, atom_idx, h_indices)

        # 把修正氢坐标后的分子赋回 final_mol2
        final_mol2 = check_mol2.GetMol()

        # 7) 更新 new_mol 供下一轮使用
        new_mol = Chem.RWMol(final_mol2)

    final_mol = new_mol.GetMol()
    return final_mol, first_atom, second_atom + (length - 1) * monomer_mol.GetNumAtoms()

def rotate_vector_to_align(a, b):
    # 归一化向量
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)

    # 计算旋转轴（a × b）
    cross_prod = np.cross(b_norm, a_norm)
    norm_cross = np.linalg.norm(cross_prod)

    if norm_cross < 1e-6:
        # 向量 a 和 b 已经平行或反向平行
        # 选择一个任意垂直于 a 的旋转轴
        arbitrary = np.array([1, 0, 0])
        if np.allclose(a_norm, arbitrary) or np.allclose(a_norm, -arbitrary):
            arbitrary = np.array([0, 1, 0])
        cross_prod = np.cross(a_norm, arbitrary)
        norm_cross = np.linalg.norm(cross_prod)
        rotation_axis = cross_prod / norm_cross
        # 计算旋转角度
        dot_prod = np.dot(a_norm, b_norm)
        angle_rad = np.pi if dot_prod < 0 else 0  # 180度或0度
    else:
        # 正常情况
        rotation_axis = cross_prod / norm_cross
        # 计算旋转角度
        dot_prod = np.dot(a_norm, b_norm)
        # 防止数值误差导致的arccos域错误
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        angle_rad = np.arccos(dot_prod)

    # 创建旋转对象
    rotation = R.from_rotvec(rotation_axis * angle_rad)
    return rotation

def get_vector(mol, index, angle):

    conf_mol = mol.GetConformer()
    atom = mol.GetAtomWithIdx(index)
    degree = atom.GetDegree()
    pos = np.array(conf_mol.GetAtomPosition(index))

    # 获取原子的邻居位置
    neighbors = atom.GetNeighbors()
    neb_pos = []
    for neighbor in neighbors:
        neighbor_idx = neighbor.GetIdx()
        neighbor_pos = conf_mol.GetAtomPosition(neighbor_idx)
        neb_pos.append(np.array([neighbor_pos.x, neighbor_pos.y, neighbor_pos.z]))

    v_norm = None
    if degree == 1:

        # 计算向量
        v = pos - neb_pos[0]
        v_norm = v / np.linalg.norm(v)

        # 设置旋转轴，这里使用z轴
        rotation_axis = np.array([0, 0, 1])

        # 旋转角度，从度转换为弧度
        rotation_degrees = angle  # 180 - 109.5
        rotation_radians = np.deg2rad(rotation_degrees)

        # 创建旋转对象
        rotation = R.from_rotvec(rotation_axis * rotation_radians)

        # 应用旋转到向量
        v_norm = rotation.apply(v_norm)

    elif degree > 1:

        sum_vi_norm = np.zeros(3)
        for i in range(degree):

            # 计算向量和归一化向量
            vi = pos - neb_pos[i]
            vi_norm = vi / np.linalg.norm(vi)

            sum_vi_norm += vi_norm

        # 计算向量 v_tail_norm
        v_norm = sum_vi_norm / np.linalg.norm(sum_vi_norm)

    return pos, v_norm

def connect_mols(mol1, mol2, tail_index, head_index, bond_length,
                 check_overlap=True, do_tetrahedral=True):
    """
    将 mol2 旋转、平移，与 mol1 通过 (tail_index) -- (head_index) 成单键拼接。
    可以检测是否重叠，如果重叠则尝试额外旋转并再次合并。
    若指定 do_tetrahedral=True，则对新加入的中心原子的氢作四面体化。
    """

    # 1. 先对齐 v1_norm 和 -v2_norm
    pos1_tail, v1_norm = get_vector(mol1, tail_index, angle=60)
    pos2_head, v2_norm = get_vector(mol2, head_index, angle=60)

    # 计算旋转，让 v1_norm 对齐到 -v2_norm
    rot = rotate_vector_to_align(v1_norm, -v2_norm)

    # 把 mol2 所有原子坐标收集，做旋转
    conf_2 = mol2.GetConformer()
    all_coords = []
    for atom_idx in range(mol2.GetNumAtoms()):
        p = conf_2.GetAtomPosition(atom_idx)
        all_coords.append(np.array([p.x, p.y, p.z]))

    # 对 mol2 施加旋转
    center = pos2_head  # 以 head 原子为锚点
    rotated_coords = []
    for pos in all_coords:
        translated = pos - center
        rotated = rot.apply(translated)
        new_pos = rotated + center
        rotated_coords.append(new_pos)

    # 更新 mol2 的 Conformer
    rotated_conf = Chem.Conformer(mol2.GetNumAtoms())
    for idx, pos in enumerate(rotated_coords):
        rotated_conf.SetAtomPosition(idx, pos)
    rotated_mol = Chem.Mol(mol2)
    rotated_mol.RemoveAllConformers()
    rotated_mol.AddConformer(rotated_conf, assignId=True)

    # 2. 再平移使得 head_index 与 mol1 的 tail_index 间距 = bond_length
    pos2_adjusted = pos1_tail + v1_norm * bond_length
    translation = pos2_adjusted - pos2_head

    new_conf = rotated_mol.GetConformer()
    for atom_idx in range(rotated_mol.GetNumAtoms()):
        pos = np.array([
            new_conf.GetAtomPosition(atom_idx).x,
            new_conf.GetAtomPosition(atom_idx).y,
            new_conf.GetAtomPosition(atom_idx).z
        ])
        pos_new = pos + translation
        new_conf.SetAtomPosition(atom_idx, pos_new)

    # 3. CombineMols + 加键
    combo = Chem.CombineMols(mol1, rotated_mol)
    editable_combo = Chem.EditableMol(combo)
    editable_combo.AddBond(
        tail_index,
        head_index + mol1.GetNumAtoms(),
        order=Chem.rdchem.BondType.SINGLE
    )
    mol_connected = editable_combo.GetMol()

    # 4. 如果需要 overlap 检测
    if check_overlap:
        rw_mol = Chem.RWMol(mol_connected)
        overlap_flag = has_overlapping_atoms(rw_mol)
        if overlap_flag:
            print("Overlap detected. Trying extra rotation on second molecule...")

            # ---- 撤销上一次合并 ----
            # 先回退到 connect 前的 mol1
            mol_connected = mol1  # 这样回到还没合并的状态

            # 额外旋转 rotated_mol 围绕 v1_norm 旋转 90 度(示例)
            from math import pi
            rotate_mol_around_axis(rotated_mol, axis=v1_norm,
                                   anchor=pos1_tail, angle_rad=pi/3)

            # 再次合并
            combo2 = Chem.CombineMols(mol1, rotated_mol)
            editable_combo2 = Chem.EditableMol(combo2)
            editable_combo2.AddBond(
                tail_index,
                head_index + mol1.GetNumAtoms(),
                order=Chem.rdchem.BondType.SINGLE
            )
            mol_connected_2 = editable_combo2.GetMol()

            # 再次检测
            rw_mol2 = Chem.RWMol(mol_connected_2)
            overlap_flag2 = has_overlapping_atoms(rw_mol2)
            if overlap_flag2:
                print("Even after extra rotation, still overlap. Keeping the new version anyway.")
                mol_connected = mol_connected_2
            else:
                mol_connected = mol_connected_2

    # 5. 如果需要，对“新加的接头原子”做氢的四面体化
    if do_tetrahedral:
        # 假设要对 mol2 的 head_index 那个碳做氢的四面体化
        # 它在合并后索引变为 head_index + mol1.GetNumAtoms()
        new_head_idx = head_index + mol1.GetNumAtoms()

        rw_connected = Chem.RWMol(mol_connected)
        # 找出这个原子的氢
        h_indices = [nbr.GetIdx() for nbr in rw_connected.GetAtomWithIdx(new_head_idx).GetNeighbors()
                     if rw_connected.GetAtomWithIdx(nbr.GetIdx()).GetAtomicNum() == 1]
        if h_indices:
            place_h_in_tetrahedral(rw_connected, new_head_idx, h_indices)
            mol_connected = rw_connected.GetMol()

    return mol_connected

def gen_3D_nocap(dum1, dum2, atom1, atom2, smiles_each, length, max_retries,):

    seg = 1
    for attempt in range(max_retries):
        # Connect the units and caps to obtain SMILES structure
        input_mol = Chem.MolFromSmiles(smiles_each)
        rw_mol = Chem.RWMol(input_mol)
        for atom in rw_mol.GetAtoms():
            if atom.GetSymbol() == '*':
                atom.SetAtomicNum(53)  # Iodine (as a placeholder for dummy atoms)
        rw_mol = Chem.AddHs(rw_mol)

        AllChem.EmbedMolecule(rw_mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(rw_mol)

        edit_m3 = Chem.EditableMol(rw_mol)

        edit_m3.RemoveAtom(dum1)  # Delete dum1
        if dum1 < dum2:
            edit_m3.RemoveAtom(dum2 - 1)  # Adjust for index shift after removing dum1
        else:
            edit_m3.RemoveAtom(dum2)
        monomer_mol = Chem.Mol(edit_m3.GetMol())

        # Adjust the order of atoms and set connection points
        if atom1 > atom2:
            atom1, atom2 = atom2, atom1

        def adjust_atom_index(dum1, dum2, atom):
            if dum1 < atom and dum2 < atom:
                return atom - 2
            elif (dum1 < atom < dum2) or (dum1 > atom > dum2):
                return atom - 1
            else:
                return atom

        first_atom = adjust_atom_index(dum1, dum2, atom1)
        second_atom = adjust_atom_index(dum1, dum2, atom2)

        bond_length = 1.5
        if length == 1:
            mol_nocap = monomer_mol
            return mol_nocap, monomer_mol, first_atom, second_atom

        if 1 < length < 6:

            mol_nocap, first_atom, second_atom = gen_3D_info(
                monomer_mol,
                length,
                first_atom,
                second_atom,
                bond_length,
            )
            return mol_nocap, monomer_mol, first_atom, second_atom

        if length > 5:

            num = length // int(seg)
            add = length % int(seg)

            mol_len5, first_atom_len5, second_atom_len5 = gen_3D_info(
                monomer_mol = monomer_mol,
                length = seg,
                first_atom = first_atom,
                second_atom = second_atom,
                bond_length = bond_length
            )

            mol_add, first_atom_len_add, second_atom_len_add = gen_3D_info(
                monomer_mol = monomer_mol,
                length = add,
                first_atom = first_atom,
                second_atom = second_atom,
                bond_length = bond_length
            )

            mol_main = Chem.Mol(mol_len5)

            for i in range(1, num):

                tail_index = second_atom_len5 + (i - 1) * mol_len5.GetNumAtoms()
                head_index = first_atom_len5

                mol_main = connect_mols(
                    mol_main,
                    mol_len5,
                    tail_index,
                    head_index,
                    bond_length=1.54,
                )

            if add > 0:

                # Add the remaining units
                tail_index = second_atom_len5 + (num - 1) * mol_len5.GetNumAtoms()
                head_index = first_atom_len_add

                mol_main = connect_mols(
                    mol_main,
                    mol_add,
                    tail_index,
                    head_index,
                    bond_length=1.54,
                )

        # check 3D structure
        mol_nocap = mol_main
        # return mol_nocap, monomer_mol, first_atom, second_atom + (length - 1) * monomer_mol.GetNumAtoms()
        overlap = has_overlapping_atoms(mol_nocap,)
        if not overlap:
            return mol_nocap, monomer_mol, first_atom, second_atom + (length - 1) * monomer_mol.GetNumAtoms()
        else:
            print(f"Attempt nocap {attempt + 1}. Retrying...")
            seg += 1
            print(seg)
            continue


def gen_3D_withcap(mol, start_atom, end_atom, max_retries=50,):

    for attempt in range(max_retries):

        capped_mol = Chem.RWMol(mol)
        terminal_atoms = [start_atom, end_atom]

        # 定义要添加的封端基团信息
        capping_info = []
        for terminal_idx in terminal_atoms:
            atom = capped_mol.GetAtomWithIdx(terminal_idx)
            h_count = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1)
            if atom.GetAtomicNum() == 6 and h_count == 2:
                capping_info.append({'type': 'H', 'atom_idx': terminal_idx})
            else:
                capping_info.append({'type': 'CH3', 'atom_idx': terminal_idx})

        for cap in capping_info:

            terminal_idx = cap['atom_idx']
            if cap['type'] == 'H':

                terminal_pos, v_norm = get_vector(capped_mol, terminal_idx, angle=109)

                C_H_bond_length = 1.12
                H_pos = terminal_pos + v_norm * C_H_bond_length

                new_atom_positions = {}

                # 添加新氢原子到分子中
                new_H = Chem.Atom(1)
                editable_mol = Chem.EditableMol(capped_mol)
                new_H_idx = editable_mol.AddAtom(new_H)
                new_atom_positions[new_H_idx] = H_pos

                editable_mol.AddBond(
                    terminal_idx,
                    new_H_idx,
                    Chem.BondType.SINGLE
                )

                capped_mol = editable_mol.GetMol()
                conformer = capped_mol.GetConformer()
                conformer.SetAtomPosition(new_H_idx, Point3D(*H_pos))

            elif cap['type'] == 'CH3':

                smi_C = 'C'
                mol_C = Chem.MolFromSmiles(smi_C)
                mol_C = Chem.AddHs(mol_C)
                AllChem.EmbedMolecule(mol_C, AllChem.ETKDG())
                h_atoms = [atom.GetIdx() for atom in mol_C.GetAtoms() if atom.GetSymbol() == 'H']
                editable_mol = Chem.EditableMol(mol_C)
                remove_h_indx = h_atoms[0]
                editable_mol.RemoveAtom(remove_h_indx)
                mol_C = editable_mol.GetMol()

                tail_index = terminal_idx
                head_index = [atom.GetIdx() for atom in mol_C.GetAtoms() if atom.GetSymbol() == 'C'][0]
                capped_mol = connect_mols(
                    capped_mol,
                    mol_C,
                    tail_index,
                    head_index,
                    bond_length=1.54,
                )

        # return capped_mol

        # 检查原子间距离是否合理
        overlap= has_overlapping_atoms(capped_mol)
        if not overlap:
            return capped_mol
        else:
            print(f"Attempt {attempt + 1}. Retrying...")
            continue
    # return capped_mol

def get_min_distance(atom1, atom2, bond_graph):
    # 检查原子1和原子2是否相连
    if bond_graph.has_edge(atom1, atom2):
        # 如果相连，则使用较小的距离（1.1 Å）
        return 1.0
    else:
        # 如果不相连，则使用较大的距离（1.2 Å）
        return 1.2

def  has_overlapping_atoms(mol):
    # 获取分子的拓扑结构
    mol_graph = Chem.rdmolops.GetAdjacencyMatrix(mol)

    # 获取所有原子的坐标
    conf = mol.GetConformer()
    positions = conf.GetPositions()

    # 获取分子中的原子类型
    atom_types = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]

    # 创建一个无向图用于存储原子间的连接关系
    bond_graph = nx.Graph()

    # 将分子中的原子添加为节点
    for i in range(mol.GetNumAtoms()):
        bond_graph.add_node(i, position=positions[i])

    # 添加边：通过 RDKit 获取化学键连接的原子对
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        bond_graph.add_edge(atom1, atom2)

    # 创建一个无向图存储计算的重叠关系
    G = nx.Graph()

    # 计算原子之间的距离并添加边
    num_atoms = mol.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(positions[i] - positions[j])
            # 获取原子对的最小距离
            actual_min_distance = get_min_distance(i, j, bond_graph)
            # 如果原子间的距离小于实际的最小距离，则认为原子发生重叠，添加边
            if distance < actual_min_distance:
                G.add_edge(i, j, weight=distance)

    # 判断图中是否有边（即是否存在重叠原子）
    if len(G.edges) > 0:
        return True  # 存在重叠原子
    else:
        return False  # 没有重叠原子


def gen_smiles_with_cap(poly_name, inti_mol, first_atom, second_atom, smiles_LCap_, smiles_RCap_):

    # Add cap to main chain
    main_mol_Dum = inti_mol

    # Left Cap
    if not smiles_LCap_:
        # Decide whether to cap with H or CH3
        first_atom_obj = main_mol_Dum.GetAtomWithIdx(first_atom)
        atomic_num = first_atom_obj.GetAtomicNum() # Find its element
        degree = first_atom_obj.GetDegree() # Find the number of bonds associated with it
        num_implicit_hs = first_atom_obj.GetNumImplicitHs()
        num_explicit_hs = first_atom_obj.GetNumExplicitHs()
        total_hs = num_implicit_hs + num_explicit_hs # Find the number of hydrogen associated with it

        if atomic_num == 6 and degree == 2 and total_hs == 2:
            pass
        else:
            # Cap with CH3
            cap_mol = Chem.MolFromSmiles('C')
            cap_add = cap_mol.GetNumAtoms()
            combo = Chem.CombineMols(cap_mol, main_mol_Dum)
            edcombo = Chem.EditableMol(combo)
            edcombo.AddBond(0, cap_add + first_atom, order=Chem.rdchem.BondType.SINGLE)
            main_mol_Dum = edcombo.GetMol()
            second_atom += cap_add  # adjust the index of second_atom

    else:
        # Existing code for handling Left Cap
        (
            unit_name,
            dum_L,
            atom_L,
            neigh_atoms_info_L,
            flag_L
        ) = Init_info_Cap(
            poly_name,
            smiles_LCap_
        )

        # Reject if SMILES is not correct
        if flag_L == 'REJECT':
            return unit_name, 'REJECT', 0

        # Editable Mol for LeftCap
        LCap_m1 = Chem.MolFromSmiles(smiles_LCap_)
        LCap_edit_m1 = Chem.EditableMol(LCap_m1)

        # Remove dummy atoms
        LCap_edit_m1.RemoveAtom(dum_L)

        # Mol without dummy atom
        LCap_m1 = LCap_edit_m1.GetMol()
        LCap_add = LCap_m1.GetNumAtoms()
        # Linking atom
        if dum_L < atom_L:
            LCap_atom = atom_L - 1
        else:
            LCap_atom = atom_L

        # Join main chain with Left Cap
        combo = Chem.CombineMols(LCap_m1, inti_mol)
        edcombo = Chem.EditableMol(combo)
        edcombo.AddBond(
            LCap_atom, first_atom + LCap_add, order=Chem.rdchem.BondType.SINGLE
        )
        main_mol_Dum = edcombo.GetMol()
        second_atom += LCap_add     # adjust the index of second_atom

    # Right Cap
    if not smiles_RCap_:
        # Decide whether to cap with H or CH3
        second_atom_obj = main_mol_Dum.GetAtomWithIdx(second_atom)
        atomic_num = second_atom_obj.GetAtomicNum()
        degree = second_atom_obj.GetDegree()
        num_implicit_hs = second_atom_obj.GetNumImplicitHs()
        num_explicit_hs = second_atom_obj.GetNumExplicitHs()
        total_hs = num_implicit_hs + num_explicit_hs

        if atomic_num == 6 and degree == 2 and total_hs == 2:
            pass
        else:
            # Cap with CH3
            cap_mol = Chem.MolFromSmiles('C')
            cap_add = cap_mol.GetNumAtoms()
            combo = Chem.CombineMols(main_mol_Dum, cap_mol)
            edcombo = Chem.EditableMol(combo)
            edcombo.AddBond(
                second_atom,
                main_mol_Dum.GetNumAtoms(),  # Index of the cap atom
                order=Chem.rdchem.BondType.SINGLE,
            )
            main_mol_Dum = edcombo.GetMol()
    else:
        # Existing code for handling Right Cap
        (
            unit_name,
            dum_R,
            atom_R,
            neigh_atoms_info_R,
            flag_R
        ) = Init_info_Cap(
            poly_name,
            smiles_RCap_
        )

        # Reject if SMILES is not correct
        if flag_R == 'REJECT':
            return unit_name, 'REJECT', 0

        # Editable Mol for RightCap
        RCap_m1 = Chem.MolFromSmiles(smiles_RCap_)
        RCap_edit_m1 = Chem.EditableMol(RCap_m1)

        # Remove dummy atoms
        RCap_edit_m1.RemoveAtom(dum_R)

        # Mol without dummy atom
        RCap_m1 = RCap_edit_m1.GetMol()

        # Linking atom
        if dum_R < atom_R:
            RCap_atom = atom_R - 1
        else:
            RCap_atom = atom_R

        # Join main chain with Right Cap
        combo = Chem.CombineMols(main_mol_Dum, RCap_m1)
        edcombo = Chem.EditableMol(combo)
        edcombo.AddBond(
            second_atom,
            RCap_atom + main_mol_Dum.GetNumAtoms(),
            order=Chem.rdchem.BondType.SINGLE,
        )
        main_mol_Dum = edcombo.GetMol()

    # Remove remain dummy atoms (virtual atoms marked as [*])
    atoms_to_remove = []

    for atom in main_mol_Dum.GetAtoms():
        if atom.GetSymbol() == '*' or atom.GetAtomicNum() == 0:  # Checking for virtual atom
            atoms_to_remove.append(atom.GetIdx())

    # Create a new editable molecule to remove the atoms
    edmol_final = Chem.RWMol(main_mol_Dum)
    # Reverse traversal to prevent index changes
    for atom_idx in reversed(atoms_to_remove):
        edmol_final.RemoveAtom(atom_idx)

    main_mol_noDum = edmol_final.GetMol()
    main_mol_noDum = Chem.RemoveHs(main_mol_noDum)

    return main_mol_noDum

def Init_info_Cap(unit_name, smiles_each_ori):
    # Get index of dummy atoms and bond type associated with it in cap
    try:
        dum_index, bond_type = FetchDum(smiles_each_ori)
        if len(dum_index) == 1:
            dum1 = dum_index[0]
        else:
            print(
                unit_name,
                ": There are more or less than one dummy atoms in the SMILES string; ",
            )
            return unit_name, 0, 0, 0, 'REJECT'
    except Exception:
        print(
            unit_name,
            ": Couldn't fetch the position of dummy atoms. Hints: (1) In SMILES string, use '*' for a dummy atom,"
            "(2) Check RDKit installation.",
        )
        return unit_name, 0, 0, 0, 'REJECT'

    # Replace '*' with dummy atom to find atoms associated with it
    smiles_each = smiles_each_ori.replace(r'*', 'Cl')

    # Convert SMILES to XYZ coordinates
    convert_smiles2xyz = smile_toxyz(unit_name, smiles_each,)

    # if fails to get XYZ coordinates; STOP
    if convert_smiles2xyz == 'NOT_DONE':
        print(
            unit_name,
            ": Couldn't get XYZ coordinates from SMILES string. Hints: (1) Check SMILES string,"
            "(2) Check RDKit installation.",
        )
        return unit_name, 0, 0, 0, 'REJECT'

    # Collect valency and connecting information for each atom
    neigh_atoms_info = connec_info('./' + unit_name + '.xyz')

    try:
        # Find connecting atoms associated with dummy atoms.
        # dum1 and dum2 are connected to atom1 and atom2, respectively.
        atom1 = neigh_atoms_info['NeiAtom'][dum1].copy()[0]

    except Exception:
        print(
            unit_name,
            ": Couldn't get the position of connecting atoms. Hints: (1) XYZ coordinates are not acceptable,"
            "(2) Check Open Babel installation.",
        )
        return unit_name, 0, 0, 0, 'REJECT'
    return (
        unit_name,
        dum1,
        atom1,
        neigh_atoms_info,
        '',
    )

def remove_numbers_from_residue_names(mol2_filename, resname):
    with open(mol2_filename, 'r') as file:
        content = file.read()

    # 使用正则表达式删除特定残基名称后的数字1（确保只删除末尾的数字1）
    updated_content = re.sub(r'({})1\b'.format(resname), r'\1', content)

    with open(mol2_filename, 'w') as file:
        file.write(updated_content)

def extract_from_top(top_file, out_itp_file, nonbonded=False, bonded=False):
    sections_to_extract = []
    if nonbonded:
        sections_to_extract = ["[ atomtypes ]"]
    elif bonded:
        sections_to_extract = ["[ moleculetype ]", "[ atoms ]", "[ bonds ]", "[ pairs ]", "[ angles ]", "[angles]", "[ dihedrals ]"]

        # 打开 .top 文件进行读取
    with open(top_file, 'r') as file:
        lines = file.readlines()

    # 初始化变量以存储提取的信息
    extracted_lines = []
    current_section = None

    # 遍历所有行，提取相关部分
    for line in lines:
        if line.strip() in sections_to_extract:
            current_section = line.strip()
            extracted_lines.append(line)  # 添加部分标题
        elif current_section and line.strip().startswith(";"):
            extracted_lines.append(line)  # 添加注释行
        elif current_section and line.strip():
            extracted_lines.append(line)  # 添加数据行
        elif line.strip() == "" and current_section:
            extracted_lines.append("\n")  # 添加部分之间的空行
            current_section = None  # 重置当前部分

    # 写入提取的内容到 bonded.itp 文件
    with open(out_itp_file, 'w') as file:
        file.writelines(extracted_lines)

def read_energy_from_xtb(filename):
    """从xtb的输出文件中读取能量值"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    # 假设能量在输出文件的第二行
    energy_line = lines[1]
    energy = float(energy_line.split()[1])
    return energy

def std_xyzfile(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    structure_count = int(lines[0].strip())  # Assuming the first line contains the number of atoms

    # Process each structure in the file
    for i in range(0, len(lines), structure_count + 2):  # +2 for the atom count and energy lines
        # Copy the atom count line
        modified_lines.append(lines[i])

        # Extract and process the energy value line
        energy_line = lines[i + 1].strip()
        energy_value = energy_line.split()[1]  # Extract the energy value
        modified_lines.append(f" {energy_value}\n")  # Reconstruct the energy line with only the energy value

        # Copy the atom coordinate lines
        for j in range(i + 2, i + structure_count + 2):
            modified_lines.append(lines[j])

    # Overwrite the original file with modified lines
    with open(file_path, 'w') as file:
        file.writelines(modified_lines)

def convert_chk_to_fchk(chk_file_path):
    fchk_file_path = chk_file_path.replace('.chk', '.fchk')

    try:
        subprocess.run(
            ['formchk', chk_file_path, fchk_file_path],
            check=True,
            stdout=subprocess.DEVNULL,  # Redirect standard output
            stderr=subprocess.DEVNULL   # Redirect standard error output
        )
    except subprocess.CalledProcessError as e:
        print(f"Error converting chk to fchk: {e}")

def calc_mol_weight(pdb_file):
    try:
        mol = Chem.MolFromPDBFile(pdb_file, removeHs=False, sanitize=False)
        if mol:
            Chem.SanitizeMol(mol)
            return Descriptors.MolWt(mol)
        else:
            raise ValueError(f"RDKit 无法解析 PDB 文件: {pdb_file}")
    except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException, ValueError):
        # 如果 RDKit 解析失败，尝试手动计算分子量
        try:
            atom_counts = defaultdict(int)
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith(("ATOM", "HETATM")):
                        element = line[76:78].strip()
                        if not element:
                            # 从原子名称推断元素符号
                            atom_name = line[12:16].strip()
                            element = ''.join([char for char in atom_name if char.isalpha()]).upper()[:2]
                        atom_counts[element] += 1

            # 常见元素的原子质量（g/mol）
            atomic_weights = {
                'H': 1.008,
                'C': 12.011,
                'N': 14.007,
                'O': 15.999,
                'F': 18.998,
                'P': 30.974,
                'S': 32.06,
                'CL': 35.45,
                'BR': 79.904,
                'I': 126.904,
                'FE': 55.845,
                'ZN': 65.38,
                # 根据需要添加更多元素
            }

            mol_weight = 0.0
            for atom, count in atom_counts.items():
                weight = atomic_weights.get(atom.upper())
                if weight is None:
                    raise ValueError(f"未知的原子类型 '{atom}' 在 PDB 文件: {pdb_file}")
                mol_weight += weight * count
            return mol_weight
        except Exception as e:
            raise ValueError(f"无法计算分子量，PDB 文件: {pdb_file}，错误: {e}")



def print_compounds(info_dict, key_name):
    """
    This function recursively searches for and prints 'compound' entries in a nested dictionary.
    """
    compounds = []
    for key, value in info_dict.items():
        # If the value is a dictionary, we make a recursive call
        if isinstance(value, dict):
            compounds.extend(print_compounds(value,key_name))
        # If the key is 'compound', we print its value
        elif key == key_name:
            compounds.append(value)
    return compounds


def extract_volume(partition, module_soft, edr_file, output_file='volume.xvg', option_id='21'):

    if partition == 'gpu':
        command = f"module load {module_soft} && echo {option_id} | gmx energy -f {edr_file} -o {output_file}"
    else:
        command = f"module load {module_soft} && echo {option_id} | gmx_mpi energy -f {edr_file} -o {output_file}"

    # 使用subprocess.run执行命令，由于这里使用bash -c，所以stdin的传递方式需要调整
    try:
        # Capture_output=True来捕获输出，而不是使用PIPE
        process = subprocess.run(['bash', '-c', command], capture_output=True, text=True, check=True)
        # 检查输出，无需单独检查returncode，因为check=True时如果命令失败会抛出异常
        print(f"Output: {process.stdout}")
        return process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.stderr}")
        return None


def read_volume_data(volume_file):

    volumes = []
    with open(volume_file, 'r') as file:
        for line in file:
            if line.startswith(('@', '#')):
                continue
            parts = line.split()
            volumes.append(float(parts[1]))

    return np.array(volumes)


def analyze_volume(volumes, start, dt_collection):

    start_time = int(start) / dt_collection
    average_volume = np.mean(volumes[int(start_time):])
    closest_index = np.argmin(np.abs(volumes - average_volume))
    return average_volume, closest_index


def extract_structure(partition, module_soft, tpr_file, xtc_file, save_gro_file, frame_time):

    if partition == 'gpu':
        command = (f"module load {module_soft} && echo 0 | gmx trjconv -s {tpr_file} -f {xtc_file} -o {save_gro_file} "
                   f"-dump {frame_time} -quiet")
    else:
        command = (f"module load {module_soft} && echo 0 | gmx_mpi trjconv -s {tpr_file} -f {xtc_file} -o {save_gro_file} "
                   f"-dump {frame_time} -quiet")

    # 使用 subprocess.run 执行命令，以更安全地处理外部命令
    try:
        # 使用 subprocess.run，避免使用shell=True以增强安全性
        process = subprocess.run(['bash', '-c', command], capture_output=True, text=True, check=True)
        print(f"Output: {process.stdout}")
        return process.stdout
    except subprocess.CalledProcessError as e:
        # 错误处理：打印错误输出并返回None
        print(f"Error executing command: {e.stderr}")
        return None


def calculate_box_size(numbers, pdb_files, density):
    total_mass = 0
    for num, file in zip(numbers, pdb_files):

        molecular_weight = calc_mol_weight(file)  # in g/mol
        total_mass += molecular_weight * num / 6.022e23  # accumulate mass of each molecule in grams

    total_volume = total_mass / density  # volume in cm^3
    length = (total_volume * 1e24) ** (1 / 3)  # convert to Angstroms
    return length


# def mol_to_networkx(mol):
#     """
#     将 RDKit 的 mol 对象转换为 networkx 图
#     节点属性包括原子类型
#     """
#     G = nx.Graph()
#     for atom in mol.GetAtoms():
#         G.add_node(atom.GetIdx(), element=atom.GetSymbol())
#     for bond in mol.GetBonds():
#         G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=str(bond.GetBondType()))
#     return G
#
#
# def get_atom_mapping(mol1, mol2):
#     """
#     获取两个 mol 对象的原子对应关系
#     返回 mapping: dict 从 mol1 节点索引到 mol2 节点索引
#     """
#     G1 = mol_to_networkx(mol1)
#     G2 = mol_to_networkx(mol2)
#
#     # 定义节点匹配函数，基于原子元素
#     def node_match(n1, n2):
#         return n1['element'] == n2['element']
#
#     # 创建同构匹配对象
#     gm = isomorphism.GraphMatcher(G1, G2, node_match=node_match)
#
#     if gm.is_isomorphic():
#         # 返回一个可能的匹配字典，从 G1 节点到 G2 节点
#         mapping = gm.mapping
#         return mapping
#     else:
#         return None
#
#

from openbabel import openbabel

# 定义原子序数到元素符号的映射表
periodic_table = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Fl', 'Lv', 'Ts', 'Og'
]


# RDKit mol 转换为 NetworkX 图
def mol_to_networkx_rdkit(mol, include_h=True):
    """
    将 RDKit 的 mol 对象转换为 networkx 图
    """
    G = nx.Graph()
    for atom in mol.GetAtoms():
        if not include_h and atom.GetSymbol() == 'H':
            continue
        G.add_node(atom.GetIdx(), element=atom.GetSymbol())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=str(bond.GetBondType()))
    return G


# Open Babel mol 转换为 NetworkX 图
def mol_to_networkx_ob(mol, include_h=True):
    """
    将 Open Babel 的 mol 对象转换为 networkx 图
    """
    G = nx.Graph()

    # 添加原子节点，调整索引从1开始到0开始
    for atom in openbabel.OBMolAtomIter(mol):
        atomic_num = atom.GetAtomicNum()  # 获取原子序数
        if atomic_num < 1 or atomic_num > len(periodic_table):
            element = 'Unknown'
        else:
            element = periodic_table[atomic_num - 1]  # 获取标准元素符号
        if not include_h and element == 'H':
            continue
        # 添加节点，索引减1
        G.add_node(atom.GetIdx() - 1, element=element)

    # 添加化学键边，调整原子索引从1开始到0开始
    for bond in openbabel.OBMolBondIter(mol):
        bond_order = bond.GetBondOrder()
        # 统一键类型
        if bond_order == 1:
            bond_type = 'SINGLE'
        elif bond_order == 2:
            bond_type = 'DOUBLE'
        elif bond_order == 3:
            bond_type = 'TRIPLE'
        else:
            bond_type = 'SINGLE'  # 默认处理
        # Open Babel 的 GetBeginAtomIdx() 和 GetEndAtomIdx() 从1开始，需要减1
        G.add_edge(bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1, bond_type=str(bond_type))

    return G


def get_atom_mapping(mol1, mol2, include_h=True):
    """
    获取两个 mol 对象的原子对应关系
    返回 mapping: dict 从 mol1 节点索引到 mol2 节点索引
    """
    # 将 mol1 和 mol2 分别转换为 NetworkX 图
    G1 = mol_to_networkx_rdkit(mol1, include_h=include_h)  # 对 RDKit mol 对象生成 NetworkX 图
    G2 = mol_to_networkx_ob(mol2, include_h=include_h)  # 对 Open Babel mol 对象生成 NetworkX 图

    # 打印图的信息（调试输出）
    print(f"G1 nodes: {G1.nodes(data=True)}")
    print(f"G2 nodes: {G2.nodes(data=True)}")
    print(f"G1 edges: {G1.edges(data=True)}")
    print(f"G2 edges: {G2.edges(data=True)}")

    # 定义节点匹配函数，基于原子元素
    def node_match(n1, n2):
        return (n1['element'] == n2['element']) and (n1.get('charge', 0) == n2.get('charge', 0))

    def bond_match(b1, b2):
        return b1['bond_type'] == b2['bond_type']

    # 创建同构匹配对象
    gm = isomorphism.GraphMatcher(G1, G2, node_match=node_match, edge_match=bond_match)
    if gm.is_isomorphic():
        # 返回一个可能的匹配字典，从 G1 节点到 G2 节点
        mapping = gm.mapping
        return mapping
    else:
        return None


def reorder_atoms(mol_3D, mapping):
    """
    根据 mapping（mol1_idx -> mol2_idx），
    让 mol1 的原子顺序与 mol2 相同。
    """
    # 创建反向映射：mol2_idx -> mol1_idx
    reverse_mapping = {v: k for k, v in mapping.items()}

    # 确保 mol2 的原子数量不超过 mol1
    num_atoms = mol_3D.GetNumAtoms()
    new_order = []
    for i in range(num_atoms):
        if i in reverse_mapping:
            new_order.append(reverse_mapping[i])
        else:
            # 如果某个 mol2 的原子在 mol1 中没有对应，则保留原顺序
            new_order.append(i)

    # 调整 mol_3D 中的原子顺序
    reordered_mol_3D = Chem.RenumberAtoms(mol_3D, new_order)

    return reordered_mol_3D

def convert_rdkit_to_openbabel(rdkit_mol):
    """
    将 RDKit 的 Mol 对象转换为 Open Babel 的 OBMol 对象
    """
    # 创建 Open Babel 的 OBConversion 对象
    ob_conversion = openbabel.OBConversion()
    ob_conversion.SetOutFormat("mol")  # 使用 MOL 格式

    # 将 RDKit 的 Mol 对象转换为 MolBlock 字符串
    mol_block = Chem.MolToMolBlock(rdkit_mol)

    # 使用 Open Babel 解析 MolBlock 并生成 OBMol 对象
    ob_mol = openbabel.OBMol()
    ob_conversion.ReadString(ob_mol, mol_block)

    return ob_mol


# Convert file type
# 1. Convert PDB to XYZ
def convert_pdb_to_xyz(pdb_filename, xyz_filename):
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "xyz")

    mol = ob.OBMol()
    obConversion.ReadFile(mol, pdb_filename)

    obConversion.WriteFile(mol, xyz_filename)

# 2. Convert XYZ to PDB
def convert_xyz_to_pdb(xyz_filename, pdb_filename, molecule_name, resname):
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "pdb")

    mol = ob.OBMol()
    obConversion.ReadFile(mol, xyz_filename)
    mol.SetTitle(molecule_name)

    for atom in ob.OBMolAtomIter(mol):
        res = atom.GetResidue()
        res.SetName(resname)
    obConversion.WriteFile(mol, pdb_filename)

# 3. Convert XYZ to MOL2
def convert_xyz_to_mol2(xyz_filename, mol2_filename, molecule_name, resname):
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "mol2")

    mol = ob.OBMol()
    obConversion.ReadFile(mol, xyz_filename)

    mol.SetTitle(molecule_name)

    for atom in ob.OBMolAtomIter(mol):
        res = atom.GetResidue()
        if res:  # 确保残基信息存在
            res.SetName(resname)

    obConversion.WriteFile(mol, mol2_filename)

    remove_numbers_from_residue_names(mol2_filename, resname)

# 4.Convert GRO to PDB
def convert_gro_to_pdb(input_gro_path, output_pdb_path):
    try:
        # Load the GRO file
        mol = next(pybel.readfile('gro', input_gro_path))

        # Save as PDB
        mol.write('pdb', output_pdb_path, overwrite=True)
        print(f"Gro converted to pdb successfully {output_pdb_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# 5. Convert LOG to XYZ
def log_to_xyz(log_file_path, xyz_file_path):
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("g09", "xyz")
    mol = ob.OBMol()

    try:
        obConversion.ReadFile(mol, log_file_path)
        obConversion.WriteFile(mol, xyz_file_path)
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

# 6. Convert SMILES to PDB
def smiles_to_pdb(smiles, output_file, molecule_name, resname):
    try:
        # Generate molecule object from SMILES string
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string. Please check the input string.")

        # Add hydrogens to the molecule
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:
            raise ValueError("Cannot embed the molecule into a 3D space.")
        AllChem.UFFOptimizeMolecule(mol)

        # Write molecule to a temporary SDF file
        tmp_sdf = "temp.sdf"
        with Chem.SDWriter(tmp_sdf) as writer:
            writer.write(mol)

        # Convert SDF to PDB using OpenBabel
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("sdf", "pdb")
        obmol = ob.OBMol()
        if not obConversion.ReadFile(obmol, tmp_sdf):
            raise IOError("Failed to read from the temporary SDF file.")

        # Set molecule name in OpenBabel
        obmol.SetTitle(molecule_name)

        # Set residue name for all atoms in the molecule in OpenBabel
        for atom in ob.OBMolAtomIter(obmol):
            res = atom.GetResidue()
            res.SetName(resname)

        if not obConversion.WriteFile(obmol, output_file):
            raise IOError("Failed to write the PDB file.")

        print(f"PDB file successfully created: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

# 7. Convert SMILES to XYZ
def smiles_to_xyz(smiles, filename, num_confs=1):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无效的 SMILES 字符串。")

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=42)
    if not result:
        raise ValueError("无法生成3D构象。")
    AllChem.UFFOptimizeMolecule(mol, confId=0)
    conf = mol.GetConformer(0)
    atoms = mol.GetAtoms()
    coords = [conf.GetAtomPosition(atom.GetIdx()) for atom in atoms]
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"SMILES: {smiles}\n")
        for atom, coord in zip(atoms, coords):
            f.write(f"{atom.GetSymbol()} {coord.x:.4f} {coord.y:.4f} {coord.z:.4f}\n")





