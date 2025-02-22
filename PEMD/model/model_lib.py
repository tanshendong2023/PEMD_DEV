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

from math import pi
from rdkit import Chem
from openbabel import pybel
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem import Descriptors
from collections import defaultdict
from openbabel import openbabel as ob
from networkx.algorithms import isomorphism
from scipy.spatial.transform import Rotation as R

import numpy as np
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType
from scipy.spatial.transform import Rotation as R
from rdkit.Chem import rdMolTransforms

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局默认方向及判断阈值
DEFAULT_DIRECTION = np.array([1.0, 0.0, 0.0])
MIN_DIRECTION_NORM = 0.1


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


# Processes a polymer’s SMILES string with dummy atoms to set up connectivity and identify the connecting atoms.
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




def rotate_mol_around_axis(mol, axis, anchor, angle_rad):
    """
    将整个分子绕给定单位向量 axis，以 anchor 为中心旋转 angle_rad 弧度。
    """
    conf = mol.GetConformer()
    rot = R.from_rotvec(axis * angle_rad)
    for atom_idx in range(mol.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(atom_idx))
        pos_shifted = pos - anchor
        pos_rot = rot.apply(pos_shifted)
        conf.SetAtomPosition(atom_idx, pos_rot + anchor)

def rotate_vector_to_align(a, b):
    """
    返回一个旋转对象，使得向量 a 旋转后与向量 b 对齐。
    """
    a_norm = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else DEFAULT_DIRECTION
    b_norm = b / np.linalg.norm(b) if np.linalg.norm(b) > 1e-6 else DEFAULT_DIRECTION
    cross_prod = np.cross(a_norm, b_norm)
    norm_cross = np.linalg.norm(cross_prod)
    if norm_cross < 1e-6:
        arbitrary = np.array([1, 0, 0])
        if np.allclose(a_norm, arbitrary) or np.allclose(a_norm, -arbitrary):
            arbitrary = np.array([0, 1, 0])
        rotation_axis = np.cross(a_norm, arbitrary)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        dot_prod = np.dot(a_norm, b_norm)
        angle_rad = np.pi if dot_prod < 0 else 0
    else:
        rotation_axis = cross_prod / norm_cross
        dot_prod = np.dot(a_norm, b_norm)
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        angle_rad = np.arccos(dot_prod)
    return R.from_rotvec(rotation_axis * angle_rad)

def _get_ideal_tetrahedral_vectors():
    """
    返回理想正四面体状态下4个顶点的归一化参考向量。
    """
    vs = [
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ]
    return [np.array(v) / np.linalg.norm(v) for v in vs]

def _kabsch_rotation(P, Q):
    """
    利用 Kabsch 算法计算最佳旋转矩阵，使得 P 旋转后与 Q 尽可能匹配。
    P, Q 均为 (n, 3) 数组。
    """
    C = np.dot(P.T, Q)
    V, S, Wt = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
    if d:
        V[:, -1] = -V[:, -1]
    return np.dot(V, Wt)

def place_h_in_tetrahedral(mol, atom_idx, h_indices):
    """
    重新定位中心原子 atom_idx 上的氢原子，使局部几何尽量符合预期构型。
    针对 NH2（氮原子、1 个重邻居、2 个氢）单独处理，
    对于其他情况仍采用正四面体方法。
    """
    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(atom_idx))
    center_atom = mol.GetAtomWithIdx(atom_idx)
    heavy_neighbors = [nbr.GetIdx() for nbr in center_atom.GetNeighbors() if nbr.GetAtomicNum() != 1]

    # 检测是否为 NH2 型：氮原子、1 个重邻居、传入2个氢
    if center_atom.GetAtomicNum() == 7 and len(heavy_neighbors) == 1 and len(h_indices) == 2:
        hv_idx = heavy_neighbors[0]
        hv_pos = np.array(conf.GetAtomPosition(hv_idx))
        v = hv_pos - center_pos
        if np.linalg.norm(v) < 1e-6:
            logger.warning("Atom %s: heavy neighbor vector too small; using default.", atom_idx)
            v = np.array([0, 0, 1])
        else:
            v = v / np.linalg.norm(v)

        # 获取理想正四面体方向
        tet_dirs = _get_ideal_tetrahedral_vectors()  # 返回4个单位向量

        # 1. 找出与 v 最一致的方向（应对应于重邻居方向）
        dots = [np.dot(d, v) for d in tet_dirs]
        idx_heavy = np.argmax(dots)

        # 2. 在剩下的3个方向中，找出与 -v 最一致的方向（对应孤对，暂不放氢）
        remaining = [(i, d) for i, d in enumerate(tet_dirs) if i != idx_heavy]
        dots_neg = [np.dot(d, -v) for i, d in remaining]
        idx_lonepair = remaining[np.argmax(dots_neg)][0]

        # 3. 剩下的两个方向用来放置氢原子
        h_dirs = [d for i, d in enumerate(tet_dirs) if i not in (idx_heavy, idx_lonepair)]
        if len(h_dirs) != 2:
            logger.error("Internal error: expected 2 hydrogen directions, got %s", len(h_dirs))
            return

        CH_BOND = 1.09  # 典型 C–H 键长
        # 首先为两个氢原子设定新的位置
        new_pos_1 = center_pos + CH_BOND * h_dirs[0]
        new_pos_2 = center_pos + CH_BOND * h_dirs[1]

        # 检查氢原子之间的距离，避免重叠
        for i, h_idx in enumerate(h_indices):
            if i == 0:
                new_pos = new_pos_1
            else:
                new_pos = new_pos_2
            for other_h_idx in h_indices:
                if other_h_idx != h_idx:
                    other_h_pos = np.array(conf.GetAtomPosition(other_h_idx))
                    if np.linalg.norm(new_pos - other_h_pos) < 0.8:  # 检查阈值，防止重叠
                        logger.warning(f"Hydrogen atoms {h_idx} and {other_h_idx} overlap! Adjusting.")
                        new_pos += np.random.uniform(0.1, 0.2, size=3)  # 轻微调整位置

        # 更新氢原子位置
        conf.SetAtomPosition(h_indices[0], new_pos_1)
        conf.SetAtomPosition(h_indices[1], new_pos_2)
        return

def get_vector(mol, index):
    """
    对于指定原子，返回其位置及其与所有邻接原子连线方向的平均单位向量。
    若无邻居或平均向量过小，则返回默认方向。
    """
    conf = mol.GetConformer()
    pos = np.array(conf.GetAtomPosition(index))
    atom = mol.GetAtomWithIdx(index)
    neighbors = atom.GetNeighbors()
    if not neighbors:
        return pos, DEFAULT_DIRECTION
    vecs = []
    for nbr in neighbors:
        nbr_pos = np.array(conf.GetAtomPosition(nbr.GetIdx()))
        v = pos - nbr_pos
        if np.linalg.norm(v) > 1e-6:
            vecs.append(v / np.linalg.norm(v))
    if not vecs:
        return pos, DEFAULT_DIRECTION
    avg = np.mean(vecs, axis=0)
    norm_avg = np.linalg.norm(avg)
    if norm_avg < MIN_DIRECTION_NORM:
        logger.warning("Atom %s: Computed local direction norm too small (%.3f); using default.", index, norm_avg)
        return pos, DEFAULT_DIRECTION
    return pos, avg / norm_avg

def align_monomer_unit(monomer, connection_atom_idx, target_position, target_direction):
    conf = monomer.GetConformer()
    B = np.array(conf.GetAtomPosition(connection_atom_idx))
    if np.linalg.norm(target_direction) < MIN_DIRECTION_NORM:
        logger.warning("Target direction is too small; using default direction.")
        target_direction = DEFAULT_DIRECTION
    _, local_dir = get_vector(monomer, connection_atom_idx)
    if np.linalg.norm(local_dir) < MIN_DIRECTION_NORM:
        logger.warning("Local direction of atom %s is too small; using default.", connection_atom_idx)
        local_dir = DEFAULT_DIRECTION
    rot_obj = rotate_vector_to_align(local_dir, -target_direction)
    for i in range(monomer.GetNumAtoms()):
        pos_i = np.array(conf.GetAtomPosition(i))
        new_pos = B + rot_obj.apply(pos_i - B)
        conf.SetAtomPosition(i, new_pos)
    B_rot = np.array(conf.GetAtomPosition(connection_atom_idx))
    translation = target_position - B_rot
    for i in range(monomer.GetNumAtoms()):
        pos_i = np.array(conf.GetAtomPosition(i))
        conf.SetAtomPosition(i, pos_i + translation)
    return monomer

def local_optimize(mol, maxIters=50, num_retries=10, perturbation=0.01):

    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    for attempt in range(num_retries):
        try:
            mol.UpdatePropertyCache(strict=False)
            _ = Chem.GetSymmSSSR(mol)

            # 优化前检查是否有重叠原子
            if has_overlapping_atoms(mol):
                logger.warning("Molecule has overlapping atoms, adjusting atomic positions.")
                conf = mol.GetConformer()
                for i in range(mol.GetNumAtoms()):
                    pos = np.array(conf.GetAtomPosition(i))
                    if mol.GetAtomWithIdx(i).GetAtomicNum() == 1:  # 仅调整氢原子
                        conf.SetAtomPosition(i, pos + np.random.uniform(0.01, 1.8, size=3))

            status = AllChem.MMFFOptimizeMolecule(mol, maxIters=maxIters)
            if status < 0:
                raise RuntimeError("MMFF optimization returned status %s" % status)
            return mol  # 优化成功，返回分子

        except Exception as e:
            logger.warning(f"Local optimization attempt {attempt + 1} failed: {e}")
            conf = mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                pos = np.array(conf.GetAtomPosition(i))
                delta = np.random.uniform(-perturbation, perturbation, size=3)
                conf.SetAtomPosition(i, pos + delta)
    logger.error("Local optimization failed after {num_retries} attempts.")
    return mol

def rotate_substructure_around_axis(mol, atom_indices, axis, anchor, angle_rad):
    """
    对分子中给定 atom_indices 列表中的原子，
    以 anchor 为中心绕单位向量 axis 旋转 angle_rad 弧度。
    """
    conf = mol.GetConformer()
    rot = R.from_rotvec(axis * angle_rad)
    for idx in atom_indices:
        pos = np.array(conf.GetAtomPosition(idx))
        pos_shifted = pos - anchor
        pos_rot = rot.apply(pos_shifted)
        conf.SetAtomPosition(idx, pos_rot + anchor)

def gen_3D_info(monomer_mol, length, polymer_conn_idx, monomer_conn_idx, bond_length):
    """
    利用刚性对齐方法将多个单体按顺序连接构建 3D 聚合物结构，增加精细的距离检查。
    """
    connecting_mol = Chem.RWMol(monomer_mol)
    num_atoms = monomer_mol.GetNumAtoms()

    for i in range(1, length):
        conf_poly = connecting_mol.GetConformer()

        tail_idx = polymer_conn_idx + (i - 1) * num_atoms
        tail_pos = np.array(conf_poly.GetAtomPosition(tail_idx))
        ideal_direction = compute_monomer_direction(monomer_mol, monomer_conn_idx, polymer_conn_idx)

        # 增加0.3 Å的额外距离以缓解关键基团过近的问题
        target_pos = tail_pos + (bond_length + 0.1) * ideal_direction

        new_unit = Chem.Mol(monomer_mol)
        new_unit = align_monomer_unit(new_unit, monomer_conn_idx, target_pos, ideal_direction)

        # 对新单元沿连接键轴进行额外旋转，中心设为 target_pos，旋转角度为 extra_angle
        extra_angle = 0.5
        atom_indices_to_rotate = [j for j in range(new_unit.GetNumAtoms()) if j != monomer_conn_idx]
        rotate_substructure_around_axis(new_unit, atom_indices_to_rotate, ideal_direction, target_pos, extra_angle)

        combined = Chem.CombineMols(connecting_mol, new_unit)
        editable = Chem.EditableMol(combined)
        new_unit_conn_idx = monomer_conn_idx + i * num_atoms
        editable.AddBond(tail_idx, new_unit_conn_idx, order=BondType.SINGLE)

        combined_mol = editable.GetMol()
        combined_mol = Chem.RWMol(combined_mol)
        h_indices = [nbr.GetIdx() for nbr in combined_mol.GetAtomWithIdx(new_unit_conn_idx).GetNeighbors()
                     if nbr.GetAtomicNum() == 1]
        place_h_in_tetrahedral(combined_mol, new_unit_conn_idx, h_indices)

        # 进行局部能量优化，帮助调整连接区域几何
        combined_mol = local_optimize(combined_mol, maxIters=100)

        # 连接后的分子重叠检查
        if has_overlapping_atoms(combined_mol):
            logger.warning(f"After unit {i + 1}, overlapping atoms detected.")

        connecting_mol = Chem.RWMol(combined_mol)

    final_polymer = connecting_mol.GetMol()
    new_tail_idx = polymer_conn_idx + (length - 1) * num_atoms
    return final_polymer, polymer_conn_idx, new_tail_idx

def gen_3D_nocap(dum1, dum2, atom1, atom2, smiles_each, length):

    input_mol = Chem.MolFromSmiles(smiles_each)
    if input_mol is None:
        raise ValueError("Invalid SMILES string.")
    rw_mol = Chem.RWMol(input_mol)
    for atom in rw_mol.GetAtoms():
        if atom.GetSymbol() == '*':
            atom.SetAtomicNum(53)
    mol_with_h = Chem.AddHs(rw_mol)
    rw_mol = Chem.RWMol(mol_with_h)

    params = AllChem.ETKDGv3()
    params.randomSeed = -1
    if AllChem.EmbedMolecule(rw_mol, params) != 0:
        logger.warning("3D embedding failed.")
    AllChem.MMFFOptimizeMolecule(rw_mol)

    indices_to_remove = sorted([dum1, dum2], reverse=True)
    for idx in indices_to_remove:
        rw_mol.RemoveAtom(idx)
    monomer_mol = Chem.Mol(rw_mol)
    if atom1 > atom2:
        atom1, atom2 = atom2, atom1

    def adjust_index(idx, removals):
        return idx - sum(1 for r in removals if r < idx)

    head_atom = adjust_index(atom1, indices_to_remove)
    tail_atom = adjust_index(atom2, indices_to_remove)
    if length == 1:
        return monomer_mol, monomer_mol, head_atom, tail_atom
    polymer, poly_conn_idx, new_tail_idx = gen_3D_info(
        monomer_mol, length,
        polymer_conn_idx=tail_atom,
        monomer_conn_idx=head_atom,
        bond_length=1.5
    )

    return polymer, monomer_mol, head_atom, new_tail_idx

def compute_monomer_direction(monomer, head_atom, tail_atom):
    """
    计算单体中两个连接原子之间的方向向量。
    对于复杂环境（例如包含羰基和醚基的单体），优先采用 tail_atom 的非氢重原子邻居作为参考，
    以获得更合理的连接方向。
    """
    from rdkit import Chem
    conf = monomer.GetConformer()
    tail_atom_obj = monomer.GetAtomWithIdx(tail_atom)
    tail_pos = np.array(conf.GetAtomPosition(tail_atom))
    head_pos = np.array(conf.GetAtomPosition(head_atom))

    # 获取 tail_atom 的所有非氢邻居（排除 head_atom）
    heavy_neighbors = [nbr for nbr in tail_atom_obj.GetNeighbors()
                       if nbr.GetAtomicNum() != 1 and nbr.GetIdx() != head_atom]

    if heavy_neighbors:
        # 选择第一个重原子作为参考
        ref_nbr = heavy_neighbors[0]
        ref_pos = np.array(conf.GetAtomPosition(ref_nbr.GetIdx()))
        vec = ref_pos - tail_pos
    else:
        vec = head_pos - tail_pos

    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        logger.warning("Monomer connection vector norm too small; using default direction.")
        return DEFAULT_DIRECTION
    return vec / norm

def gen_3D_withcap(mol, start_atom, end_atom):

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

            terminal_pos, v_norm = get_vector(capped_mol, terminal_idx)

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
                trans = True
            )

    # 检查原子间距离是否合理
    overlap = check_molecule_structure(capped_mol, energy_threshold=50.0)
    if not overlap:
        return capped_mol
    else:
        print(f"Generated the pdb file failed !!!")

def connect_mols(mol1, mol2, tail_index, head_index, bond_length, trans, do_tetrahedral=True):
    # 获取连接原子（尾部原子和头部原子）的位置和方向
    pos1_tail, v1_norm = get_vector(mol1, tail_index)  # 获取mol1尾部原子的方向
    pos2_head, v2_norm = get_vector(mol2, head_index)  # 获取mol2头部原子的方向

    # 计算旋转，使v1_norm对齐到-v2_norm
    rot = rotate_vector_to_align(v1_norm, -v2_norm)

    # 获取mol2中所有原子的坐标
    conf_2 = mol2.GetConformer()
    all_coords = []
    for atom_idx in range(mol2.GetNumAtoms()):
        p = conf_2.GetAtomPosition(atom_idx)
        all_coords.append(np.array([p.x, p.y, p.z]))

    # 旋转mol2的原子
    center = pos2_head  # 以头部原子作为旋转的中心
    rotated_coords = []
    for pos in all_coords:
        translated = pos - center
        rotated = rot.apply(translated)  # 旋转原子
        new_pos = rotated + center
        rotated_coords.append(new_pos)

    # 更新mol2的Conformer
    rotated_conf = Chem.Conformer(mol2.GetNumAtoms())
    for idx, pos in enumerate(rotated_coords):
        rotated_conf.SetAtomPosition(idx, pos)
    rotated_mol = Chem.Mol(mol2)
    rotated_mol.RemoveAllConformers()
    rotated_mol.AddConformer(rotated_conf, assignId=True)

    # 计算并平移mol2，使得它与mol1的尾端原子之间的距离为bond_length
    pos2_adjusted = pos1_tail + v1_norm * bond_length  # 目标位置
    translation = pos2_adjusted - pos2_head  # 计算平移量

    new_conf = rotated_mol.GetConformer()
    for atom_idx in range(rotated_mol.GetNumAtoms()):
        pos = np.array([
            new_conf.GetAtomPosition(atom_idx).x,
            new_conf.GetAtomPosition(atom_idx).y,
            new_conf.GetAtomPosition(atom_idx).z
        ])
        pos_new = pos + translation  # 平移后的新位置
        new_conf.SetAtomPosition(atom_idx, pos_new)

    # 如果需要，进一步旋转mol2
    if trans:
        rotate_mol_around_axis(rotated_mol, axis=v1_norm, anchor=pos1_tail, angle_rad=np.pi / 2)

    # 合并mol1和mol2
    combo = Chem.CombineMols(mol1, rotated_mol)
    editable_combo = Chem.EditableMol(combo)
    editable_combo.AddBond(
        tail_index,
        head_index + mol1.GetNumAtoms(),
        order=Chem.rdchem.BondType.SINGLE
    )
    mol_connected = editable_combo.GetMol()

    # 进行重叠原子检查
    if has_overlapping_atoms(mol_connected):
        logger.warning("After connecting molecules, overlapping atoms detected.")
        # 对分子进行微调，避免重叠
        mol_connected = local_optimize(mol_connected, maxIters=100)

    # 如果需要，对新连接的头部原子做氢的四面体化处理
    if do_tetrahedral:
        # 获取新连接的头部原子的氢原子索引
        new_head_idx = head_index + mol1.GetNumAtoms()

        rw_connected = Chem.RWMol(mol_connected)
        # 获取头部原子的氢原子
        h_indices = [nbr.GetIdx() for nbr in rw_connected.GetAtomWithIdx(new_head_idx).GetNeighbors()
                     if rw_connected.GetAtomWithIdx(nbr.GetIdx()).GetAtomicNum() == 1]
        if h_indices:
            place_h_in_tetrahedral(rw_connected, new_head_idx, h_indices)  # 对氢原子进行四面体化
            mol_connected = rw_connected.GetMol()

    return mol_connected

def check_molecule_structure(mol, energy_threshold=50.0):

    # 1. 尝试对分子进行基本的 Sanitize 检查
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        # 如果在 Sanitize 过程中出现错误，分子不合理
        return False

    # 2. 检查是否存在 3D 构象
    if mol.GetNumConformers() == 0:
        return False

    # 3. 使用力场进行简要优化并检查能量
    #    这里使用 UFF 作为示例，也可以使用 MMFF:
    #    AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s")
    try:
        status = AllChem.UFFOptimizeMolecule(mol)  # 返回 0 表示正常收敛
        if status != 0:
            # 如果优化没有收敛，可能是结构不合理或存在其他问题
            return False

        # 获取最终能量
        ff = AllChem.UFFGetMoleculeForceField(mol)
        final_energy = ff.CalcEnergy()

        if final_energy > energy_threshold:
            # 如果能量过高，可能存在严重扭曲或应变
            return False
    except Exception:
        # 如果力场优化过程出现任何异常，也视为不合理
        return False

    # 如果以上检查都通过，视为结构“合理”
    return True


def has_overlapping_atoms(mol, connected_distance=1.0, disconnected_distance=1.55):
    """
    检查分子中是否存在原子重叠：
      - 如果两个原子通过化学键相连，则允许的最小距离为 connected_distance
      - 如果不相连，则默认使用 disconnected_distance，
        如果任一原子为氧或卤素（F, Cl, Br, I）或两个原子均为碳，
        则要求最小距离为 1.6 Å（你也可以修改为 2.1 Å，根据需要）。
    当检测到原子对距离过近时，会输出相关信息，包括原子的名称。
    """
    # 获取分子的拓扑结构（邻接矩阵）
    mol_graph = Chem.rdmolops.GetAdjacencyMatrix(mol)

    # 获取所有原子的坐标
    conf = mol.GetConformer()
    positions = conf.GetPositions()

    # 构建一个无向图来存储化学键连接关系
    bond_graph = nx.Graph()
    for i in range(mol.GetNumAtoms()):
        bond_graph.add_node(i, position=positions[i])
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        bond_graph.add_edge(atom1, atom2)

    # 创建一个图用于存储重叠关系
    G = nx.Graph()
    num_atoms = mol.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(positions[i] - positions[j])
            # 获取原子对允许的最小距离
            actual_min_distance = get_min_distance(mol, i, j, bond_graph, connected_distance, disconnected_distance)
            # 如果实际距离小于允许距离，则认为这对原子重叠
            if distance < actual_min_distance:
                atom1_name = mol.GetAtomWithIdx(i).GetSymbol()
                atom2_name = mol.GetAtomWithIdx(j).GetSymbol()
                print(f"Overlapping detected: Atom {atom1_name} ({i}) and Atom {atom2_name} ({j}) are too close "
                      f"(distance: {distance:.2f} Å, allowed minimum: {actual_min_distance:.2f} Å)")
                G.add_edge(i, j, weight=distance)

    # 如果图中有边，则表示存在重叠原子
    return len(G.edges) > 0


def get_min_distance(mol, atom1, atom2, bond_graph, connected_distance=1.0, disconnected_distance=1.55):
    """
    根据原子对的连接情况及原子类型返回最小允许距离：
      - 如果 atom1 和 atom2 之间存在化学键，则返回 connected_distance
      - 如果不相连，则：
          * 如果任一原子为氧、卤素（F, Cl, Br, I）、氢原子，
            或两个原子均为碳，则返回 1.6 Å （你可以根据需要调整该数值，例如改为 2.1 Å）
          * 如果有氧、卤素与氢原子之间的连接，返回 1.8 Å
          * 否则返回 disconnected_distance。
    """
    if bond_graph.has_edge(atom1, atom2):
        return connected_distance
    else:
        symbol1 = mol.GetAtomWithIdx(atom1).GetSymbol()
        symbol2 = mol.GetAtomWithIdx(atom2).GetSymbol()

        # 判断条件：氧、卤素和氢原子之间的连接返回 1.8 Å
        if (symbol1 in ['O', 'F', 'Cl', 'Br', 'I'] and symbol2 in ['H']) or \
                (symbol1 in ['H'] and symbol2 in ['O', 'F', 'Cl', 'Br', 'I']) or \
                (symbol1 == 'N' and symbol2 == 'O') or (symbol1 == 'O' and symbol2 == 'N'):
            return 1.8
        # 判断条件：氧、卤素、氮和碳之间的连接返回 1.6 Å
        elif (symbol1 in ['O', 'F', 'Cl', 'Br', 'I'] and symbol2 in ['O', 'F', 'Cl', 'Br', 'I']) or \
                (symbol1 == 'C' and symbol2 == 'O') or (symbol1 == 'O' and symbol2 == 'C'):
            return 1.6
        else:
            return disconnected_distance


def set_connection_dihedral(mol, tail_index, head_index_new, fixed_angle=180.0):
    """
    固定连接键处的二面角以刚性化聚合物链片段。

    参数：
      - mol: 合并后的分子（RDKit Mol 对象）。
      - tail_index: 来自第一个分子（mol1）的连接原子索引。
      - head_index_new: 来自第二个分子（mol2）经过合并后对应的连接原子索引。
      - fixed_angle: 希望设置的二面角（单位：度），例如 180 度。

    实现思路：
      1. 对 tail_index，选择其除 head_index_new 外的一个邻居作为 dihedral 第一个原子。
      2. 对 head_index_new，选择其除 tail_index 外的一个邻居作为 dihedral 第四个原子。
      3. 利用 rdMolTransforms.SetDihedralDeg 设置这 4 个原子的二面角。
    """
    conf = mol.GetConformer()
    tail_atom = mol.GetAtomWithIdx(tail_index)
    head_atom = mol.GetAtomWithIdx(head_index_new)

    # 找出 tail_index 的其他邻居（排除已连接的 head_index_new）
    tail_neighbors = [nbr.GetIdx() for nbr in tail_atom.GetNeighbors() if nbr.GetIdx() != head_index_new]
    if len(tail_neighbors) == 0:
        print(f"Warning: Tail atom {tail_index} 无其他邻居，无法定义二面角！")
        return
    atom_a = tail_neighbors[0]

    # 找出 head_index_new 的其他邻居（排除 tail_index）
    head_neighbors = [nbr.GetIdx() for nbr in head_atom.GetNeighbors() if nbr.GetIdx() != tail_index]
    if len(head_neighbors) == 0:
        print(f"Warning: Head atom {head_index_new} 无其他邻居，无法定义二面角！")
        return
    atom_d = head_neighbors[0]

    rdMolTransforms.SetDihedralDeg(conf, atom_a, tail_index, head_index_new, atom_d, fixed_angle)
    print(f"已将原子 ({atom_a}, {tail_index}, {head_index_new}, {atom_d}) 的二面角设置为 {fixed_angle}°.")

# (symbol1 == 'C' and symbol2 == 'C') or
#            (symbol1 == 'O' and symbol2 == 'H') or (symbol1 == 'H' and symbol2 == 'O'):
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





