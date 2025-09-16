"""
PEMD code library.

Developed by: Tan Shendong
Date: 2025.05.23
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import PEMD.io as io
import PEMD.constants as const

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from PEMD.model import model_lib
from rdkit.Chem import Descriptors
from rdkit.Geometry import Point3D
from collections import defaultdict
from openbabel import openbabel as ob
from rdkit.Chem.rdchem import BondType
from scipy.spatial.transform import Rotation as R


lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenBabel setup
obConversion = ob.OBConversion()
ff = ob.OBForceField.FindForceField('UFF')
mol = ob.OBMol()
np.set_printoptions(precision=20)

def gen_sequence_copolymer_3D(name,
                              smiles_A,
                              smiles_B,
                              sequence,
                              bond_length=1.5,
                              left_cap_smiles=None,
                              right_cap_smiles=None):
    """
    通用序列构建：sequence 是一个列表，如 ['A','B','B','A',…]
    """
    # 1. 预先初始化 A、B 单体的信息
    dumA1, dumA2, atomA1, atomA2 = Init_info(name, smiles_A)
    dumB1, dumB2, atomB1, atomB2 = Init_info(name, smiles_B)

    first_unit = sequence[0]
    if first_unit == 'A':
        dum1, dum2, atom1, atom2, smiles_mid = dumA1, dumA2, atomA1, atomA2, smiles_A
    else:
        dum1, dum2, atom1, atom2, smiles_mid = dumB1, dumB2, atomB1, atomB2, smiles_B

    mol_1, h_1, t_1 = prepare_monomer_nocap(smiles_mid, dum1, dum2, atom1, atom2)

    connecting_mol = Chem.RWMol(mol_1)

    # 3. 依次添加后续单元
    tail_idx = t_1
    num_atom = connecting_mol.GetNumAtoms()
    for unit in sequence[1:]:
        if unit == 'A':
            dum1, dum2, atom1, atom2, smiles_mid = dumA1, dumA2, atomA1, atomA2, smiles_A
        else:
            dum1, dum2, atom1, atom2, smiles_mid = dumB1, dumB2, atomB1, atomB2, smiles_B

        mon, h, t = prepare_monomer_nocap(smiles_mid, dum1, dum2, atom1, atom2)

        conf_poly = connecting_mol.GetConformer()
        tail_pos = np.array(conf_poly.GetAtomPosition(tail_idx))

        _, ideal_direction = get_vector(connecting_mol, tail_idx)

        # 增加0.1 Å的额外距离以缓解关键基团过近的问题
        target_pos = tail_pos + (bond_length + 0.1) * ideal_direction

        new_unit = Chem.Mol(mon)
        new_unit = align_monomer_unit(new_unit, h, target_pos, ideal_direction)

        # 对新单元沿连接键轴进行额外旋转，中心设为 target_pos，旋转角度为 extra_angle
        if not check_3d_structure(connecting_mol):
            extra_angle = 0.20
            atom_indices_to_rotate = [j for j in range(new_unit.GetNumAtoms()) if j != h_1]
            rotate_substructure_around_axis(new_unit, atom_indices_to_rotate, ideal_direction, target_pos,
                                            extra_angle)

        combined = Chem.CombineMols(connecting_mol, new_unit)
        editable = Chem.EditableMol(combined)
        head_idx = num_atom + h
        editable.AddBond(tail_idx, head_idx, order=BondType.SINGLE)

        combined_mol = editable.GetMol()
        combined_mol = Chem.RWMol(combined_mol)
        h_indices = [nbr.GetIdx() for nbr in combined_mol.GetAtomWithIdx(head_idx).GetNeighbors()
                     if nbr.GetAtomicNum() == 1]
        place_h_in_tetrahedral(combined_mol, head_idx, h_indices)

        # 进行局部能量优化，帮助调整连接区域几何
        if not check_3d_structure(combined_mol):
            combined_mol = local_optimize(combined_mol, maxIters=150)
        connecting_mol = Chem.RWMol(combined_mol)

        tail_idx = num_atom + t
        num_atom = num_atom + new_unit.GetNumAtoms()

    length = len(sequence)
    final_poly = gen_3D_withcap(
        connecting_mol,
        h_1,
        tail_idx,
        length,
        left_cap_smiles=left_cap_smiles,
        right_cap_smiles=right_cap_smiles,
    )

    return final_poly


# Processes a polymer’s SMILES string with dummy atoms to set up connectivity and identify the connecting atoms.
def Init_info(name, smiles_mid):
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
    xyz_filename = io.smile_toxyz(
        name,
        smiles_each,       # Replace '*' with dummy atom
    )

    # Collect valency and connecting information for each atom according to XYZ coordinates
    neigh_atoms_info = connec_info(xyz_filename)

    # Find connecting atoms associated with dummy atoms.
    # Dum1 and dum2 are connected to atom1 and atom2, respectively.
    atom1 = neigh_atoms_info['NeiAtom'][dum1].copy()[0]
    atom2 = neigh_atoms_info['NeiAtom'][dum2].copy()[0]

    Path(xyz_filename).unlink(missing_ok=True)  # Clean up the temporary XYZ file

    return dum1, dum2, atom1, atom2,


# Get index of dummy atoms and bond type associated with it
def FetchDum(smiles):
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


def prepare_monomer_nocap(smiles_mid: str,
                          dum1: int,
                          dum2: int,
                          atom1: int,
                          atom2: int) -> tuple[Chem.Mol, int, int]:
    """
    将带 dummy 原子的 SMILES:
      - 插入 3D 坐标并优化
      - 添加氢，Embed & Optimize
      - 移除 dummy 原子
    返回:
      - monomer: 去除 dummy 后的 RDKit Mol
      - head_idx: 删除后对应 atom1 的索引
      - tail_idx: 删除后对应 atom2 的索引
    """
    # 1. 生成 RDKit 分子，替换 '*' 为原子
    mol = Chem.MolFromSmiles(smiles_mid)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles_mid}")
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetSymbol() == '*':
            atom.SetAtomicNum(53)  # Iodine 代替 dummy
    # 2. 添加氢并 embed
    rw = Chem.RWMol(Chem.AddHs(rw))
    params = AllChem.ETKDGv3()
    params.randomSeed = -1
    if AllChem.EmbedMolecule(rw, params) != 0:
        logger.warning("3D embedding failed for monomer.")
    AllChem.MMFFOptimizeMolecule(rw)

    # 3. 移除 dummy 原子
    to_remove = sorted([dum1, dum2], reverse=True)
    for idx in to_remove:
        rw.RemoveAtom(idx)
    monomer = rw.GetMol()

    # 4. 计算新的 head/tail 索引
    def adjust(i: int) -> int:
        return i - sum(1 for r in to_remove if r < i)

    new_head = adjust(atom1)
    new_tail = adjust(atom2)
    if new_head > new_tail:
        new_head, new_tail = new_tail, new_head

    return monomer, new_head, new_tail

def prepare_cap_monomer(smiles_cap: str) -> tuple[Chem.Mol, int, np.ndarray]:
    """Prepare a capping fragment defined by a SMILES string containing a single dummy atom."""
    mol = Chem.MolFromSmiles(smiles_cap)
    if mol is None:
        raise ValueError(f"Invalid cap SMILES: {smiles_cap}")

    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_indices) != 1:
        raise ValueError("Cap SMILES must contain exactly one dummy atom '*' or '[*]'.")

    dummy_idx = dummy_indices[0]
    dummy_atom = mol.GetAtomWithIdx(dummy_idx)
    neighbors = list(dummy_atom.GetNeighbors())
    if len(neighbors) != 1:
        raise ValueError("Cap dummy atom must be connected to exactly one atom.")

    connection_idx = neighbors[0].GetIdx()

    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(dummy_idx).SetAtomicNum(53)  # Use iodine as a placeholder heavy atom

    rw = Chem.RWMol(Chem.AddHs(rw))
    params = AllChem.ETKDGv3()
    params.randomSeed = -1
    if AllChem.EmbedMolecule(rw, params) != 0:
        logger.warning("3D embedding failed for cap %s.", smiles_cap)
    try:
        AllChem.MMFFOptimizeMolecule(rw)
    except Exception as exc:  # pragma: no cover - RDKit errors are data dependent
        logger.warning("MMFF optimization failed for cap %s: %s", smiles_cap, exc)

    conf = rw.GetConformer()
    attachment_vec = np.array(conf.GetAtomPosition(dummy_idx)) - np.array(conf.GetAtomPosition(connection_idx))
    if np.linalg.norm(attachment_vec) < const.MIN_DIRECTION_NORM:
        logger.warning("Attachment direction too small for cap %s; using default.", smiles_cap)
        attachment_vec = const.DEFAULT_DIRECTION
    else:
        attachment_vec = attachment_vec / np.linalg.norm(attachment_vec)

    rw.RemoveAtom(dummy_idx)
    if connection_idx > dummy_idx:
        connection_idx -= 1

    cap_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(cap_mol)
    except Exception as exc:  # pragma: no cover - depends on specific SMILES
        logger.warning("Sanitization failed for cap %s: %s", smiles_cap, exc)

    return cap_mol, connection_idx, attachment_vec


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
        return pos, const.DEFAULT_DIRECTION
    vecs = []
    for nbr in neighbors:
        nbr_pos = np.array(conf.GetAtomPosition(nbr.GetIdx()))
        v = pos - nbr_pos
        if np.linalg.norm(v) > 1e-6:
            vecs.append(v / np.linalg.norm(v))
    if not vecs:
        return pos, const.DEFAULT_DIRECTION
    avg = np.mean(vecs, axis=0)
    norm_avg = np.linalg.norm(avg)
    if norm_avg < const.MIN_DIRECTION_NORM:
        logger.warning("Atom %s: Computed local direction norm too small (%.3f); using default.", index, norm_avg)
        return pos, const.DEFAULT_DIRECTION
    return pos, avg / norm_avg


def align_monomer_unit(monomer,
                       connection_atom_idx,
                       target_position,
                       target_direction,
                       local_reference_direction=None):

    conf = monomer.GetConformer()
    B = np.array(conf.GetAtomPosition(connection_atom_idx))
    if np.linalg.norm(target_direction) < const.MIN_DIRECTION_NORM:
        logger.warning("Target direction is too small; using default direction.")
        target_direction = const.DEFAULT_DIRECTION
    if local_reference_direction is None:
        _, local_dir = get_vector(monomer, connection_atom_idx)
    else:
        local_dir = np.array(local_reference_direction, dtype=float)
    if np.linalg.norm(local_dir) < const.MIN_DIRECTION_NORM:
        logger.warning("Local direction of atom %s is too small; using default.", connection_atom_idx)
        local_dir = const.DEFAULT_DIRECTION
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

def local_optimize(mol, maxIters=100, num_retries=1000, perturbation=0.01):

    for attempt in range(num_retries):
        try:
            mol.UpdatePropertyCache(strict=False)
            _ = Chem.GetSymmSSSR(mol)

            # 优化前检查是否有重叠原子
            if not check_3d_structure(mol):
                logger.warning("\nMolecule has overlapping atoms, adjusting atomic positions.")
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
    logger.error(f"Local optimization failed after {num_retries} attempts.")
    return mol

def rotate_vector_to_align(a, b):
    """
    返回一个旋转对象，使得向量 a 旋转后与向量 b 对齐。
    """
    a_norm = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else const.DEFAULT_DIRECTION
    b_norm = b / np.linalg.norm(b) if np.linalg.norm(b) > 1e-6 else const.DEFAULT_DIRECTION
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


def estimate_bond_length(atom_num1: int, atom_num2: int, fallback: float = 1.5) -> float:
    """Estimate a bond length based on covalent radii with a safe fallback."""
    pt = Chem.GetPeriodicTable()
    try:
        length = pt.GetRcovalent(atom_num1) + pt.GetRcovalent(atom_num2)
    except Exception:
        return fallback
    if not np.isfinite(length) or length <= 0:
        return fallback
    return float(length)


def attach_fragment(base_mol: Chem.Mol,
                    fragment: Chem.Mol,
                    terminal_idx: int,
                    fragment_connection_idx: int) -> Chem.Mol:
    n_base = base_mol.GetNumAtoms()
    combo = Chem.CombineMols(base_mol, fragment)
    ed = Chem.EditableMol(combo)
    new_idx = fragment_connection_idx + n_base
    ed.AddBond(terminal_idx, new_idx, order=Chem.rdchem.BondType.SINGLE)
    combined = ed.GetMol()

    rw = Chem.RWMol(combined)
    h_inds = [
        nbr.GetIdx()
        for nbr in rw.GetAtomWithIdx(new_idx).GetNeighbors()
        if rw.GetAtomWithIdx(nbr.GetIdx()).GetAtomicNum() == 1
    ]
    if h_inds:
        place_h_in_tetrahedral(rw, new_idx, h_inds)
    return rw.GetMol()


def attach_hydrogen_cap(base_mol: Chem.Mol, terminal_idx: int) -> Chem.Mol:
    terminal_pos, v_norm = get_vector(base_mol, terminal_idx)
    atom_num = base_mol.GetAtomWithIdx(terminal_idx).GetAtomicNum()
    bond_length = estimate_bond_length(atom_num, 1, fallback=1.1)
    H_pos = terminal_pos + v_norm * bond_length

    editable_mol = Chem.EditableMol(base_mol)
    new_H_idx = editable_mol.AddAtom(Chem.Atom(1))
    editable_mol.AddBond(
        terminal_idx,
        new_H_idx,
        Chem.BondType.SINGLE,
    )
    capped = editable_mol.GetMol()
    conformer = capped.GetConformer()
    conformer.SetAtomPosition(new_H_idx, Point3D(*H_pos))
    return capped

def attach_methyl_cap(base_mol: Chem.Mol, terminal_idx: int) -> Chem.Mol:
    fragment = Chem.AddHs(Chem.MolFromSmiles('C'))
    params = AllChem.ETKDG()
    params.randomSeed = -1
    if AllChem.EmbedMolecule(fragment, params) != 0:
        logger.warning("3D embedding failed for methyl cap; proceeding without optimization.")
    h_atoms = [a.GetIdx() for a in fragment.GetAtoms() if a.GetSymbol() == 'H']
    if not h_atoms:
        raise ValueError("Failed to construct methyl fragment with hydrogens.")
    em = Chem.EditableMol(fragment)
    em.RemoveAtom(h_atoms[0])  # 删除一个 H 以连接主链
    fragment = em.GetMol()

    connection_idx = [a.GetIdx() for a in fragment.GetAtoms() if a.GetSymbol() == 'C'][0]
    tail_pos, vec = get_vector(base_mol, terminal_idx)
    atom_poly = base_mol.GetAtomWithIdx(terminal_idx).GetAtomicNum()
    atom_cap = fragment.GetAtomWithIdx(connection_idx).GetAtomicNum()
    bond_length = estimate_bond_length(atom_poly, atom_cap)
    target_pos = tail_pos + (bond_length + 0.1) * vec

    aligned_fragment = align_monomer_unit(
        Chem.Mol(fragment),
        connection_idx,
        target_pos,
        vec,
    )
    return attach_fragment(base_mol, aligned_fragment, terminal_idx, connection_idx)


def attach_custom_cap(base_mol: Chem.Mol, terminal_idx: int, cap_smiles: str) -> Chem.Mol:
    cap_mol, connection_idx, attachment_vec = prepare_cap_monomer(cap_smiles)

    tail_pos, vec = get_vector(base_mol, terminal_idx)
    atom_poly = base_mol.GetAtomWithIdx(terminal_idx).GetAtomicNum()
    atom_cap = cap_mol.GetAtomWithIdx(connection_idx).GetAtomicNum()
    bond_length = estimate_bond_length(atom_poly, atom_cap)
    target_pos = tail_pos + (bond_length + 0.1) * vec

    aligned_fragment = align_monomer_unit(
        Chem.Mol(cap_mol),
        connection_idx,
        target_pos,
        vec,
        local_reference_direction=attachment_vec,
    )
    return attach_fragment(base_mol, aligned_fragment, terminal_idx, connection_idx)


def attach_default_cap(base_mol: Chem.Mol, terminal_idx: int) -> Chem.Mol:
    atom = base_mol.GetAtomWithIdx(terminal_idx)
    h_count = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1)
    if atom.GetAtomicNum() == 6 and h_count == 2:
        return attach_hydrogen_cap(base_mol, terminal_idx)
    return attach_methyl_cap(base_mol, terminal_idx)


def gen_3D_withcap(mol, start_atom, end_atom, length, left_cap_smiles=None, right_cap_smiles=None):

    capped_mol = Chem.Mol(mol)
    terminal_data = [
        (start_atom, left_cap_smiles),
        (end_atom, right_cap_smiles),
    ]

    for terminal_idx, cap_smiles in terminal_data:
        if cap_smiles:
            try:
                capped_mol = attach_custom_cap(capped_mol, terminal_idx, cap_smiles)
                continue
            except ValueError as exc:
                logger.error(
                    "Failed to apply custom cap %s at atom %s: %s. Using default capping.",
                    cap_smiles,
                    terminal_idx,
                    exc,
                )
        capped_mol = attach_default_cap(capped_mol, terminal_idx)

    # 检查原子间距离是否合理
    valid_structure = check_3d_structure(capped_mol)
    if length <= 3 or valid_structure:
        return capped_mol

    logger.warning("Failed to generate the final PDB file.")
    return None

def check_3d_structure(mol, confId=0, dist_min=0.7, bond_s=2.7, bond_a=1.9, bond_d=1.8, bond_t=1.4):

    coord = np.array(mol.GetConformer(confId).GetPositions())

    dist_matrix = model_lib.distance_matrix(coord)
    dist_matrix = np.where(dist_matrix == 0, dist_min, dist_matrix)

    # Cheking bond length
    bond_l_c = True
    for b in mol.GetBonds():
        bond_l = dist_matrix[b.GetBeginAtom().GetIdx(), b.GetEndAtom().GetIdx()]
        if b.GetBondTypeAsDouble() == 1.0 and bond_l > bond_s:
            bond_l_c = False
            break
        elif b.GetBondTypeAsDouble() == 1.5 and bond_l > bond_a:
            bond_l_c = False
            break
        elif b.GetBondTypeAsDouble() == 2.0 and bond_l > bond_d:
            bond_l_c = False
            break
        elif b.GetBondTypeAsDouble() == 3.0 and bond_l > bond_t:
            bond_l_c = False
            break

    if dist_matrix.min() >= dist_min and bond_l_c:
        check = True
    else:
        check = False

    return check

def calculate_box_size(numbers, pdb_files, density):
    total_mass = 0
    for num, file in zip(numbers, pdb_files):

        molecular_weight = calc_mol_weight(file)  # in g/mol
        total_mass += molecular_weight * num / 6.022e23  # accumulate mass of each molecule in grams

    total_volume = total_mass / density  # volume in cm^3
    length = (total_volume * 1e24) ** (1 / 3)  # convert to Angstroms
    return length


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




