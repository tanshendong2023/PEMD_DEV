"""
PEMD code library.

Developed by: Tan Shendong
Date: 2025.05.23
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import PEMD.io as io
import PEMD.constants as const

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
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


def gen_sequence_copolymer_3D(name, smiles_A, smiles_B, sequence, bond_length=1.5):
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
        if has_overlapping_atoms(connecting_mol):
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
        if has_overlapping_atoms(combined_mol):
            combined_mol = local_optimize(combined_mol, maxIters=150)
        connecting_mol = Chem.RWMol(combined_mol)

        tail_idx = num_atom + t
        num_atom = num_atom + new_unit.GetNumAtoms()

    length = len(sequence)
    final_poly = gen_3D_withcap(connecting_mol, h_1, tail_idx, length)

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

def align_monomer_unit(monomer, connection_atom_idx, target_position, target_direction):
    conf = monomer.GetConformer()
    B = np.array(conf.GetAtomPosition(connection_atom_idx))
    if np.linalg.norm(target_direction) < const.MIN_DIRECTION_NORM:
        logger.warning("Target direction is too small; using default direction.")
        target_direction = const.DEFAULT_DIRECTION
    _, local_dir = get_vector(monomer, connection_atom_idx)
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

def has_overlapping_atoms(mol, connected_distance=1.0, disconnected_distance=1.56):
    """
    检查分子中是否存在原子重叠：
      - 如果两个原子通过化学键相连，则允许的最小距离为 connected_distance
      - 如果不相连，则默认使用 disconnected_distance，
        如果任一原子为氧或卤素（F, Cl, Br, I）或两个原子均为碳，
        则要求最小距离为 1.6 Å（你也可以修改为 2.1 Å，根据需要）。
    当检测到原子对距离过近时，会输出相关信息，包括原子的名称。
    """
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
            actual_min = get_min_distance(mol, i, j, bond_graph,
                                          connected_distance,
                                          disconnected_distance)
            if distance < actual_min:
                atom_i = mol.GetAtomWithIdx(i)
                atom_j = mol.GetAtomWithIdx(j)
                coord_i = positions[i]
                coord_j = positions[j]
                conf_id = mol.GetConformer().GetId()
                logger.info(
                    "Overlapping detected in conformer %s: Atom %s (%d) at %s"
                    " and Atom %s (%d) at %s are too close (distance: %.2f Å, "
                    "allowed: %.2f Å)",
                    conf_id,
                    atom_i.GetSymbol(),
                    i,
                    coord_i,
                    atom_j.GetSymbol(),
                    j,
                    coord_j,
                    distance,
                    actual_min,
                )
                G.add_edge(i, j, weight=distance)

    # 如果图中有边，则表示存在重叠原子
    return len(G.edges) > 0

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
            if has_overlapping_atoms(mol):
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
    logger.error("Local optimization failed after {num_retries} attempts.")
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
            return 1.75
        # 判断条件：氧、卤素、氮和碳之间的连接返回 1.6 Å
        elif (symbol1 in ['O', 'F', 'Cl', 'Br', 'I'] and symbol2 in ['O', 'F', 'Cl', 'Br', 'I']) or \
                (symbol1 == 'C' and symbol2 == 'O') or (symbol1 == 'O' and symbol2 == 'C'):
            return 1.6
        else:
            return disconnected_distance

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

def gen_3D_withcap(mol, start_atom, end_atom, length):

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
            # 1. 构造 CH3· 片段
            mol_C = Chem.AddHs(Chem.MolFromSmiles('C'))
            AllChem.EmbedMolecule(mol_C, AllChem.ETKDG())
            h_atoms = [a.GetIdx() for a in mol_C.GetAtoms() if a.GetSymbol() == 'H']
            em = Chem.EditableMol(mol_C)
            em.RemoveAtom(h_atoms[0])  # 删除一个 H
            mol_C = em.GetMol()

            # 2. 连接索引
            tail_index = terminal_idx
            head_index = [a.GetIdx() for a in mol_C.GetAtoms() if a.GetSymbol() == 'C'][0]

            # 3. 计算目标位置并对齐
            tail_pos, vec = get_vector(capped_mol, tail_index)
            bond_length = 1.54
            target_pos = tail_pos + (bond_length + 0.1) * vec
            new_unit = align_monomer_unit(Chem.Mol(mol_C), head_index, target_pos, vec)

            # 4. 合并并加键（注意偏移）
            n1 = capped_mol.GetNumAtoms()
            combo = Chem.CombineMols(capped_mol, new_unit)
            ed = Chem.EditableMol(combo)
            new_idx = head_index + n1
            ed.AddBond(tail_index, new_idx, order=Chem.rdchem.BondType.SINGLE)
            combined = ed.GetMol()

            # 5. 四面体重排
            rw = Chem.RWMol(combined)
            h_inds = [nbr.GetIdx() for nbr in rw.GetAtomWithIdx(new_idx).GetNeighbors()
                      if rw.GetAtomWithIdx(nbr.GetIdx()).GetAtomicNum() == 1]
            if h_inds:
                place_h_in_tetrahedral(rw, new_idx, h_inds)
            capped_mol = rw.GetMol()

    # 检查原子间距离是否合理
    overlap = check_molecule_structure(capped_mol, energy_threshold=50.0)
    # return capped_mol
    if length <= 3 or not overlap:
        return capped_mol
    else:
        logger.warning("Failed to generate the final PDB file.")

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

