"""
Polymer model building tools.

Developed by: Tan Shendong
Date: 2024.01.18
"""


import os
import math
import random
from rdkit import Chem
from openbabel import pybel
from PEMD.model import model_lib
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D


def gen_poly_smiles(poly_name, repeating_unit, length, leftcap, rightcap,):
    # Generate the SMILES representation of the polymer.
    (
        dum1,
        dum2,
        atom1,
        atom2,
    ) = model_lib.Init_info(
        poly_name,
        repeating_unit,
    )

    smiles_poly = model_lib.gen_oligomer_smiles(
        poly_name,
        dum1,
        dum2,
        atom1,
        atom2,
        repeating_unit,
        length,
        leftcap,
        rightcap,
    )

    if os.path.exists(poly_name + '.xyz'):
        os.remove(poly_name + '.xyz')             # Delete intermediate XYZ file if exists

    return smiles_poly

def gen_poly_3D(work_dir, poly_name, length, smiles, max_attempts=3):
    # Read SMILES using Pybel and generate a molecule object
    mol = pybel.readstring("smi", smiles)
    mol.addh()
    mol.make3D()
    obmol = mol.OBMol
    DEG_TO_RAD = math.pi / 180
    num_iterations = 10
    angle_range = (0, 360)

    # Randomly set torsion angles
    for _ in range(num_iterations):
        for obatom in pybel.ob.OBMolAtomIter(obmol):
            for bond in pybel.ob.OBAtomBondIter(obatom):
                neighbor = bond.GetNbrAtom(obatom)
                if len(list(pybel.ob.OBAtomAtomIter(neighbor))) < 2:
                    continue
                angle = random.uniform(*angle_range)
                try:
                    n1 = next(pybel.ob.OBAtomAtomIter(neighbor))
                    n2 = next(pybel.ob.OBAtomAtomIter(n1))
                    obmol.SetTorsion(obatom.GetIdx(), neighbor.GetIdx(), n1.GetIdx(), n2.GetIdx(), angle * DEG_TO_RAD)
                except StopIteration:
                    continue

    # Perform local optimization
    mol.localopt()

    # Convert Pybel molecule to RDKit molecule
    rdkit_mol = convert_pybel_to_rdkit(mol)

    # Initialize attempt counter
    attempt = 0

    while attempt < max_attempts:
        # Check bond lengths
        long_bonds = check_bond_lengths_rdkit(rdkit_mol)
        # Check bond angles
        unreasonable_angles = check_bond_angles_rdkit(rdkit_mol)

        if not long_bonds and not unreasonable_angles:
            print("All bond lengths and angles are within the reasonable range.")
            break  # Exit the loop if structure is acceptable

        print(f"Attempt {attempt + 1} of {max_attempts}: Detected issues in the molecular geometry.")

        # Optionally, print detailed warnings
        if long_bonds:
            print(f"  - Detected {len(long_bonds)} bonds exceeding the reasonable length:")
            for bond in long_bonds:
                atom1_idx, atom2_idx, distance, standard_length = bond
                atom1 = rdkit_mol.GetAtomWithIdx(atom1_idx - 1).GetSymbol()
                atom2 = rdkit_mol.GetAtomWithIdx(atom2_idx - 1).GetSymbol()
                print(
                    f"    * Bond between atom {atom1_idx} ({atom1}) and atom {atom2_idx} ({atom2}) has a distance of {distance:.2f} Å (standard: {standard_length} Å)")

        if unreasonable_angles:
            print(f"  - Detected {len(unreasonable_angles)} bond angles outside the reasonable range:")
            for angle in unreasonable_angles:
                atom1_idx, center_atom_idx, atom2_idx, actual_angle, expected_angle = angle
                atom1 = rdkit_mol.GetAtomWithIdx(atom1_idx - 1).GetSymbol()
                center_atom = rdkit_mol.GetAtomWithIdx(center_atom_idx - 1).GetSymbol()
                atom2 = rdkit_mol.GetAtomWithIdx(atom2_idx - 1).GetSymbol()
                print(
                    f"    * Bond angle between atom {atom1_idx} ({atom1}) - atom {center_atom_idx} ({center_atom}) - atom {atom2_idx} ({atom2}) is {actual_angle:.2f}°, expected {expected_angle}°")

        # Re-optimize the molecule
        print("Re-optimizing the molecule...")
        mol.localopt()

        # Convert again to RDKit molecule after optimization
        rdkit_mol = convert_pybel_to_rdkit(mol)

        attempt += 1

    if attempt == max_attempts and (long_bonds or unreasonable_angles):
        print("Maximum optimization attempts reached. Some bonds or angles are still outside the reasonable range.")
        # Optionally, raise an exception or proceed with warnings
        # raise ValueError("Unacceptable molecular geometry detected after multiple optimization attempts.")

    # Export to PDB file
    pdb_file = os.path.join(work_dir, f"{poly_name}_N{length}.pdb")
    mol.write("pdb", pdb_file, overwrite=True)

    return f"{poly_name}_N{length}.pdb"

def convert_pybel_to_rdkit(pybel_mol):
    """
    Convert a Pybel molecule to an RDKit molecule.
    """
    mol_block = pybel_mol.write("mol")
    rdkit_mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
    if rdkit_mol is None:
        raise ValueError("Unable to convert Pybel molecule to RDKit molecule.")
    return rdkit_mol

def check_bond_lengths_rdkit(rdkit_mol, max_deviation=0.2):
    STANDARD_BOND_LENGTHS = {
        Chem.rdchem.BondType.SINGLE: 1.54,
        Chem.rdchem.BondType.DOUBLE: 1.34,
        Chem.rdchem.BondType.TRIPLE: 1.20,
        Chem.rdchem.BondType.AROMATIC: 1.39,
        # Add more bond types as needed
    }

    def get_standard_bond_length(bond):
        bond_type = bond.GetBondType()
        return STANDARD_BOND_LENGTHS.get(bond_type, None)

    long_bonds = []
    conf = rdkit_mol.GetConformer()
    for bond in rdkit_mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        idx1 = atom1.GetIdx()
        idx2 = atom2.GetIdx()
        pos1 = conf.GetAtomPosition(idx1)
        pos2 = conf.GetAtomPosition(idx2)
        distance = pos1.Distance(pos2)
        standard_length = get_standard_bond_length(bond)
        if standard_length is None:
            continue  # Skip bonds without defined standard lengths
        max_allowed = standard_length * (1 + max_deviation)
        if distance > max_allowed:
            long_bonds.append((idx1 + 1, idx2 + 1, distance, standard_length))  # RDKit indices start at 0
    return long_bonds

def check_bond_angles_rdkit(rdkit_mol, max_deviation=15):
    unreasonable_angles = []
    conf = rdkit_mol.GetConformer()

    # Define expected bond angles (can be extended as needed)
    EXPECTED_ANGLES = {
        'C': 109.5,  # Tetrahedral
        'O': 104.5,  # Similar to water molecule
        'N': 107.0,  # Triazine rings, etc.
        # Add more atom types as needed
    }

    for atom in rdkit_mol.GetAtoms():
        neighbors = atom.GetNeighbors()
        if len(neighbors) < 2:
            continue
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                atom1 = neighbors[i]
                atom2 = neighbors[j]
                idx1 = atom1.GetIdx()
                idx2 = atom2.GetIdx()
                center_idx = atom.GetIdx()

                pos1 = conf.GetAtomPosition(idx1)
                pos2 = conf.GetAtomPosition(idx2)
                pos_center = conf.GetAtomPosition(center_idx)

                # Calculate angle
                angle_rad = rdMolTransforms.GetAngleRad(rdkit_mol.GetConformer(), center_idx, idx1, idx2)
                angle_deg = math.degrees(angle_rad)

                # Get expected angle
                center_atom_type = atom.GetSymbol()
                expected_angle = EXPECTED_ANGLES.get(center_atom_type, 109.5)  # Default to tetrahedral

                if abs(angle_deg - expected_angle) > max_deviation:
                    unreasonable_angles.append((idx1 + 1, center_idx + 1, idx2 + 1, angle_deg, expected_angle))

    return unreasonable_angles

# def gen_poly_3D(work_dir, poly_name, length, smiles):
#     mol = pybel.readstring("smi", smiles)
#     mol.addh()
#     mol.make3D()
#     obmol = mol.OBMol
#     DEG_TO_RAD = math.pi / 180
#     num_iterations = 10
#     angle_range = (0, 360)
#
#     for _ in range(num_iterations):
#         for obatom in pybel.ob.OBMolAtomIter(obmol):
#             for bond in pybel.ob.OBAtomBondIter(obatom):
#                 neighbor = bond.GetNbrAtom(obatom)
#                 if len(list(pybel.ob.OBAtomAtomIter(neighbor))) < 2:
#                     continue
#                 angle = random.uniform(*angle_range)
#                 try:
#                     n1 = next(pybel.ob.OBAtomAtomIter(neighbor))
#                     n2 = next(pybel.ob.OBAtomAtomIter(n1))
#                     obmol.SetTorsion(obatom.GetIdx(), neighbor.GetIdx(), n1.GetIdx(), n2.GetIdx(), angle * DEG_TO_RAD)
#
#                 except StopIteration:
#                     continue
#
#     mol.localopt()
#
#     pdb_file = os.path.join(work_dir, f"{poly_name}_N{length}.pdb")
#     mol.write("pdb", pdb_file, overwrite=True)
#
#     return f"{poly_name}_N{length}.pdb"

# def gen_poly_3D(work_dir, poly_name, length, smiles):
#     # 读取SMILES字符串并生成分子
#     mol = pybel.readstring("smi", smiles)
#     mol.addh()
#     mol.make3D()
#
#     obmol = mol.OBMol
#     DEG_TO_RAD = math.pi / 180
#     num_iterations = 10
#     angle_range = (0, 360)
#
#     # 使用力场初始化优化以避免初始重叠
#     mol.localopt(forcefield="MMFF94", steps=500)
#
#     for _ in range(num_iterations):
#         torsion_set = False
#         for obatom in openbabel.OBMolAtomIter(obmol):
#             for bond in openbabel.OBAtomBondIter(obatom):
#                 neighbor = bond.GetNbrAtom(obatom)
#                 # 仅对具有至少两个连接的原子进行扭转
#                 if len(list(openbabel.OBAtomAtomIter(neighbor))) < 2:
#                     continue
#                 angle = random.uniform(*angle_range)
#                 try:
#                     # 获取四个原子用于设置扭转角
#                     n1 = next(openbabel.OBAtomAtomIter(neighbor))
#                     n2 = next(openbabel.OBAtomAtomIter(n1))
#                     obmol.SetTorsion(obatom.GetIdx(), neighbor.GetIdx(), n1.GetIdx(), n2.GetIdx(), angle * DEG_TO_RAD)
#                     torsion_set = True
#                 except StopIteration:
#                     continue
#         if torsion_set:
#             # 每次迭代后进行力场优化，避免原子重叠
#             mol.localopt(forcefield="MMFF94", steps=500)
#             # 检查最小原子间距
#             min_distance = get_min_distance(obmol)
#             if min_distance < 1.0:  # 根据需要调整阈值
#                 print(f"警告：检测到最小原子距离 {min_distance:.2f} Å，小于阈值，重新调整结构。")
#                 # 可以选择重新设置扭转角或采取其他措施
#                 continue
#
#     mol.localopt(forcefield="MMFF94", steps=500)
#
#     pdb_file = os.path.join(work_dir, f"{poly_name}_N{length}.pdb")
#     mol.write("pdb", pdb_file, overwrite=True)

    # return f"{poly_name}_N{length}.pdb"


# def get_min_distance(obmol):
#     min_dist = float('inf')
#     atoms = list(openbabel.OBMolAtomIter(obmol))
#     num_atoms = len(atoms)
#     for i in range(num_atoms):
#         atom1 = atoms[i]
#         x1, y1, z1 = atom1.GetX(), atom1.GetY(), atom1.GetZ()
#         for j in range(i + 1, num_atoms):
#             atom2 = atoms[j]
#             x2, y2, z2 = atom2.GetX(), atom2.GetY(), atom2.GetZ()
#             distance = calculate_distance(x1, y1, z1, x2, y2, z2)
#             if distance < min_dist:
#                 min_dist = distance
#     return min_dist

# def calculate_distance(x1, y1, z1, x2, y2, z2):
#     return math.sqrt(
#         (x1 - x2) ** 2 +
#         (y1 - y2) ** 2 +
#         (z1 - z2) ** 2
#     )

def calc_poly_chains(num_Li_salt , conc_Li_salt, mass_per_chain):

    # calculate the mol of LiTFSI salt
    avogadro_number = 6.022e23  # unit 1/mol
    mol_Li_salt = num_Li_salt / avogadro_number # mol

    # calculate the total mass of the polymer
    total_mass_polymer =  mol_Li_salt / (conc_Li_salt / 1000)  # g

    # calculate the number of polymer chains
    num_chains = (total_mass_polymer*avogadro_number) / mass_per_chain  # no unit; mass_per_chain input unit g/mol

    return int(num_chains)

def calc_poly_length(total_mass_polymer, smiles_repeating_unit, smiles_leftcap, smiles_rightcap, ):
    # remove [*] from the repeating unit SMILES, add hydrogens, and calculate the molecular weight
    simplified_smiles_repeating_unit = smiles_repeating_unit.replace('[*]', '')
    molecule_repeating_unit = Chem.MolFromSmiles(simplified_smiles_repeating_unit)
    mol_weight_repeating_unit = Descriptors.MolWt(molecule_repeating_unit) - 2 * 1.008

    # remove [*] from the end group SMILES, add hydrogens, and calculate the molecular weight
    simplified_smiles_rightcap = smiles_rightcap.replace('[*]', '')
    simplified_smiles_leftcap = smiles_leftcap.replace('[*]', '')
    molecule_rightcap = Chem.MolFromSmiles(simplified_smiles_rightcap)
    molecule_leftcap = Chem.MolFromSmiles(simplified_smiles_leftcap)
    mol_weight_end_group = Descriptors.MolWt(molecule_rightcap) + Descriptors.MolWt(molecule_leftcap) - 2 * 1.008

    # calculate the mass of the polymer chain
    mass_polymer_chain = total_mass_polymer - mol_weight_end_group

    # calculate the number of repeating units in the polymer chain
    length = round(mass_polymer_chain / mol_weight_repeating_unit)

    return length


