"""
Polymer model building tools.

Developed by: Tan Shendong
Date: 2024.01.18
"""


import os
import math

from rdkit import Chem
from PEMD.model import model_lib
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolTransforms


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

def gen_copoly_smiles(poly_name, repeating_unit, x_length, y_length):
    # Generate the SMILES connecting two type units.
    (
        dum1,
        dum2,
        atom1,
        atom2,
    ) = model_lib.Init_info(
        poly_name,
        repeating_unit[0],
    )

    (
        dum3,
        dum4,
        atom3,
        atom4,
    ) = model_lib.Init_info(
        poly_name,
        repeating_unit[1],
    )

    # Connecting x first type units.
    (
        inti_mol_x,
        monomer_mol_x,
        start_atom_x,
        end_atom_x,
    ) = model_lib.gen_smiles_nocap(
        dum1,
        dum2,
        atom1,
        atom2,
        repeating_unit[0],
        x_length,
    )

    # Connecting y second type units.
    (
        inti_mol_y,
        monomer_mol_y,
        start_atom_y,
        end_atom_y,
    ) = model_lib.gen_smiles_nocap(
        dum3,
        dum4,
        atom3,
        atom4,
        repeating_unit[1],
        y_length,
    )

    # Delete dum2 of first type units.
    (
        dumx1,
        dumx2,
        atomx1,
        atomx2,
    ) = model_lib.Init_info(
        poly_name,
        Chem.MolToSmiles(inti_mol_x, canonical=False), # Must add canonical is false to genereate right smiles corresbonding to mol objective.
    )

    edit_m_x = Chem.EditableMol(inti_mol_x)
    edit_m_x.RemoveAtom(int(dumx2))
    molx_without_dum2 = Chem.Mol(edit_m_x.GetMol())

    if atomx1 > atomx2:
        atomx1, atomx2 = atomx2, atomx1

    if  atomx2 > dumx2:
        second_atom_x = atomx2 - 1
    else:
        second_atom_x = atomx2

    # Delete dum1 of second type units.
    (
        dumy1,
        dumy2,
        atomy1,
        atomy2,
    ) = model_lib.Init_info(
        poly_name,
        Chem.MolToSmiles(inti_mol_y, canonical=False),
    )

    edit_m_y = Chem.EditableMol(inti_mol_y)
    edit_m_y.RemoveAtom(dumy1) # Delete dum1
    moly_without_dum1 = Chem.Mol(edit_m_y.GetMol())
    if atomy1 > atomy2:
        atomy1, atomy2 = atomy2, atomy1

    if  atomy1 > dumy1:
        first_atom_y = atomy1 - 1
    else:
        first_atom_y = atomy1

    # Connecting first and second type units.
    combo = Chem.CombineMols(molx_without_dum2, moly_without_dum1)
    edcombo = Chem.EditableMol(combo)
    edcombo.AddBond(
        second_atom_x,
        first_atom_y + molx_without_dum2.GetNumAtoms(),
        order=Chem.rdchem.BondType.SINGLE,
    )# Add bond according to the index of atoms to be connected
    unit_mol = edcombo.GetMol()
    unit_smiles = Chem.MolToSmiles(unit_mol)

    return unit_smiles

def gen_poly_3D(poly_name, repeating_unit, length, max_retries = 10):

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
    ) = model_lib.gen_3D_nocap(
        dum1,
        dum2,
        atom1,
        atom2,
        repeating_unit,
        length,
        max_retries
    )

    mol_3D = model_lib.gen_3D_withcap(
        inti_mol3,
        start_atom,
        end_atom,
        max_retries
    )

    # 处理基于 SMILES 的分子
    # mol = Chem.MolFromSmiles(smiles)
    # mol_with_h = Chem.AddHs(mol)

    # mol = pybel.readstring("smi", smiles)
    # mol.addh()
    # mol.make3D()
    # mol_with_h = convert_pybel_to_rdkit(mol)
    # ob_conversion = openbabel.OBConversion()
    # ob_conversion.SetInFormat("smi")
    #
    # ob_mol = openbabel.OBMol()
    # ob_conversion.ReadString(ob_mol, smiles)
    # ob_mol.AddHydrogens()
    #
    # mapping = model_lib.get_atom_mapping(mol_3D, ob_mol)
    #
    # if mapping:
        # 复制坐标
        # reordered_mol_3D = model_lib.reorder_atoms(mol_3D, mapping)

        # 保存最终 PDB 文件
    # pdb_file = os.path.join(work_dir, f"{poly_name}_N{length}.pdb")
    # Chem.MolToXYZFile(mol_3D, 'mid.xyz')
    # model_lib.convert_xyz_to_pdb('mid.xyz', pdb_file, poly_name, poly_resname)
    # os.remove('mid.xyz')
    #
    # print(f"Generated {pdb_file}")

    return mol_3D

# def gen_poly_3D(work_dir, poly_name, poly_resname, length, smiles, max_attempts=3):
#     # Read SMILES using Pybel and generate a molecule object
#     mol = pybel.readstring("smi", smiles)
#     mol.addh()
#     mol.make3D()
#     obmol = mol.OBMol
#     DEG_TO_RAD = math.pi / 180
#     num_iterations = 10
#     angle_range = (0, 360)
#
#     # Randomly set torsion angles
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
#                 except StopIteration:
#                     continue
#
#     # Perform local optimization
#     mol.localopt()
#
#     # Convert Pybel molecule to RDKit molecule
#     rdkit_mol = convert_pybel_to_rdkit(mol)
#
#     # Initialize attempt counter
#     attempt = 0
#
#     while attempt < max_attempts:
#         long_bonds = check_bond_lengths_rdkit(rdkit_mol)
#         unreasonable_angles = check_bond_angles_rdkit(rdkit_mol)
#
#         if not long_bonds and not unreasonable_angles:
#             break  # Exit the loop if structure is acceptable
#
#         # Optionally, print detailed warnings
#         if long_bonds:
#             for bond in long_bonds:
#                 atom1_idx, atom2_idx, distance, standard_length = bond
#                 atom1 = rdkit_mol.GetAtomWithIdx(atom1_idx - 1).GetSymbol()
#                 atom2 = rdkit_mol.GetAtomWithIdx(atom2_idx - 1).GetSymbol()
#
#         if unreasonable_angles:
#             for angle in unreasonable_angles:
#                 atom1_idx, center_atom_idx, atom2_idx, actual_angle, expected_angle = angle
#                 atom1 = rdkit_mol.GetAtomWithIdx(atom1_idx - 1).GetSymbol()
#                 center_atom = rdkit_mol.GetAtomWithIdx(center_atom_idx - 1).GetSymbol()
#                 atom2 = rdkit_mol.GetAtomWithIdx(atom2_idx - 1).GetSymbol()
#
#         # Re-optimize the molecule
#         mol.localopt()
#
#         # Convert again to RDKit molecule after optimization
#         rdkit_mol = convert_pybel_to_rdkit(mol)
#
#         attempt += 1
#
#     pdb_file = os.path.join(work_dir, f"{poly_name}_N{length}.pdb")
#     Chem.MolToXYZFile(rdkit_mol, 'mid.xyz')
#     model_lib.convert_xyz_to_pdb('mid.xyz', pdb_file, poly_name, poly_resname)
#     os.remove('mid.xyz')
#
#     return f"{poly_name}_N{length}.pdb"

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


