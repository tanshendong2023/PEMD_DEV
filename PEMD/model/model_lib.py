"""
PEMD code library.

Developed by: Tan Shendong
Date: 2024.03.15
"""


import re
import subprocess
import numpy as np
import pandas as pd
from collections import deque
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel as ob
from openbabel import openbabel
from openbabel import pybel
from rdkit.Geometry import Point3D
from rdkit.Chem import Descriptors
from networkx.algorithms import isomorphism


# OpenBabel setup
obConversion = ob.OBConversion()
ff = ob.OBForceField.FindForceField('UFF')
mol = ob.OBMol()
np.set_printoptions(precision=20)

def count_atoms(mol, atom_type, length):
    # Initialize the counter for the specified atom type
    atom_count = 0
    # Iterate through all atoms in the molecule
    for atom in mol.GetAtoms():
        # Check if the atom is of the specified type
        if atom.GetSymbol() == atom_type:
            atom_count += 1
    return round(atom_count / length)

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

def mol_to_nx(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), element=atom.GetSymbol())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return G


def is_isomorphic(G1, G2):
    GM = isomorphism.GraphMatcher(G1, G2, node_match=lambda x, y: x['element'] == y['element'])
    return GM.is_isomorphic()

def convert_pdb_to_xyz(pdb_filename, xyz_filename):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "xyz")

    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, pdb_filename)

    obConversion.WriteFile(mol, xyz_filename)

def convert_xyz_to_pdb(xyz_filename, pdb_filename, molecule_name, resname):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "pdb")

    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, xyz_filename)
    mol.SetTitle(molecule_name)

    for atom in openbabel.OBMolAtomIter(mol):
        res = atom.GetResidue()
        res.SetName(resname)
    obConversion.WriteFile(mol, pdb_filename)


def convert_xyz_to_mol2(xyz_filename, mol2_filename, molecule_name, resname):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "mol2")

    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, xyz_filename)

    mol.SetTitle(molecule_name)

    for atom in openbabel.OBMolAtomIter(mol):
        res = atom.GetResidue()
        if res:  # 确保残基信息存在
            res.SetName(resname)

    obConversion.WriteFile(mol, mol2_filename)

    remove_numbers_from_residue_names(mol2_filename, resname)


def convert_gro_to_pdb(input_gro_path, output_pdb_path):

    try:
        # Load the GRO file
        mol = next(pybel.readfile('gro', input_gro_path))

        # Save as PDB
        mol.write('pdb', output_pdb_path, overwrite=True)
        print(f"Gro converted to pdb successfully {output_pdb_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

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

def mol_to_xyz(mol, conf_id, filename):
    """将RDKit分子对象的构象保存为XYZ格式文件"""
    xyz = Chem.MolToXYZBlock(mol, confId=conf_id)
    with open(filename, 'w') as f:
        f.write(xyz)

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


def log_to_xyz(log_file_path, xyz_file_path):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("g09", "xyz")
    mol = openbabel.OBMol()

    try:
        obConversion.ReadFile(mol, log_file_path)
        obConversion.WriteFile(mol, xyz_file_path)
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

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
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
    if mol is not None:
        mol_weight = Descriptors.MolWt(mol)
        return mol_weight
    else:
        raise ValueError(f"Unable to read molecular structure from {pdb_file}")

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
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("sdf", "pdb")
        obmol = openbabel.OBMol()
        if not obConversion.ReadFile(obmol, tmp_sdf):
            raise IOError("Failed to read from the temporary SDF file.")

        # Set molecule name in OpenBabel
        obmol.SetTitle(molecule_name)

        # Set residue name for all atoms in the molecule in OpenBabel
        for atom in openbabel.OBMolAtomIter(obmol):
            res = atom.GetResidue()
            res.SetName(resname)

        if not obConversion.WriteFile(obmol, output_file):
            raise IOError("Failed to write the PDB file.")

        print(f"PDB file successfully created: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

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
    """
    计算并返回平均体积及最接近平均体积的帧索引。
    """
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













