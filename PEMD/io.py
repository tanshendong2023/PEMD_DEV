# PEMD/io.py

"""
I/O utilities: format conversion and file read/write.
"""

import re
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel as ob
from openbabel import pybel


# OpenBabel setup
obConversion = ob.OBConversion()
ff = ob.OBForceField.FindForceField('UFF')
mol = ob.OBMol()
np.set_printoptions(precision=20)

def rdkitmol2xyz(name: str, mol: Chem.Mol, out_dir: str = '.', conf_id: int = 0) -> str:
    """
    Write an RDKit Mol object to an XYZ file.
    Tries RDKit's MolToXYZFile first; falls back to OpenBabel if that fails.
    Returns the path to the written XYZ file.
    """
    os.makedirs(out_dir, exist_ok=True)
    xyz_path = os.path.join(out_dir, f"{name}.xyz")
    try:
        Chem.MolToXYZFile(mol, xyz_path, confId=conf_id)
    except Exception:
        # Fallback via writing a MOL file then converting with OpenBabel
        mol_path = os.path.join(out_dir, f"{name}.mol")
        Chem.MolToMolFile(mol, mol_path, confId=conf_id)
        ob_conversion = ob.OBConversion()
        ob_conversion.SetInAndOutFormats("mol", "xyz")
        ob_mol = ob.OBMol()
        ob_conversion.ReadFile(ob_mol, mol_path)
        ob_conversion.WriteFile(ob_mol, xyz_path)
    return xyz_path

def smile_toxyz(name: str, smiles: str, out_dir: str = '.') -> str:
    """
    Embed 3D coordinates from SMILES and write to an XYZ file.
    Returns the path to the written XYZ file.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    AllChem.EmbedMolecule(mol)
    mol.SetProp("_Name", name)
    AllChem.UFFOptimizeMolecule(mol)
    return rdkitmol2xyz(name, mol, out_dir, conf_id=-1)

def convert_xyz_to_pdb(xyz_file: str, pdb_file: str, mol_name: str, resname: str) -> str:
    """
    Convert an XYZ file to PDB using OpenBabel, setting molecule name and residue name.
    Returns the path to the written PDB file.
    """
    ob_conversion = ob.OBConversion()
    ob_conversion.SetInAndOutFormats("xyz", "pdb")
    ob_mol = ob.OBMol()
    ob_conversion.ReadFile(ob_mol, xyz_file)
    ob_mol.SetTitle(mol_name)
    for atom in ob.OBMolAtomIter(ob_mol):
        atom.GetResidue().SetName(resname)
    # Ensure the output directory exists
    out_dir = os.path.dirname(pdb_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ob_conversion.WriteFile(ob_mol, pdb_file)
    return pdb_file

def convert_pdb_to_xyz(pdb_file: str, xyz_file: str) -> str:
    """
    Convert a PDB file to XYZ using OpenBabel.
    Returns the path to the written XYZ file.
    """
    ob_conversion = ob.OBConversion()
    ob_conversion.SetInAndOutFormats("pdb", "xyz")
    ob_mol = ob.OBMol()
    ob_conversion.ReadFile(ob_mol, pdb_file)
    ob_conversion.WriteFile(ob_mol, xyz_file)
    return xyz_file

def convert_xyz_to_mol2(xyz_file: str, mol2_file: str, mol_name: str, resname: str) -> str:
    """
    Convert an XYZ file to MOL2 using OpenBabel, setting molecule and residue names.
    Returns the path to the written MOL2 file.
    """
    ob_conversion = ob.OBConversion()
    ob_conversion.SetInAndOutFormats("xyz", "mol2")
    ob_mol = ob.OBMol()
    ob_conversion.ReadFile(ob_mol, xyz_file)
    ob_mol.SetTitle(mol_name)
    for atom in ob.OBMolAtomIter(ob_mol):
        atom.GetResidue().SetName(resname)
    ob_conversion.WriteFile(ob_mol, mol2_file)
    return mol2_file

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


def smiles_to_pdb(smiles: str,
                  output_file: str,
                  molecule_name: str,
                  resname: str,
                  max_confs: int = 5,
                  large_threshold: int = 90000) -> None:
    """
    Convert a SMILES string to a PDB (or mmCIF if very large) file.
    """
    try:
        # 1) Parse and add H
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string.")
        mol = Chem.AddHs(mol)

        # 2) Embed multiple conformers
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.maxAttempts = 5000
        params.useRandomCoords = True
        conf_ids = AllChem.EmbedMultipleConfs(mol,
                                              numConfs=max_confs,
                                              params=params)
        if not conf_ids:
            raise RuntimeError("ETKDG embedding failed for all conformers.")

        # 3) Prepare MMFF props
        props = AllChem.MMFFGetMoleculeProperties(mol,
                                                  mmffVariant='MMFF94')

        best_id, best_e = None, float('inf')
        for cid in conf_ids:
            AllChem.MMFFOptimizeMolecule(mol,
                                         confId=cid,
                                         maxIters=200)
            ff = AllChem.MMFFGetMoleculeForceField(mol,
                                                   props,
                                                   nonBondedThresh=100.0,
                                                   confId=cid)
            E = ff.CalcEnergy()
            if E < best_e:
                best_e, best_id = E, cid

        # 4) Fallback if best_id never set
        if best_id not in conf_ids:
            best_id = conf_ids[0]
            print(f"Warning: best_id was unset, defaulting to {best_id}")

        # 5) Clone the best conformer before removing
        orig_conf = mol.GetConformer(best_id)
        new_conf = Chem.Conformer(orig_conf)  # deep copy

        # 6) Retain only the best conformer
        mol.RemoveAllConformers()
        mol.AddConformer(new_conf, assignId=True)

        # 7) Write to temp SDF
        tmp_sdf = "temp.sdf"
        with Chem.SDWriter(tmp_sdf) as writer:
            writer.write(mol)

        # 8) Convert to PDB or mmCIF
        obConversion = ob.OBConversion()
        fmt = "cif" if mol.GetNumAtoms() > large_threshold else "pdb"
        obConversion.SetInAndOutFormats("sdf", fmt)

        obmol = ob.OBMol()
        if not obConversion.ReadFile(obmol, tmp_sdf):
            raise IOError("Failed to read temporary SDF.")

        obmol.SetTitle(molecule_name)
        for atom in ob.OBMolAtomIter(obmol):
            atom.GetResidue().SetName(resname)

        if not obConversion.WriteFile(obmol, output_file):
            raise IOError(f"Failed to write {fmt.upper()} file.")

        print(f"Successfully wrote {output_file} "
              f"({mol.GetNumAtoms()} atoms, format={fmt.upper()})")

    except Exception as e:
        print(f"Error in smiles_to_pdb: {e}")
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

def convert_rdkit_to_openbabel(rdkit_mol):
    """
    将 RDKit 的 Mol 对象转换为 Open Babel 的 OBMol 对象
    """
    # 创建 Open Babel 的 OBConversion 对象
    ob_conversion = ob.OBConversion()
    ob_conversion.SetOutFormat("mol")  # 使用 MOL 格式

    # 将 RDKit 的 Mol 对象转换为 MolBlock 字符串
    mol_block = Chem.MolToMolBlock(rdkit_mol)

    # 使用 Open Babel 解析 MolBlock 并生成 OBMol 对象
    ob_mol = ob.OBMol()
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


def remove_numbers_from_residue_names(mol2_filename, resname):
    with open(mol2_filename, 'r') as file:
        content = file.read()

    # 使用正则表达式删除特定残基名称后的数字1（确保只删除末尾的数字1）
    updated_content = re.sub(r'({})1\b'.format(resname), r'\1', content)

    with open(mol2_filename, 'w') as file:
        file.write(updated_content)


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

