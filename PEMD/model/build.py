"""
Polymer model building tools.

Developed by: Tan Shendong
Date: 2024.01.18
"""


import os
import random
import itertools
import subprocess
import pandas as pd
from rdkit import Chem
from pathlib import Path
from shutil import which
from openbabel import pybel
from PEMD.model import model_lib
from rdkit.Chem import Descriptors
from LigParGenPEMD import Converter


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


def gen_poly_3D(poly_name, length, smiles, core = 32, atom_typing_ = 'pysimm', ):

    # build directory
    current_path = os.getcwd()
    out_dir = os.path.join(current_path, f'{poly_name}_N{length}')
    os.makedirs(out_dir, exist_ok=True)

    relax_polymer_lmp_dir = os.path.join(out_dir, 'relax_polymer_lmp')
    os.makedirs(relax_polymer_lmp_dir, exist_ok=True)

    # print(smiles)
    mol = pybel.readstring("smi", smiles)
    mol.addh()
    mol.make3D()
    obmol = mol.OBMol

    angle_range = (0, 0.5)
    for obatom in pybel.ob.OBMolAtomIter(obmol):
        for bond in pybel.ob.OBAtomBondIter(obatom):
            neighbor = bond.GetNbrAtom(obatom)
            if len(list(pybel.ob.OBAtomAtomIter(neighbor))) < 2:
                continue
            angle = random.uniform(*angle_range)
            n1 = next(pybel.ob.OBAtomAtomIter(neighbor))
            n2 = next(pybel.ob.OBAtomAtomIter(n1))
            obmol.SetTorsion(obatom.GetIdx(), neighbor.GetIdx(), n1.GetIdx(), n2.GetIdx(), angle)
    mol.localopt()

    # 构建文件名基础，这样可以避免重复拼接字符串
    file_base = f"{poly_name}_N{length}"

    # 使用os.path.join构建完整的文件路径，确保路径在不同操作系统上的兼容性
    pdb_file = os.path.join(relax_polymer_lmp_dir, f"{file_base}.pdb")
    xyz_file = os.path.join(relax_polymer_lmp_dir, f"{file_base}.xyz")
    mol_file = os.path.join(relax_polymer_lmp_dir, f"{file_base}.mol2")

    mol.write("pdb", pdb_file, overwrite=True)
    mol.write("xyz", xyz_file, overwrite=True)
    mol.write("mol2", mol_file, overwrite=True)

    print("\n", poly_name, ": Performing a short MD simulation using LAMMPS...\n", )

    model_lib.get_gaff2(poly_name, length, relax_polymer_lmp_dir, mol, atom_typing=atom_typing_)
    model_lib.relax_polymer_lmp(poly_name, length, relax_polymer_lmp_dir, core)


def calculate_box_size(numbers, pdb_files, density):
    total_mass = 0
    for num, file in zip(numbers, pdb_files):

        molecular_weight = model_lib.calc_mol_weight(file)  # in g/mol
        total_mass += molecular_weight * num / 6.022e23  # accumulate mass of each molecule in grams

    total_volume = total_mass / density  # volume in cm^3
    length = (total_volume * 1e24) ** (1 / 3)  # convert to Angstroms
    return length


# define the function to generate the packmol input file
def gen_packmol_input(model_info, density, add_length, out_dir, packinp_name='pack.inp',
                      packout_name='pack_cell.pdb'):

    current_path = os.getcwd()

    # unit_name = model_info['polymer']['compound']
    # length = model_info['polymer']['length'][1]

    MD_dir = os.path.join(current_path, out_dir)
    os.mkdir(MD_dir)
    # build_lib.build_dir(MD_dir)  # 确保这个函数可以正确创建目录

    packinp_path = os.path.join(MD_dir, packinp_name)

    numbers = model_lib.print_compounds(model_info,'numbers')
    compounds = model_lib.print_compounds(model_info,'compound')

    pdb_files = []
    for com in compounds:
        # if com == model_info['polymer']['compound']:
        #     ff_dir = os.path.join(current_path, f'{unit_name}_N{length}', 'ff_dir')
        #     filepath = os.path.join(ff_dir, f"{com}.pdb")
        # else:
        filepath = os.path.join(MD_dir, f"{com}.pdb")
        pdb_files.append(filepath)

    box_length = calculate_box_size(numbers, pdb_files, density) + add_length  # add 10 Angstroms to each side

    file_contents = "tolerance 2.0\n"
    file_contents += f"add_box_sides 1.2\n"
    file_contents += f"output {packout_name}\n"
    file_contents += "filetype pdb\n\n"
    file_contents += f"seed {random.randint(1, 100000)}\n\n"  # Add random seed for reproducibility

    # 循环遍历每种分子的数量和对应的 PDB 文件
    for num, file in zip(numbers, pdb_files):
        file_contents = file_contents + f"\nstructure {file}\n"
        file_contents = file_contents + f"  number {num}\n"
        file_contents = file_contents + f"  inside box 0.0 0.0 0.0 {box_length:.2f} {box_length:.2f} {box_length:.2f}\n"
        file_contents = file_contents + "end structure\n\n"

    # write to file
    with open(packinp_path, 'w') as file:
        file.write(file_contents)
    print(f"packmol input file generation successful：{packinp_path}")

    return packinp_path


def run_packmol(out_dir, input_file='pack.inp', output_file='pack.out'):
    current_path = os.getcwd()
    if not which("packmol"):
        raise RuntimeError(
            "Running Packmol requires the executable 'packmol' to be in the path. Please "
            "download packmol from https://github.com/leandromartinez98/packmol and follow the "
            "instructions in the README to compile. Don't forget to add the packmol binary to your path"
        )

    try:
        MD_dir = os.path.join(current_path, out_dir)
        os.chdir(MD_dir)
        p = subprocess.run(
            f"packmol < {input_file}",
            check=True,
            shell=True,
            capture_output=True,
        )

        # Check for errors in packmol output
        if "ERROR" in p.stdout.decode():
            if "Could not open file." in p.stdout.decode():
                raise ValueError(
                    "Your packmol might be too old to handle paths with spaces. "
                    "Please try again with a newer version or use paths without spaces."
                )
            msg = p.stdout.decode().split("ERROR")[-1]
            raise ValueError(f"Packmol failed with return code 0 and stdout: {msg}")

        # Write packmol output to the specified output file
        with open(Path(MD_dir, output_file), mode="w") as out:
            out.write(p.stdout.decode())

    except subprocess.CalledProcessError as exc:
        if exc.returncode != 173:  # Only raise the error if it's not the specific 'STOP 173' case
            raise ValueError(f"Packmol failed with error code {exc.returncode} and stderr: {exc.stderr.decode()}") from exc
        else:
            print("Packmol ended with 'STOP 173', but this error is being ignored.")

    finally:
        # Change back to the original working directory
        os.chdir(current_path)


