"""
Polymer model building tools.

Developed by: Tan Shendong
Date: 2024.01.18
"""


import os
import random

from PEMD import io
from rdkit import Chem
from PEMD.model import polymer
from rdkit.Chem import Descriptors


# homopolymer -A-A-A-
def gen_homopolymer_3D(poly_name, smiles, length):
    sequence = ['A'] * length
    return polymer.gen_sequence_copolymer_3D(poly_name, poly_name, smiles, smiles, sequence)

# random copolymer -A-B-A-A-B-B-
def gen_random_copolymer_3D(poly_name_A, poly_name_B, smiles_A, smiles_B, length, frac_A=0.5):
    sequence = ['A' if random.random() < frac_A else 'B' for _ in range(length)]
    return polymer.gen_sequence_copolymer_3D(poly_name_A, poly_name_B, smiles_A, smiles_B, sequence)

# alternating copolymer -A-B-A-B-
def gen_alternating_copolymer_3D(poly_name_A, poly_name_B, smiles_A, smiles_B, length):
    sequence = ['A' if i % 2 == 0 else 'B' for i in range(length)]
    return polymer.gen_sequence_copolymer_3D(poly_name_A, poly_name_B, smiles_A, smiles_B, sequence)

# block copolymer -A-A-A-B-B-B-
def gen_block_copolymer_3D(poly_name_A, poly_name_B, smiles_A, smiles_B, block_sizes,):
    sequence = []
    for i, blk in enumerate(block_sizes):
        mon = 'A' if i % 2 == 0 else 'B'
        sequence += [mon] * blk
    return polymer.gen_sequence_copolymer_3D(poly_name_A, poly_name_B, smiles_A, smiles_B, sequence,)

def mol_to_pdb(work_dir, mol, poly_name, poly_resname, pdb_filename):

    pdb_file = os.path.join(work_dir, pdb_filename)
    Chem.MolToXYZFile(mol, 'mid.xyz', confId=0)
    io.convert_xyz_to_pdb('mid.xyz', pdb_file, poly_name, poly_resname)
    os.remove('mid.xyz')


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


