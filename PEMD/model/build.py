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

def gen_poly_3D(work_dir, poly_name, length, smiles):
    mol = pybel.readstring("smi", smiles)
    mol.addh()
    mol.make3D()
    obmol = mol.OBMol
    DEG_TO_RAD = math.pi / 180
    num_iterations = 10
    angle_range = (0, 360)

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

    mol.localopt()

    pdb_file = os.path.join(work_dir, f"{poly_name}_N{length}.pdb")
    mol.write("pdb", pdb_file, overwrite=True)

    return f"{poly_name}_N{length}.pdb"

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


