"""Polymer model building tools.

The :func:`gen_copolymer_3D` helper offers a unified interface to build
homopolymers and various copolymers. Legacy functions remain as thin wrappers
for backward compatibility.
"""


import os
import random

from PEMD import io
from rdkit import Chem
from PEMD.model import polymer
from rdkit.Chem import Descriptors


def gen_copolymer_3D(poly_name_A,
                     poly_name_B,
                     smiles_A,
                     smiles_B,
                     *,
                     mode: str | None = None,
                     length: int | None = None,
                     frac_A: float = 0.5,
                     block_sizes: list[int] | None = None,
                     sequence: list[str] | None = None):
    """Generate a 3D copolymer model.

    Parameters
    ----------
    poly_name_A, poly_name_B : str
        Names of the two monomer units.
    smiles_A, smiles_B : str
        SMILES of the two monomer units.
    mode : {"homopolymer", "random", "alternating", "block"}, optional
        Sequence generation mode. Ignored if ``sequence`` is provided.
    length : int, optional
        Polymer length for ``homopolymer``, ``random`` and ``alternating``.
    frac_A : float, default 0.5
        Fraction of monomer A for ``random`` mode.
    block_sizes : list[int], optional
        Sizes of each block for ``block`` mode.
    sequence : list[str], optional
        Explicit sequence composed of 'A' and 'B'.
    """

    if sequence is None:
        if mode == "homopolymer":
            if length is None:
                raise ValueError("length is required for homopolymer mode")
            sequence = ['A'] * length
        elif mode == "random":
            if length is None:
                raise ValueError("length is required for random mode")
            sequence = [
                'A' if random.random() < frac_A else 'B'
                for _ in range(length)
            ]
        elif mode == "alternating":
            if length is None:
                raise ValueError("length is required for alternating mode")
            sequence = ['A' if i % 2 == 0 else 'B' for i in range(length)]
        elif mode == "block":
            if not block_sizes:
                raise ValueError("block_sizes is required for block mode")
            sequence = []
            for i, blk in enumerate(block_sizes):
                mon = 'A' if i % 2 == 0 else 'B'
                sequence += [mon] * blk
        else:
            raise ValueError("mode must be provided when sequence is None")

    return polymer.gen_sequence_copolymer_3D(
        poly_name_A,
        poly_name_B,
        smiles_A,
        smiles_B,
        sequence,
    )


# homopolymer -A-A-A-
def gen_homopolymer_3D(poly_name, smiles, length):
    """Deprecated wrapper for :func:`gen_copolymer_3D`."""
    return gen_copolymer_3D(
        poly_name,
        poly_name,
        smiles,
        smiles,
        mode="homopolymer",
        length=length,
    )

# random copolymer -A-B-A-A-B-B-
def gen_random_copolymer_3D(
    poly_name_A,
    poly_name_B,
    smiles_A,
    smiles_B,
    length,
    frac_A=0.5,
):
    """Deprecated wrapper for :func:`gen_copolymer_3D`."""
    return gen_copolymer_3D(
        poly_name_A,
        poly_name_B,
        smiles_A,
        smiles_B,
        mode="random",
        length=length,
        frac_A=frac_A,
    )

# alternating copolymer -A-B-A-B-
def gen_alternating_copolymer_3D(
    poly_name_A,
    poly_name_B,
    smiles_A,
    smiles_B,
    length,
):
    """Deprecated wrapper for :func:`gen_copolymer_3D`."""
    return gen_copolymer_3D(
        poly_name_A,
        poly_name_B,
        smiles_A,
        smiles_B,
        mode="alternating",
        length=length,
    )

# block copolymer -A-A-A-B-B-B-
def gen_block_copolymer_3D(
    poly_name_A,
    poly_name_B,
    smiles_A,
    smiles_B,
    block_sizes,
):
    """Deprecated wrapper for :func:`gen_copolymer_3D`."""
    return gen_copolymer_3D(
        poly_name_A,
        poly_name_B,
        smiles_A,
        smiles_B,
        mode="block",
        block_sizes=block_sizes,
    )

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


