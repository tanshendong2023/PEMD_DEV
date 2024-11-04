#!/usr/bin/env python


"""
This script is used to generate the Fluorinated polymer monomer and perform QM calculations,
including the HOMO, LUMO energy and the dipole_moment.
Author: Shendong Tan
Date: 2024-03-30
"""


from PEMD.model import poly, PEMD_lib
from PEMD.simulation import qm
from PEMD.analysis import prop


unit_name = 'F-PEO'
repeating_unit = '[*]CCO[*]'
leftcap = 'C[*]'
rightcap = 'C[*]'
length = 1


if __name__ == '__main__':
    # Generate Fluorinated polymer monomer from smiles
    smiles_poly_list, mol_poly_list = poly.F_poly_gen(unit_name, repeating_unit, leftcap, rightcap, length, )

    # Loop through polymer list
    for i, mol in enumerate(mol_poly_list):

        F_num = PEMD_lib.count_atoms(mol, 'F', length)
        out_dir = f"{F_num}{unit_name}_N{length}_{i + 1}"

        # Perform first conformation search using CREST
        structures = qm.unit_conformer_search_crest(mol, unit_name, out_dir, length, numconf=10, core=32, )

        # Perform second conformation search using Gaussian
        sorted_df = qm.conformer_search_gaussian(out_dir, structures, unit_name, charge=0, multiplicity=1, core=32,
                                                 memory='64GB', chk=True, opt_method='B3LYP', opt_basis='6-311+g(d,p)',
                                                 dispersion_corr='em=GD3BJ', freq='freq',
                                                 solv_model='scrf=(pcm,solvent=generic,read)', custom_solv='', )

        # obtain the HOMO and LUMO property
        prop.homo_lumo_energy(sorted_df, unit_name, out_dir, length)

        # obtain the dipole_moment property
        prop.dipole_moment(sorted_df, unit_name, out_dir, length)




