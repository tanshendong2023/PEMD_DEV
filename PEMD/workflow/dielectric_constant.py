#!/usr/bin/env python


"""
This script is used to start the molecular dynamics of polymer.
Author: Shendong Tan
Date: 2024-04-22
"""

from PEMD.simulation import qm, MD
from PEMD.model import build

# from PEMD.analysis import prop


# Define the model information for simulation system
model_info = {
    'polymer': {
        'compound': 'PEO',
        'resname': 'MOL',
        'repeating_unit': '[*]CCO[*]',
        'terminal_cap': 'C[*]',
        'length': [10, 50],
        'numbers': 20,
        'scale': 1.0,
        'charge': 0,
    },
    }


if __name__ == '__main__':
    # 1. obtain RESP charge fitting result
    # Generate polymer monomer from smiles
    smiles_resp = build.gen_poly_smiles(model_info, resp=True)

    # Perform first conformation search using xtb for RESP charge fitting
    structures = qm.conformer_search_xtb(model_info, smiles_resp, epsilon=5, core=32, polymer=True, work_dir=None,
                                         max_conformers=1000, top_n_MMFF=50, top_n_xtb=5, )

    # Perform second conformation search using Gaussian for RESP charge fitting
    sorted_df = qm.conformer_search_gaussian(structures, model_info, polymer=True, work_dir=None, core=16,
                                             memory='64GB', function='B3LYP', basis_set='6-311+g(d,p)',
                                             dispersion_corr='em=GD3BJ', )

    # Perform RESP charge fitting
    qm.calc_resp_gaussian(sorted_df, model_info, epsilon=5.0, epsinf=2.1, polymer=True, work_dir=None, numconf=5,
                          core=32, memory='64GB', method='resp', )

    # 2. start MD simulation for sigle-chain polymer
    # Generate polymer chain from smiles
    smiles_MD = build.gen_poly_smiles(model_info, resp=False)

    # Build polymer chain
    build.gen_poly_3D(model_info, smiles_MD, core=32, )

    # Generate the topology and itp files
    MD.gen_gmx_oplsaa(model_info, out_dir='MD_dir')

    # Apply RESP charge to the polymer chain
    qm.apply_chg_topoly(model_info, out_dir='MD_dir', end_repeating=2, method='resp', target_sum_chg=0, )

    # 3. start MD simulation for amorphous polymer system
    # Generate the packmol input file
    build.gen_packmol_input(model_info, density=0.8, add_length=25, out_dir='MD_dir', packinp_name='pack.inp',
                            packout_name='pack_cell.pdb', )

    # Run packmol
    build.run_packmol(out_dir='MD_dir', input_file='pack.inp', output_file='pack.out', )

    # Pre-run gromacs
    MD.pre_run_gmx(model_info, density=0.8, add_length=25, out_dir='MD_dir', packout_name='pack_cell.pdb', core=64,
                   partition='standard', T_target=298, top_filename='topol.top',
                   module_soft='GROMACS/2021.7-ompi', output_str='pre_eq', )





