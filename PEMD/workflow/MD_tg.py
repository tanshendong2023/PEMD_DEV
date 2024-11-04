#!/usr/bin/env python


"""
This script is used to start the molecular dynamics of polymer.
Author: Shendong Tan
Date: 2024-04-01
"""

from PEMD.simulation import qm, MD
from PEMD.model import build
from PEMD.analysis import prop


model_info = {
    'polymer': {
        'compound': 'PEO',
        'resname': 'MOL',
        'numbers': 20,
    },
    }


if __name__ == '__main__':
    # 1. obtain RESP charge fitting result
    # Generate polymer monomer from smiles
    smiles_resp = build.gen_poly_smiles(model_info, resp=True)

    # Perform first conformation search using xtb for RESP charge fitting
    structures = qm.conformer_search_xtb(model_info, smiles_resp, epsilon=5, core=32, polymer=True, work_dir=None,
                                         max_conformers=1000, top_n_MMFF=100, top_n_xtb=10, )

    # Perform second conformation search using Gaussian for RESP charge fitting
    sorted_df = qm.conformer_search_gaussian(structures, model_info, polymer=True, work_dir=None, charge=0,
                                             multiplicity=1, core = 32, memory= '64GB', chk=True, opt_method='B3LYP',
                                             opt_basis='6-311+g(d,p)', dispersion_corr='em=GD3BJ', freq='freq',
                                             solv_model='scrf=(pcm,solvent=generic,read)', custom_solv='eps=5.0 \nepsinf=2.1', )

    # Perform RESP charge fitting
    qm.calc_resp_gaussian(sorted_df, model_info, epsinf=2.1, polymer=True, work_dir=None, numconf=5, core=32, memory='64GB',
                          eps=5.0, method='resp2', )

    # 2. start MD simulation for sigle-chain polymer
    # Generate polymer chain from smiles
    smiles_MD= build.gen_poly_smiles(model_info, resp=False)

    # Build polymer chain
    build.gen_poly_3D(model_info, smiles_MD, core = 32, )

    # Generate the topology and itp files
    nonbonditp_filename, bonditp_filename = MD.gen_gmx_oplsaa(model_info, )

    # Apply RESP charge to the polymer chain
    qm.apply_chg_topoly(model_info, end_repeating=2, method='resp2', target_sum_chg=0, )

    # 3. start MD simulation for amorphous polymer system
    # Generate the packmol input file
    build.gen_packmol_input(model_info, density=0.8, add_length=25, out_dir='MD_dir', packinp_name='pack.inp',
                            packout_name='pack_cell.pdb', )

    # Run packmol
    build.run_packmol(out_dir='MD_dir', input_file='pack.inp', output_file='pack.out', )

    # Pre-run gromacs
    MD.pre_run_gmx(model_info, density=0.8, add_length=25, out_dir='MD_dir', packout_name='pack_cell.pdb', core=64,
                   T_target=333, top_filename='topol.top', module_soft='GROMACS/2021.7-ompi', output_str='pre_eq')

    # Run gromacs for glass transition temperature
    MD.run_gmx_tg(out_dir='tg_dir', input_str='pre_eq', out_str='npt_anneal_tg', top_filename='topol.top',
                  module_soft='GROMACS/2021.7-ompi', anneal_rate=0.01, core=64, Tinit=600, Tfinal=100, )

    # Post-process for glass transition temperature
    df=prop.dens_temp(out_dir='tg_dir', tpr_file='npt_anneal_tg.tpr', edr_file='npt_anneal_tg.edr',
                      module_soft='GROMACS/2021.7-ompi', initial_time=500, time_gap=4000, duration=1000,
                      temp_initial=600,temp_decrement=20, max_time=102000, summary_file="dens_tem.csv")

    df_tg=prop.fit_tg(df, param_file="fitting_tg.csv")




