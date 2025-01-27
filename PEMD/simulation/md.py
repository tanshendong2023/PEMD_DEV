"""
Polymer model MD tools.

Developed by: Tan Shendong
Date: 2024.03.26
"""


import os
from PEMD.model import model_lib
from PEMD.simulation.gromacs import PEMDGROMACS


def relax_poly_chain(
        work_dir,
        name,
        resname,
        gro_file,
        temperature,
        gpu,
):

    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    molecules = []
    molecule = {
        "name": name,
        "number": 1,
        "resname": resname,
    }
    molecules.append(molecule)

    gmx = PEMDGROMACS(
        MD_dir,
        molecules,
        temperature,
        gpu,
    )

    gmx.gen_top_file(
        top_filename = 'topol.top'
    )

    gmx.gen_em_mdp_file(
        filename = 'em.mdp'
    )

    gmx.gen_nvt_mdp_file(
        filename = 'nvt.mdp'
    )

    gmx.commands_pdbtogro(
        gro_file,
    ).run_local()

    gmx.commands_em(
        input_gro = 'conf.gro'
    ).run_local()

    gmx.commands_nvt(
        input_gro = 'em.gro',
        output_str = 'nvt'
    ).run_local()

    gmx.commands_grotopdb(
        gro_filename = 'nvt.gro',
        pdb_filename = f'{name}.pdb',
    ).run_local()

    print(f'The polymer chain has been relaxed and the {name}.pdb were saved successful.')

def anneal_amorph_poly(
        work_dir,
        molecules,
        temperature,
        T_high_increase,
        anneal_rate,
        anneal_npoints,
        packmol_pdb,
        density,
        add_length,
        gpu
):
    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    gmx = PEMDGROMACS(
        MD_dir,
        molecules,
        temperature,
        gpu,
    )

    gmx.gen_top_file(
        top_filename = 'topol.top'
    )

    gmx.gen_em_mdp_file(
        filename = 'em.mdp'
    )

    gmx.gen_nvt_mdp_file(
        filename = 'nvt.mdp'
    )

    gmx.gen_npt_anneal_mdp_file(
        T_high_increase,
        anneal_rate,
        anneal_npoints,
        filename = 'npt_anneal.mdp'
    )

    gmx.gen_npt_mdp_file(
        filename = 'npt_eq.mdp'
    )

    gmx.commands_pdbtogro(
        packmol_pdb,
        density,
        add_length
    ).run_local()

    gmx.commands_em(
        input_gro = 'conf.gro'
    ).run_local()

    gmx.commands_nvt(
        input_gro = 'em.gro',
        output_str = 'nvt'
    ).run_local()

    gmx.commands_npt_anneal(
        input_gro = 'nvt.gro'
    ).run_local()

    gmx.commands_npt(
        input_gro = 'npt_anneal.gro',
        output_str = 'npt_eq'
    ).run_local()

    gmx.commands_extract_volume(
        edr_file = 'npt_eq.edr',
        output_file = 'volume.xvg'
    ).run_local()

    volumes_path = os.path.join(MD_dir, 'volume.xvg')
    volumes = model_lib.read_volume_data(volumes_path)

    (
        average_volume,
        frame_time
     ) = model_lib.analyze_volume(
        volumes,
        start=4000,
        dt_collection=5
    )

    gmx.commands_extract_structure(
        tpr_file = 'npt_eq.tpr',
        xtc_file = 'npt_eq.xtc',
        save_gro_file = 'pre_eq.gro',
        frame_time = frame_time
    ).run_local()

def run_gmx_prod(
        work_dir,
        molecules,
        temperature,
        nstep_ns,
        gpu
):

    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    gmx = PEMDGROMACS(
        MD_dir,
        molecules,
        temperature,
        gpu
    )

    gmx.gen_top_file(
        top_filename = 'topol.top'
    )

    # generation nvt production mdp file, 200ns
    nstep = int(nstep_ns*1000000)   # ns to fs
    gmx.gen_nvt_mdp_file(
        nsteps_nvt = nstep,
        filename = 'nvt_prod.mdp',
    )

    gmx.commands_nvt_product(
        input_gro = 'pre_eq.gro',
        output_str = 'nvt_prod'
    ).run_local()

    gmx.commands_wraptounwrap(
        output_str = 'nvt_prod'
    ).run_local()



































