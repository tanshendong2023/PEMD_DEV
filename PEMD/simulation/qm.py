# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# simulation.qm module
# ******************************************************************************

import os
import glob
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from PEMD.simulation import sim_lib
from PEMD.model import model_lib
from PEMD.simulation.xtb import PEMDXtb
from PEMD.simulation.gaussian import PEMDGaussian
from PEMD.simulation.multiwfn import PEMDMultiwfn


# Input: smiles (str)
# Output: a xyz file
# Description: Generates multiple conformers for a molecule from a SMILES string, optimizes them using the MMFF94
# force field, and saves the optimized conformers to a single XYZ file.
def gen_conf_rdkit(
        work_dir,
        name,
        smiles,
        max_conformers,
        top_n_MMFF
):

    # build dir
    conf_dir = os.path.join(work_dir, f'conformer_search_{name}')
    os.makedirs(conf_dir, exist_ok=True)

    # Generate multiple conformers
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=max_conformers, randomSeed=20)
    props = AllChem.MMFFGetMoleculeProperties(mol)

    # Minimize the energy of each conformer and store the energy
    minimized_conformers = []
    for conf_id in ids:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        energy = ff.Minimize()
        minimized_conformers.append((conf_id, energy))

    # Sort the conformers by energy and select the top N conformers
    minimized_conformers.sort(key=lambda x: x[1])
    top_conformers = minimized_conformers[:top_n_MMFF]

    # save the top conformers to xyz files
    for conf_id, (original_id, _) in enumerate(top_conformers):
        xyz_file = os.path.join(conf_dir, f'conf_{conf_id}.xyz')
        model_lib.mol_to_xyz(mol, conf_id, xyz_file)


# input: a xyz file
# output:  a xyz file
# Description: Sorts the conformers in a XYZ file by energy calculated by xTB and saves the sorted conformers to a
# new file
def opt_conf_xtb(
        work_dir,
        name,
        top_n_MMFF,
        top_n_xtb,
        chg=0,
        mult=1,
        gfn=2,
):

    # build dir
    conf_dir = os.path.join(work_dir, f'conformer_search_{name}')
    os.makedirs(conf_dir, exist_ok=True)

    # Perform xTB optimization on the selected conformers
    for conf_id in range(top_n_MMFF):
        conf_xyz_file = os.path.join(conf_dir, f'conf_{conf_id}.xyz')
        outfile_headname = f'conf_{conf_id}'

        # Create xtb object
        PEMDXtb(
            conf_dir,
            chg,
            mult,
            gfn
        ).run_local(
            conf_xyz_file,
            outfile_headname
        )

    print("XTB run locally successfully!")
    filenames = glob.glob(os.path.join(conf_dir, '*.xtbopt.xyz'))
    merged_file = os.path.join(conf_dir, 'merged.xyz')
    with open(merged_file, 'w') as outfile:
        for fname in filenames:
            with open(fname, 'r') as infile:
                outfile.write(infile.read())

    output_file = f'{name}_xtb_top{top_n_xtb}.xyz'
    sim_lib.order_energy_xtb(
        work_dir,
        merged_file,
        top_n_xtb,
        output_file
    )
    return output_file


# input: a xyz file
# output: a xyz file
# description:
def opt_conf_gaussian(
        work_dir,
        name,
        xyz_file,
        top_n_qm,
        chg=0,
        mult=1,
        function='B3LYP',
        basis_set='6-311+g(d,p)',
        epsilon=5.0,
        core=64,
        mem='64GB',
):

    conf_dir = os.path.join(work_dir, f'conformer_search_{name}')
    os.makedirs(conf_dir, exist_ok=True)

    structures = sim_lib.read_xyz_file(xyz_file)

    for i, structure in enumerate(structures):

        filename = f'conf_{i}.gjf'
        Gau = PEMDGaussian(
            conf_dir,
            filename,
            core,
            mem,
            chg,
            mult,
            function,
            basis_set,
            epsilon,
        )

        Gau.generate_input_file(
            structure,
        )

        Gau.run_local()

    output_file = f"{name}_gaussian_top{top_n_qm}.xyz"
    sim_lib.order_energy_gaussian(
        conf_dir,
        top_n_qm,
        output_file,
    )
    return output_file

def calc_resp_gaussian(
        work_dir,
        name,
        xyz_file,
        chg=0,
        mult=1,
        function='B3LYP',
        basis_set='6-311+g(d,p)',
        epsilon=5.0,
        core=32,
        mem='64GB',
):
    # Build the resp_dir.
    resp_dir = os.path.join(work_dir, f'resp_dir_{name}')
    os.makedirs(resp_dir, exist_ok=True)

    # Read xyz file as a list of structures.
    structures = sim_lib.read_xyz_file(xyz_file)

    # Generate Gaussian input files of selected conformers.
    for idx, structure in enumerate(structures):
        filename = f"conf_{idx}.gjf"
        Gau = PEMDGaussian(
            resp_dir,
            filename,
            core,
            mem,
            chg,
            mult,
            function,
            basis_set,
            epsilon,
        )
        Gau.generate_input_file_resp(
            structure,
        )

        Gau.run_local()


def RESP_fit_Multiwfn(
        work_dir,
        name,
        method = "resp2",
        delta=0.5
):

    # Build the resp_dir.
    resp_dir = os.path.join(work_dir, f'resp_dir_{name}')
    os.makedirs(resp_dir, exist_ok=True)

    # Fina chk files, convert them to fchk files.
    chk_files = glob.glob(os.path.join(resp_dir, 'SP*.chk'))
    for chk_file in chk_files:
        model_lib.convert_chk_to_fchk(chk_file)

    # Calculation RESP charges using Multiwfn.
    PEMDMultiwfn(resp_dir).resp_run_local(method)

    # Read charges data of solvation state.
    solv_chg_df = pd.DataFrame()
    solv_chg_files = glob.glob(os.path.join(resp_dir, 'SP_solv_conf*.chg'))
    # Calculate average charges of solvation state.
    for file in solv_chg_files:
        data = pd.read_csv(file, sep=r'\s+', names=['atom', 'X', 'Y', 'Z', 'charge'])
        data['position'] = data.index
        solv_chg_df = pd.concat([solv_chg_df, data], ignore_index=True)
    average_charges_solv = solv_chg_df.groupby('position')['charge'].mean().reset_index()

    # If using RESP2 method, calculate weighted charge of both solvation and gas states.
    if method == 'resp2':
        # Read charges data of gas state.
        gas_chg_df = pd.DataFrame()
        gas_chg_files = glob.glob(os.path.join(resp_dir, 'SP_gas_conf*.chg'))
        # Calculate average charges of gas state.
        for file in gas_chg_files:
            data = pd.read_csv(file, sep=r'\s+', names=['atom', 'X', 'Y', 'Z', 'charge'])
            data['position'] = data.index
            gas_chg_df = pd.concat([gas_chg_df, data], ignore_index=True)
        average_charges_gas = gas_chg_df.groupby('position')['charge'].mean().reset_index()
        # Combine the average charges of solvation and gas states, calculated by weight.
        average_charges = average_charges_solv.copy()
        average_charges['charge'] = average_charges_solv['charge'] * delta + average_charges_gas['charge'] * (1 - delta)
    else:
        # If using RESP method, just calculate average charges of solvation state.
        average_charges = average_charges_solv

    # Extract atomic types and add to the results.
    reference_file = solv_chg_files[0]
    ref_data = pd.read_csv(reference_file, sep=r'\s+', names=['atom', 'X', 'Y', 'Z', 'charge'])
    atom_types = ref_data['atom']
    average_charges['atom'] = atom_types.values
    average_charges = average_charges[['atom', 'charge']]

    # Save to csv file.
    csv_filepath = os.path.join(resp_dir, f'{method}_average_chg.csv')
    average_charges.to_csv(csv_filepath, index=False)

    return average_charges








