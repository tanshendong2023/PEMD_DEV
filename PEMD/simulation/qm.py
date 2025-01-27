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

    # Generate multiple conformers
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=max_conformers, randomSeed=20)
    props = AllChem.MMFFGetMoleculeProperties(mol)

    # Minimize the energy of each conformer and store the energy
    minimized_conformers = []
    for conf_id in ids:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        status = ff.Minimize()
        if status != 0:
            print(f"Conformer {conf_id} optimization did not converge. Status code: {status}")
        energy = ff.CalcEnergy()
        minimized_conformers.append((conf_id, energy))

    print(f"Generated {len(minimized_conformers)} conformers for {name}")

    # Sort the conformers by energy and select the top N conformers
    minimized_conformers.sort(key=lambda x: x[1])
    top_conformers = minimized_conformers[:top_n_MMFF]

    # merge the top conformers to a single xyz file
    output_filename = f'{name}_MMFF_top{top_n_MMFF}.xyz'
    output_xyz_filepath = os.path.join(work_dir, output_filename)
    with open(output_xyz_filepath, 'w') as merged_xyz:
        for idx, (conf_id, energy) in enumerate(top_conformers):
            conf = mol.GetConformer(conf_id)
            atoms = mol.GetAtoms()
            num_atoms = mol.GetNumAtoms()
            merged_xyz.write(f"{num_atoms}\n")
            merged_xyz.write(f"Conformer {idx + 1}, Energy: {energy:.4f} kcal/mol\n")
            for atom in atoms:
                pos = conf.GetAtomPosition(atom.GetIdx())
                element = atom.GetSymbol()
                merged_xyz.write(f"{element} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")

    print(f"Top {top_n_MMFF} conformers were saved to {output_filename}")

    return output_filename

def opt_conf_xtb(
        work_dir,
        xyz_filename,
        name,
        top_n_xtb,
        chg,
        mult,
        gfn,
        optimize=True
):

    xtb_dir = os.path.join(work_dir, f'XTB_{name}')
    os.makedirs(xtb_dir, exist_ok=True)

    xyz_filepath = os.path.join(work_dir, xyz_filename)
    structures = sim_lib.read_xyz_file(xyz_filepath)  # 读取XYZ文件，返回结构列表

    energy_list = []

    for idx, structure in enumerate(structures):
        comment = structure['comment']
        atoms = structure['atoms']
        conf_xyz_file = os.path.join(xtb_dir, f'conf_{idx}.xyz')
        with open(conf_xyz_file, 'w') as f:
            f.write(f"{structure['num_atoms']}\n")
            f.write(f"{comment}\n")
            for atom in atoms:
                f.write(f"{atom}\n")

        outfile_headname = f'conf_{idx}'

        xtb_calculator = PEMDXtb(
            work_dir=xtb_dir,
            chg=chg,
            mult=mult,
            gfn=gfn
        )
        result = xtb_calculator.run_local(
            xyz_filename=conf_xyz_file,
            outfile_headname=outfile_headname,
            optimize=optimize
        )

        if not optimize:
            if isinstance(result, dict):
                energy_info = result.get('energy_info')
                if energy_info and 'total_energy' in energy_info:
                    energy = energy_info['total_energy']
                    energy_list.append({
                        'idx': idx,
                        'energy': energy,
                        'filename': f'conf_{idx}.xyz'
                    })
                else:
                    print(f"结构 conf_{idx}.xyz 能量提取失败。")
            else:
                print(f"结构 conf_{idx}.xyz 计算失败。")
        else:
            energy_info = result.get('energy_info')
            if energy_info and 'total_energy' in energy_info:
                energy = energy_info['total_energy']
                energy_list.append({
                    'idx': idx,
                    'energy': energy,
                    'filename': f'conf_{idx}.xtbopt.xyz'
                })
            else:
                print(f"结构 conf_{idx}.xyz 优化后的能量提取失败。")

    print("XTB run locally successfully!")

    if not energy_list:
        print("未成功提取任何能量值。")
        return None

    sorted_energies = sorted(energy_list, key=lambda x: x['energy'])
    top_structures = sorted_energies[:top_n_xtb]

    output_file = os.path.join(work_dir, f'{name}_xtb_top{top_n_xtb}.xyz')
    with open(output_file, 'w') as outfile:
        for struct in top_structures:
            struct_file = os.path.join(xtb_dir, struct['filename'])
            if os.path.isfile(struct_file):
                with open(struct_file, 'r') as infile:
                    outfile.write(infile.read())
            else:
                print(f"选中的结构文件 {struct_file} 不存在。")

    print(f"Top {top_n_xtb} conformers were saved to {output_file}")

    return output_file


# input: a xyz file
# output: a xyz file
# description:
def opt_conf_gaussian(
        work_dir,
        name,
        xyz_filename,
        top_n_qm,
        chg=0,
        mult=1,
        function='B3LYP',
        basis_set='6-31+g(d,p)',
        epsilon=5.0,
        core=64,
        mem='128GB',
        multi_step=False,
        max_attempts=1
):

    conf_dir = os.path.join(work_dir, f'QM_{name}')
    os.makedirs(conf_dir, exist_ok=True)

    xyz_filepath = os.path.join(work_dir, xyz_filename)
    structures = sim_lib.read_xyz_file(xyz_filepath)

    for idx, structure in enumerate(structures):

        filename = f'conf_{idx}.gjf'
        Gau = PEMDGaussian(
            work_dir=conf_dir,
            filename=filename,
            core=core,
            mem=mem,
            chg=chg,
            mult=mult,
            function=function,
            basis_set=basis_set,
            epsilon=epsilon,
            multi_step=multi_step,  # 根据 gaucontinue 设置 multi_step
            max_attempts=max_attempts,          # 可根据需要调整最大尝试次数
        )

        state, log_filename = Gau.run_local(
            structure=structure,
            resp=False,
            chk=False,
        )

        if state == 'success':
            Gau.logger.info(f"Optimization succeeded for {filename}.")
            print(f"Structure {idx}: Optimization SUCCESS.")
        else:
            Gau.logger.error(f"Optimization failed for {filename}.")
            print(f"Structure {idx}: Optimization FAILED.")

    output_file = f"{name}_gaussian_top{top_n_qm}.xyz"
    sim_lib.order_energy_gaussian(
        conf_dir,
        top_n_qm,
        output_file,
    )
    return output_file

def qm_gaussian(
        work_dir,
        xyz_filename,
        gjf_filename,
        chg=0,
        mult=1,
        function='B3LYP',
        basis_set='6-31+g(d,p)',
        epsilon=5.0,
        core=64,
        mem='128GB',
        chk=False,
        multi_step=True,
        max_attempts=2,
):
    os.makedirs(work_dir, exist_ok=True)

    xyz_filepath = os.path.join(work_dir, xyz_filename)
    structures = sim_lib.read_xyz_file(xyz_filepath)

    for idx, structure in enumerate(structures):

        filename = f'{gjf_filename}_{idx}.gjf'
        Gau = PEMDGaussian(
            work_dir=work_dir,
            filename=filename,
            core=core,
            mem=mem,
            chg=chg,
            mult=mult,
            function=function,
            basis_set=basis_set,
            epsilon=epsilon,
            multi_step=multi_step,
            max_attempts=max_attempts,
        )

        state, log_filename = Gau.run_local(
            structure=structure,
            resp=False,
            chk=chk,
        )

        if state == 'success':
            Gau.logger.info(f"Optimization succeeded for {filename}.")
            print(f"Structure {idx}: Optimization SUCCESS.")
        else:
            Gau.logger.error(f"Optimization failed for {filename}.")
            print(f"Structure {idx}: Optimization FAILED.")


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
            work_dir=resp_dir,
            filename=filename,
            core=core,
            mem=mem,
            chg=chg,
            mult=mult,
            function=function,
            basis_set=basis_set,
            epsilon=epsilon,
            multi_step=False,  # RESP 计算不启用多步计算
            max_attempts=1,  # 仅尝试一次
        )

        # Gau.generate_input_file_resp(
        #     structure=structure,
        # )

        state, log_filename = Gau.run_local(
            structure=structure,
            resp=True,
            chk=False,
        )

        if state == 'failed':
            Gau.logger.error(f"RESP calculation failed for {filename}.")

# def RESP_fit_Multiwfn(
#         work_dir,
#         name,
#         smiles,
#         method = "resp2",
#         delta=0.5
# ):
#
#     # Build the resp_dir.
#     resp_dir = os.path.join(work_dir, f'resp_dir_{name}')
#     os.makedirs(resp_dir, exist_ok=True)
#
#     # Fina chk files, convert them to fchk files.
#     chk_files = glob.glob(os.path.join(resp_dir, 'SP*.chk'))
#     for chk_file in chk_files:
#         model_lib.convert_chk_to_fchk(chk_file)
#
#     # Calculation RESP charges using Multiwfn.
#     PEMDMultiwfn(resp_dir).resp_run_local(method)
#
#     # Read charges data of solvation state.
#     solv_chg_df = pd.DataFrame()
#     solv_chg_files = glob.glob(os.path.join(resp_dir, 'SP_solv_conf*.chg'))
#     # Calculate average charges of solvation state.
#     for file in solv_chg_files:
#         data = pd.read_csv(file, sep=r'\s+', names=['atom', 'X', 'Y', 'Z', 'charge'])
#         data['position'] = data.index
#         solv_chg_df = pd.concat([solv_chg_df, data], ignore_index=True)
#     average_charges_solv = solv_chg_df.groupby('position')['charge'].mean().reset_index()
#
#     # If using RESP2 method, calculate weighted charge of both solvation and gas states.
#     if method == 'resp2':
#         # Read charges data of gas state.
#         gas_chg_df = pd.DataFrame()
#         gas_chg_files = glob.glob(os.path.join(resp_dir, 'SP_gas_conf*.chg'))
#         # Calculate average charges of gas state.
#         for file in gas_chg_files:
#             data = pd.read_csv(file, sep=r'\s+', names=['atom', 'X', 'Y', 'Z', 'charge'])
#             data['position'] = data.index
#             gas_chg_df = pd.concat([gas_chg_df, data], ignore_index=True)
#         average_charges_gas = gas_chg_df.groupby('position')['charge'].mean().reset_index()
#         # Combine the average charges of solvation and gas states, calculated by weight.
#         average_charges = average_charges_solv.copy()
#         average_charges['charge'] = average_charges_solv['charge'] * delta + average_charges_gas['charge'] * (1 - delta)
#     else:
#         # If using RESP method, just calculate average charges of solvation state.
#         average_charges = average_charges_solv
#
#     # Extract atomic types and add to the results.
#     reference_file = solv_chg_files[0]
#     ref_data = pd.read_csv(reference_file, sep=r'\s+', names=['atom', 'X', 'Y', 'Z', 'charge'])
#     atom_types = ref_data['atom']
#     average_charges['atom'] = atom_types.values
#     average_charges = average_charges[['atom', 'charge']]
#
#     # Save to csv file.
#     csv_filepath = os.path.join(resp_dir, f'{method}_average_chg.csv')
#     average_charges.to_csv(csv_filepath, index=False)
#
#     return average_charges

def RESP_fit_Multiwfn(
        work_dir,
        name,
        method="resp2",
        delta=0.5
):
    # Build the resp_dir.
    resp_dir = os.path.join(work_dir, f'resp_dir_{name}')
    os.makedirs(resp_dir, exist_ok=True)

    # Find chk files and convert them to fchk files.
    chk_files = glob.glob(os.path.join(resp_dir, 'SP*.chk'))
    for chk_file in chk_files:
        model_lib.convert_chk_to_fchk(chk_file)

    # Calculate RESP charges using Multiwfn.
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

    # Extract atomic types and retain the position for mapping.
    reference_file = solv_chg_files[0]
    ref_data = pd.read_csv(reference_file, sep=r'\s+', names=['atom', 'X', 'Y', 'Z', 'charge'])
    atom_types = ref_data['atom']
    average_charges['atom'] = atom_types.values
    # Retain 'position' to map charges to atoms
    average_charges = average_charges[['position', 'atom', 'charge']]

    # Save to csv file.
    csv_filepath = os.path.join(resp_dir, f'{method}_average_chg.csv')
    average_charges.to_csv(csv_filepath, index=False)

    return average_charges







