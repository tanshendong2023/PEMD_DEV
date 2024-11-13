# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# simulation.qm module
# ******************************************************************************

import os
import time
import glob
import subprocess
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from simple_slurm import Slurm
from PEMD.simulation import sim_lib
from PEMD.model import model_lib
from importlib import resources

from PEMD.simulation.multiwfn import PEMDMultiwfn
from PEMD.simulation.xtb import PEMDXtb
from PEMD.simulation.gaussian import PEMDGaussian
from PEMD.simulation.slurm import PEMDSlurm
from PEMD.simulation.sim_lib import build_directory

def gen_conf_rdkit(
        work_dir,
        smiles,
        max_conformers,
        top_n_MMFF
):
    """
    Generates multiple conformers for a molecule from a SMILES string, optimizes them using the MMFF94
    force field, and saves the optimized conformers to a single XYZ file.
    """
    # Build dir "./conformer_search/rdkit_work"
    rdkit_dir = os.path.join(work_dir, 'conformer_search', 'rdkit_work')
    # If the folder exists, clear its contents
    build_directory(rdkit_dir)

    # Generate max_conformers random conformers form SMILES structure
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=max_conformers, randomSeed=1)
    props = AllChem.MMFFGetMoleculeProperties(mol)

    # Minimize the energy of each conformer and store the energy
    minimized_conformers = []
    for conf_id in ids:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        energy = ff.Minimize()
        minimized_conformers.append((conf_id, energy))

    # Sort the conformers by energy and select the top_n_MMFF conformers with the lowest energy
    minimized_conformers.sort(key=lambda x: x[1])
    top_conformers = minimized_conformers[:top_n_MMFF]

    # Save the coordinates of top_n_MMFF conformers to xyz files
    for conf_id, _ in top_conformers:
        xyz_file = os.path.join(rdkit_dir, f'conf_{conf_id}.xyz')
        model_lib.mol_to_xyz(mol, conf_id, xyz_file)

    # Save all the coordinates to merged_file
    filenames = glob.glob(os.path.join(rdkit_dir, '*.xyz'))
    merged_file = os.path.join(work_dir, 'merged_rdkit.xyz')
    with open(merged_file, 'w') as outfile:
        for fname in filenames:
            with open(fname, 'r') as infile:
                outfile.write(infile.read())
    print(f"The generated conformers based on rdkit were saved to {merged_file}")

def opt_conf_xtb(
        work_dir,
        xyz_file,
        chg=0,
        mult=1,
        gfn=2,
        slurm=False,
        job_name='xtb',
        nodes=1,
        ntasks_per_node=64,
        partition='standard',
):
    # Perform xtb energy optimization on the conformational structure obtained by rdkit calculation.
    # Build xtb_dir.
    xtb_dir = os.path.join(work_dir, 'conformer_search', 'xtb_work')
    build_directory(xtb_dir)

    # Read xyz file as a list of structures.
    structures = sim_lib.read_xyz_file(xyz_file)
    # Perform xTB optimization on the selected conformers.
    for conf_id, structure in enumerate(structures):
        conf_xyz_file = os.path.join(xtb_dir, f'conf_{conf_id}.xyz')
        outfile_headname = f'conf_{conf_id}'
        with open(conf_xyz_file, 'w') as file:
            num_atoms = structure['num_atoms']
            comment = structure['comment']
            atoms = structure['atoms']
            file.write(f"{num_atoms}\n{comment}\n")
            for atom in atoms:
                file.write(f"{atom}\n")

        # Run xTB locally.
        if not slurm:
            PEMDXtb(
                work_dir,
                chg,
                mult,
                gfn
            ).run_local(
                conf_xyz_file,
                xtb_dir,
                outfile_headname
            )

    if not slurm:
        # Integrate the optimized structure and energy results of xTB calculation into one xyz file.
        filenames = glob.glob(os.path.join(xtb_dir, '*.xtbopt.xyz'))
        merged_file = os.path.join(work_dir, 'merged_xtb.xyz')
        with open(merged_file, 'w') as outfile:
            for fname in filenames:
                with open(fname, 'r') as infile:
                    outfile.write(infile.read())
        print(f"XTB run locally successfully!\nThe optimization conformers based on xtb were saved to {merged_file}")
    else:
        # Generate XTB submit script.
        script_name = f'sub.xtb'
        script_path = PEMDXtb(
            work_dir,
            chg,
            mult,
            gfn
        ).gen_slurm(
            script_name,
            job_name,
            nodes,
            ntasks_per_node,
            partition
        )
        print(f"XTB submit script generated successfully!\nThe submit script was saved to {script_path}/sub.xtb")

def opt_conf_gaussian(
        work_dir,
        xyz_file,
        chg=0,
        mult=1,
        function='B3LYP',
        basis_set='6-311+g(d,p)',
        epsilon=5.0,
        mem='64GB',
        job_name='g16',
        nodes=1,
        ntasks_per_node=64,
        partition='standard',
):
    # Build the gasussian_dir.
    gaussian_dir = os.path.join(work_dir, 'conformer_search', 'gaussian_work')
    build_directory(gaussian_dir)

    # Read xyz file as a list of structures.
    structures = sim_lib.read_xyz_file(xyz_file)
    # Generate Gaussian input files of selected conformers.
    for i, structure in enumerate(structures):
        filename = f"conf_{i}.gjf"
        Gau = PEMDGaussian(
            work_dir,
            ntasks_per_node,
            mem,
            chg,
            mult,
            function,
            basis_set,
            epsilon,
        )
        Gau.generate_input_file(
            gaussian_dir,
            structure,
            filename=filename
        )

    # Generate Gaussian submit script.
    script_name = f'sub.gaussian'
    script_path = Gau.gen_slurm(
        script_name,
        job_name,
        nodes,
        ntasks_per_node,
        partition
    )
    print(f"Gaussian submit script generated successfully!\nThe submit script was saved to {script_path}/sub.gaussian")

def calc_resp_gaussian(
        work_dir,
        xyz_file,
        chg=0,
        mult=1,
        function='B3LYP',
        basis_set='6-311+g(d,p)',
        epsilon=5.0,
        mem='64GB',
        job_name='g16',
        nodes=1,
        ntasks_per_node=32,
        partition='standard',
):
    # Build the resp_dir.
    resp_dir = os.path.join(work_dir, 'resp_charge_fitting')
    build_directory(resp_dir)

    # Read xyz file as a list of structures.
    structures = sim_lib.read_xyz_file(xyz_file)

    # Generate Gaussian input files of selected conformers.
    for idx, structure in enumerate(structures):
        filename = f"conf_{idx}.gjf"
        Gau = PEMDGaussian(
            work_dir,
            ntasks_per_node,
            mem,
            chg,
            mult,
            function,
            basis_set,
            epsilon,
        )
        Gau.generate_input_file_resp(
            resp_dir,
            structure,
            filename = filename,
            idx = idx,
        )

    # Generate Gaussian submit script.
    script_name = f'sub.gaussian_resp'
    script_path = Gau.gen_slurm(
        script_name,
        job_name,
        nodes,
        ntasks_per_node,
        partition,
    )
    print(
        f"Gaussian submit script generated successfully!\nThe submit script was saved to {script_path}/sub.gaussian_resp")

def RESP_fit_Multiwfn(resp_dir, method = "resp2"):
    # Fina chk files, convert them to fchk files.
    chk_files = glob.glob(os.path.join(resp_dir, 'SP*.chk'))
    for chk_file in chk_files:
        model_lib.convert_chk_to_fchk(chk_file)

    # Calculation RESP charges using Multiwfn.
    PEMDMultiwfn(resp_dir).resp_run_local(method)

def calculate_average_charges(resp_dir, method='resp2', delta=0.5):
    # Read charges data of solvation state.
    solv_chg_df = pd.DataFrame()
    solv_chg_files = glob.glob(os.path.join(resp_dir, 'SP_solv_conf*.chg'))
    # Calculate average charges of solvation state.
    for file in solv_chg_files:
        data = pd.read_csv(file, delim_whitespace=True, names=['Atom', 'X', 'Y', 'Z', 'Charge'])
        data['Position'] = data.index
        solv_chg_df = pd.concat([solv_chg_df, data], ignore_index=True)
    average_charges_solv = solv_chg_df.groupby('Position')['Charge'].mean().reset_index()

    # If using RESP2 method, calculate weighted charge of both solvation and gas states.
    if method == 'resp2':
        # Read charges data of gas state.
        gas_chg_df = pd.DataFrame()
        gas_chg_files = glob.glob(os.path.join(resp_dir, 'SP_gas_conf*.chg'))
        # Calculate average charges of gas state.
        for file in gas_chg_files:
            data = pd.read_csv(file, delim_whitespace=True, names=['Atom', 'X', 'Y', 'Z', 'Charge'])
            data['Position'] = data.index
            gas_chg_df = pd.concat([gas_chg_df, data], ignore_index=True)
        average_charges_gas = gas_chg_df.groupby('Position')['Charge'].mean().reset_index()
        # Combine the average charges of solvation and gas states, calculated by weight.
        average_charges = average_charges_solv.copy()
        average_charges['Charge'] = average_charges_solv['Charge'] * delta + average_charges_gas['Charge'] * (1 - delta)
    else:
        # If using RESP method, just calculate average charges of solvation state.
        average_charges = average_charges_solv

    # Extract atomic types and add to the results.
    reference_file = solv_chg_files[0]
    ref_data = pd.read_csv(reference_file, delim_whitespace=True, names=['Atom', 'X', 'Y', 'Z', 'Charge'])
    atom_types = ref_data['Atom']
    average_charges['Atom'] = atom_types.values
    average_charges = average_charges[['Atom', 'Charge']]

    # Save to csv file.
    csv_filepath = os.path.join(resp_dir, f'{method}_average_chg.csv')
    average_charges.to_csv(csv_filepath, index=False)

    return average_charges

def apply_chg_topoly(model_info, out_dir, end_repeating=2, method='resp2', target_sum_chg=0):

    current_path = os.getcwd()
    unit_name = model_info['polymer']['compound']
    length_resp = model_info['polymer']['length'][0]
    length_MD = model_info['polymer']['length'][1]
    out_dir_resp = os.path.join(current_path, f'{unit_name}_N{length_resp}')
    out_dir_MD = os.path.join(current_path, f'{unit_name}_N{length_MD}')

    # read resp fitting result from csv file
    resp_chg_file = os.path.join(out_dir_resp, 'resp_work', f'{method}_chg.csv')
    resp_chg_df = pd.read_csv(resp_chg_file)

    repeating_unit = model_info['polymer']['repeating_unit']

    (top_N_noH_df, tail_N_noH_df, mid_ave_chg_noH_df, top_N_H_df, tail_N_H_df, mid_ave_chg_H_df) = (
        model_lib.ave_chg_to_df(resp_chg_df, repeating_unit, end_repeating,))

    # read the xyz file
    relax_polymer_lmp_dir = os.path.join(out_dir_MD, 'relax_polymer_lmp')
    xyz_file_path = os.path.join(relax_polymer_lmp_dir, f'{unit_name}_N{length_MD}_gmx.xyz')
    atoms_chg_df = model_lib.xyz_to_df(xyz_file_path)

    # deal with the mid non-H atoms
    atoms_chg_noH_df = atoms_chg_df[atoms_chg_df['atom'] != 'H']

    cleaned_smiles = repeating_unit.replace('[*]', '')
    molecule = Chem.MolFromSmiles(cleaned_smiles)
    atom_count = molecule.GetNumAtoms()
    N = atom_count * end_repeating + 1

    mid_atoms_chg_noH_df = atoms_chg_noH_df.drop(
        atoms_chg_noH_df.head(N).index.union(atoms_chg_noH_df.tail(N).index)).reset_index(drop=True)

    # traverse the DataFrame of mid-atoms
    for idx, row in mid_atoms_chg_noH_df.iterrows():
        # calculate the position of the current atom in the repeating unit
        position_in_cycle = idx % atom_count
        # find the average charge value of the atom at the corresponding position
        ave_chg_noH = mid_ave_chg_noH_df.iloc[position_in_cycle]['charge']
        # update the charge value
        mid_atoms_chg_noH_df.at[idx, 'charge'] = ave_chg_noH

    # deal with the mid H atoms
    atoms_chg_H_df = atoms_chg_df[atoms_chg_df['atom'] == 'H']

    molecule_with_h = Chem.AddHs(molecule)
    num_H_repeating = molecule_with_h.GetNumAtoms() - molecule.GetNumAtoms() - 2
    N_H = num_H_repeating * end_repeating + 3

    mid_atoms_chg_H_df = atoms_chg_H_df.drop(
        atoms_chg_H_df.head(N_H).index.union(atoms_chg_H_df.tail(N_H).index)).reset_index(drop=True)

    # traverse the DataFrame of mid-atoms
    for idx, row in mid_atoms_chg_H_df.iterrows():
        # calculate the position of the current atom in the repeating unit
        position_in_cycle = idx % num_H_repeating
        # find the average charge value of the atom at the corresponding position
        avg_chg_H = mid_ave_chg_H_df.iloc[position_in_cycle]['charge']
        # update the charge value
        mid_atoms_chg_H_df.at[idx, 'charge'] = avg_chg_H

    charge_update_df = pd.concat([top_N_noH_df, mid_atoms_chg_noH_df, tail_N_noH_df, top_N_H_df, mid_atoms_chg_H_df,
                                tail_N_H_df], ignore_index=True)

    # charge neutralize and scale
    corr_factor = model_info['polymer']['scale']
    charge_update_df_cor = charge_neutralize_scale(charge_update_df, target_sum_chg, corr_factor)

    itp_filepath = os.path.join(current_path, out_dir, f'{unit_name}_bonded.itp')

    # 读取.itp文件
    with open(itp_filepath, 'r') as file:
        lines = file.readlines()

    # 找到[ atoms ]部分的开始和结束
    in_section = False  # 标记是否处于指定部分
    start_index = end_index = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("[") and 'atoms' in line.split():
            in_section = True
            continue
        if in_section:
            if line.strip().startswith(";"):
                start_index = i + 1  # 跳过部分标题和列标题
                continue
            if line.strip() == "":
                end_index = i
                break

    # update the charge value in the [ atoms ] section
    charge_index = 0  # 用于跟踪DataFrame中当前的电荷索引
    for i in range(start_index, end_index):
        parts = lines[i].split()
        if charge_index < len(charge_update_df_cor):
            new_charge = charge_update_df_cor.iloc[charge_index]['charge']
            parts[6] = f'{new_charge:.8f}'  # 更新电荷值，假设电荷值在第7个字段
            lines[i] = ' '.join(parts) + '\n'
            charge_index += 1

    # save the updated itp file
    new_itp_filepath = os.path.join(current_path, out_dir, f'{unit_name}_bonded.itp')
    with open(new_itp_filepath, 'w') as file:
        file.writelines(lines)


def apply_chg_tomole(name, out_dir, corr_factor, method, target_sum_chg=0,):

    # read resp fitting result from csv file
    current_path = os.getcwd()
    MD_dir = os.path.join(current_path, out_dir)
    resp_chg_file = os.path.join(MD_dir, 'resp_work', f'{method}_chg.csv')
    resp_chg_df = pd.read_csv(resp_chg_file)

    # charge neutralize and scale
    charge_update_df_cor = charge_neutralize_scale(resp_chg_df , target_sum_chg, corr_factor)

    itp_filepath = os.path.join(MD_dir, f'{name}_bonded.itp')

    # 读取.itp文件
    with open(itp_filepath, 'r') as file:
        lines = file.readlines()

    # 找到[ atoms ]部分的开始和结束
    in_section = False  # 标记是否处于指定部分
    start_index = end_index = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("[") and 'atoms' in line.split():
            in_section = True
            continue
        if in_section:
            if line.strip().startswith(";"):
                start_index = i + 1  # 跳过部分标题和列标题
                continue
            if line.strip() == "":
                end_index = i
                break

    # update the charge value in the [ atoms ] section
    charge_index = 0  # 用于跟踪DataFrame中当前的电荷索引
    for i in range(start_index, end_index):
        parts = lines[i].split()
        if charge_index < len(charge_update_df_cor):
            new_charge = charge_update_df_cor.iloc[charge_index]['charge']
            parts[6] = f'{new_charge:.8f}'  # 更新电荷值，假设电荷值在第7个字段
            lines[i] = ' '.join(parts) + '\n'
            charge_index += 1

    # save the updated itp file
    new_itp_filepath = os.path.join(MD_dir,f'{name}_bonded.itp')
    with open(new_itp_filepath, 'w') as file:
        file.writelines(lines)

def scale_chg_itp(name, filename, corr_factor, target_sum_chg):
    # 标记开始读取数据
    start_reading = False
    atoms = []

    with open(filename, 'r') as file:
        for line in file:
            if line.strip().startswith("[") and 'atoms' in line.split():  # 找到原子信息开始的地方
                start_reading = True
                continue
            if start_reading:
                if line.strip() == "":  # 遇到空行，停止读取
                    break
                if line.strip().startswith(";"):  # 忽略注释行
                    continue
                parts = line.split()
                if len(parts) >= 7:  # 确保行包含足够的数据
                    atom_id = parts[4]  # 假设原子类型在第5列
                    charge = float(parts[6])  # 假设电荷在第7列
                    atoms.append([atom_id, charge])

    # create DataFrame
    df = pd.DataFrame(atoms, columns=['atom', 'charge'])
    # charge neutralize and scale
    charge_update_df_cor = charge_neutralize_scale(df, target_sum_chg, corr_factor)

    # reas itp file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 找到[ atoms ]部分的开始和结束
    in_section = False  # 标记是否处于指定部分
    start_index = end_index = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("[") and 'atoms' in line.split():
            in_section = True
            continue
        if in_section:
            if line.strip().startswith(";"):
                start_index = i + 1  # 跳过部分标题和列标题
                continue
            if line.strip() == "":
                end_index = i
                break

    # update the charge value in the [ atoms ] section
    charge_index = 0  # 用于跟踪DataFrame中当前的电荷索引
    for i in range(start_index, end_index):
        parts = lines[i].split()
        if charge_index < len(charge_update_df_cor):
            new_charge = charge_update_df_cor.iloc[charge_index]['charge']
            parts[6] = f'{new_charge:.8f}'  # 更新电荷值，假设电荷值在第7个字段
            lines[i] = ' '.join(parts) + '\n'
            charge_index += 1

    # save the updated itp file
    current_path = os.getcwd()
    MD_dir = os.path.join(current_path, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)
    new_itp_filepath = os.path.join(MD_dir, f'{name}_bonded.itp')
    with open(new_itp_filepath, 'w') as file:
        file.writelines(lines)

def charge_neutralize_scale(df, target_total_charge, correction_factor):
    current_total_charge = df['charge'].sum()  # calculate the total charge of the current system
    charge_difference = target_total_charge - current_total_charge
    charge_adjustment_per_atom = charge_difference / len(df)
    # update the charge value
    df['charge'] = (df['charge'] + charge_adjustment_per_atom) * correction_factor

    return df



