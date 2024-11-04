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
from PEMD.simulation.xtb import PEMDXtb
from PEMD.simulation.gaussian import PEMDGaussian
from PEMD.simulation.slurm import PEMDSlurm


# Input: smiles (str)
# Output: a xyz file
# Description: Generates multiple conformers for a molecule from a SMILES string, optimizes them using the MMFF94
# force field, and saves the optimized conformers to a single XYZ file.
def gen_conf_rdkit(
        work_dir,
        smiles,
        max_conformers,
        top_n_MMFF
):

    # build dir
    rdkit_dir = os.path.join(work_dir, 'conformer_search', 'rdkit_work')
    os.makedirs(rdkit_dir, exist_ok=True)

    # Generate multiple conformers
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

    # Sort the conformers by energy and select the top N conformers
    minimized_conformers.sort(key=lambda x: x[1])
    top_conformers = minimized_conformers[:top_n_MMFF]

    # save the top conformers to xyz files
    for conf_id, _ in top_conformers:
        xyz_file = os.path.join(rdkit_dir, f'conf_{conf_id}.xyz')
        model_lib.mol_to_xyz(mol, conf_id, xyz_file)
    filenames = glob.glob(os.path.join(rdkit_dir, '*.xyz'))
    merged_file = os.path.join(work_dir, 'merged_rdkit.xyz')
    with open(merged_file, 'w') as outfile:
        for fname in filenames:
            with open(fname, 'r') as infile:
                outfile.write(infile.read())
    print(f"The generated conformers based on rdkit saved to {merged_file}")
    # return merged_file

# input: a xyz file
# output:  a xyz file
# Description: Sorts the conformers in a XYZ file by energy calculated by xTB and saves the sorted conformers to a
# new file
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

    # build dir
    xtb_dir = os.path.join(work_dir, 'conformer_search', 'xtb_work')
    os.makedirs(xtb_dir, exist_ok=True)

    # read xyz file as a list of structures
    structures = sim_lib.read_xyz_file(xyz_file)

    # Perform xTB optimization on the selected conformers
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

        # Create xtb object
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

        # Run xtb local or generate SLURM script
        # if not slurm:
        #     xtb.run_local()
        # else:
        #     script_name = f'sub.xtb_{conf_id}'
        #     xtb.gen_slurm(
        #         script_name,
        #         job_name,
        #         nodes,
        #         ntasks_per_node,
        #         partition
        #     )

    if not slurm:
        print("XTB run locally successfully!")
        filenames = glob.glob(os.path.join(xtb_dir, '*.xtbopt.xyz'))
        merged_file = os.path.join(xtb_dir, 'merged.xyz')
        with open(merged_file, 'w') as outfile:
            for fname in filenames:
                with open(fname, 'r') as infile:
                    outfile.write(infile.read())
        return merged_file
    else:
        script_name = f'sub.xtb'
        PEMDXtb(
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
        print("XTB submit scripts generated successfully!")
        # return xtb_dir   # Optionally return a list of script names or paths

# input: a xyz file
# output: a xyz file
# description:
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

    gaussian_dir = os.path.join(work_dir, 'conformer_search', 'gaussian_work')
    os.makedirs(gaussian_dir, exist_ok=True)

    structures = sim_lib.read_xyz_file(xyz_file)

    # job_ids = []
    for i, structure in enumerate(structures):

        # 调用 QMcalc_gaussian 函数生成 Gaussian 输入文件
        filename = f"conf_{i}.gjf"
        Gau = PEMDGaussian(
            gaussian_dir,
            ntasks_per_node,
            mem,
            chg,
            mult,
            function,
            basis_set,
            epsilon,
        )

        Gau.generate_input_file(
            work_dir,
            structure,
            filename=filename
        )

        script_name = f'sub.gaussian'
        Gau.gen_slurm(
            script_name,
            job_name,
            nodes,
            ntasks_per_node,
            partition
        )

    print("Gaussian input files generated successfully!")


def calc_resp_gaussian(
        sorted_df,
        epsilon,
        core=32,
        memory='64GB',
        method='resp2',
):

    current_path = os.getcwd()
    resp_dir = os.path.join(current_path, 'RESP_work')
    os.makedirs(resp_dir, exist_ok=True)

    job_ids = []
    for i in range(5):    # only calculate the first 5 conformers
        log_file_path = sorted_df.iloc[0]['File_Path']
        chk_name = log_file_path.replace('.log', '.chk')

        # RESP template
        file_contents = f"nprocshared={core}\n"
        file_contents += f"%mem={memory}\n"
        file_contents += f"%oldchk={chk_name}\n"
        file_contents += f"%chk={resp_dir}/SP_gas_conf_{i}.chk\n"
        file_contents += f"# B3LYP/def2TZVP em=GD3BJ geom=allcheck\n\n"
        file_contents += "--link1--\n"
        file_contents += f"nprocshared={core}\n"
        file_contents += f"%mem={memory}\n"
        file_contents += f"%oldchk={chk_name}\n"
        file_contents += f"%chk={resp_dir}/SP_solv_conf_{i}.chk\n"
        file_contents += f"# B3LYP/def2TZVP em=GD3BJ scrf=(pcm,solvent=generic,read) geom=allcheck\n\n"
        file_contents += f"eps={epsilon}\n"
        file_contents += f"epsinf=2.1\n"
        file_contents += '\n\n'

        # create gjf file
        out_file = os.path.join(resp_dir, f'resp_conf_{i}.gjf')
        with open(out_file, 'w') as file:
            file.write(file_contents)

        slurm = Slurm(J='g16',
                      N=1,
                      n=f'{core}',
                      output=f'{resp_dir}/slurm.%A.out'
                      )

        job_id = slurm.sbatch(f'g16 {resp_dir}/resp_conf_{i}.gjf')
        time.sleep(1)
        job_ids.append(job_id)

    # check the status of the gaussian job
    while True:
        all_completed = True
        for job_id in job_ids:
            status = sim_lib.get_slurm_job_status(job_id)
            if status not in ['COMPLETED', 'FAILED', 'CANCELLED']:
                all_completed = False
                break
        if all_completed:
            print("All gaussian tasks finished, order structure with energy calculated by gaussian...")
            print("RESP calculation finish, executing the resp fit with Multiwfn...")
            df = RESP_fit_Multiwfn(resp_dir, method,)
            break
        else:
            print("RESP calculation not finish, waiting...")
            time.sleep(10)  # wait for 30 seconds

    return df


def RESP_fit_Multiwfn(out_dir, method,):

    current_path = os.getcwd()
    resp_dir = os.path.join(current_path, 'RESP_work')
    os.makedirs(resp_dir, exist_ok=True)

    chk_files = glob.glob('*.chk')
    for chk_file in chk_files:
        model_lib.convert_chk_to_fchk(chk_file)

    # 初始化DataFrame
    resp_chg_df = pd.DataFrame()

    # 使用importlib.resources获取脚本路径
    with resources.path("PEMD.analysis", "calcRESP.sh") as script_path:
        for i in range(5):   # only calculate the first 5 conformers
            if method == 'resp':
                command = ["bash", str(script_path), f"SP_gas_conf_{i}.fchk"]
            elif method == 'resp2':
                command = ["bash", str(script_path), f"SP_gas_conf_{i}.fchk", f"SP_solv_conf_{i}.fchk"]
            else:
                raise ValueError("Unsupported method. Please choose 'resp' or 'resp2'.")

            # 使用subprocess模块调用脚本
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # 输出命令执行结果
            if process.returncode == 0:
                print(f"RESP fitting for the {i + 1}-th structure has been successfully completed.")
            else:
                print(f"RESP fitting for the {i + 1}-th structure failed : {process.stderr}")

            if method == 'resp':
                with open('SP_solv.chg', 'r') as file:
                    lines = file.readlines()
            elif method == 'resp2':
                with open('RESP2.chg', 'r') as file:
                    lines = file.readlines()

            # 第一次循环时读取原子和电荷，后续只更新电荷
            if i == 0:
                data = []
                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:  # 假设格式为：Atom X Y Z Charge
                        atom_name = parts[0]
                        charge = float(parts[-1])
                        data.append((atom_name, charge))

                resp_chg_df = pd.DataFrame(data, columns=['atom', f'charge_{i}'])
            else:
                charges = []
                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:
                        charge = float(parts[-1])
                        charges.append(charge)

                # 将新的电荷数据添加为DataFrame的新列
                resp_chg_df[f'charge_{i}'] = charges

    # 计算所有charge列的平均值，并将结果存储在新列'charge'中
    charge_columns = [col for col in resp_chg_df.columns if 'charge' in col]
    resp_chg_df['charge'] = resp_chg_df[charge_columns].mean(axis=1)
    # 删除原始的charge列
    resp_chg_df.drop(columns=charge_columns, inplace=True)

    os.chdir(current_path)

    # to csv file
    csv_filepath = os.path.join(resp_dir, f'{method}_chg.csv')
    resp_chg_df.to_csv(csv_filepath, index=False)

    return resp_chg_df


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



