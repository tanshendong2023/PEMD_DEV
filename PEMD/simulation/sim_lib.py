# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import os
import re
import subprocess
import numpy as np
import pandas as pd
from rdkit import Chem
from openbabel import openbabel as ob
from PEMD.model import model_lib


# OpenBabel setup
obConversion = ob.OBConversion()
ff = ob.OBForceField.FindForceField('UFF')
mol = ob.OBMol()
np.set_printoptions(precision=20)

def get_slurm_job_status(job_id):
    command = f'sacct -j {job_id} --format=State --noheader'
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Split the output by newlines and strip whitespace
    statuses = [line.strip() for line in process.stdout.strip().split('\n')]
    # Check if all statuses indicate the job is completed
    if all(status == 'COMPLETED' for status in statuses):
        return 'COMPLETED'
    elif any(status == 'FAILED' for status in statuses):
        return 'FAILED'
    elif any(status == 'CANCELLED' for status in statuses):
        return 'CANCELLED'
    else:
        return 'RUNNING'

# Modified order_energy_xtb function
def order_energy_xtb(work_dir, xyz_file, numconf, output_file):

    sorted_xtb_file = os.path.join(work_dir, output_file)

    structures = []
    current_structure = []

    with open(xyz_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.isdigit():
                if current_structure:
                    if len(current_structure) >= 2:
                        energy_line = current_structure[1]
                        try:
                            energy_match = re.search(r"[-+]?\d*\.\d+|\d+", energy_line)
                            if energy_match:
                                energy = float(energy_match.group())
                            else:
                                raise ValueError("No numeric value found")
                        except ValueError:
                            print(f"Could not parse energy value: {energy_line}")
                            energy = float('inf')
                        structures.append((energy, current_structure))
                    else:
                        print("Malformed structure encountered.")
                    current_structure = []
                current_structure.append(line)
            else:
                current_structure.append(line)

    if current_structure:
        if len(current_structure) >= 2:
            energy_line = current_structure[1]
            try:
                energy_match = re.search(r"[-+]?\d*\.\d+|\d+", energy_line)
                if energy_match:
                    energy = float(energy_match.group())
                else:
                    raise ValueError("No numeric value found")
            except ValueError:
                print(f"Could not parse energy value: {energy_line}")
                energy = float('inf')
            structures.append((energy, current_structure))
        else:
            print("Malformed structure encountered.")

    structures.sort(key=lambda x: x[0])
    selected_structures = structures[:numconf]

    with open(sorted_xtb_file, 'w') as outfile:
        for energy, structure in selected_structures:
            for line_num, line in enumerate(structure):
                if line_num == 1:
                    outfile.write(f"Energy = {energy}\n")
                else:
                    outfile.write(f"{line}\n")

    print(f"The lowest {numconf} energy structures have been written to {output_file}")
    # return sorted_xtb_file

# input: a xyz file
# output: a list store the xyz structure
# Description: read the xyz file and store the structure in a list
def read_xyz_file(file_path):
    structures = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        num_atoms_line = lines[i].strip()
        if num_atoms_line.isdigit():
            num_atoms = int(num_atoms_line)
            comment_line = lines[i + 1].strip()
            atoms = []
            for j in range(i + 2, i + 2 + num_atoms):
                atom_line = lines[j].strip()
                atoms.append(atom_line)
            structure = {
                'num_atoms': num_atoms,
                'comment': comment_line,
                'atoms': atoms
            }
            structures.append(structure)
            i = i + 2 + num_atoms
        else:
            i += 1
    return structures

def submit_job_to_slurm(command, job_name, node, core, mem, gaussian_dir, file, soft):

    file_contents = f"#!/bin/bash"
    file_contents += f"#SBATCH -J {job_name}\n"
    file_contents += f"#SBATCH -N {node}\n"
    file_contents += f"#SBATCH -n {core}\n"
    file_contents += f"#SBATCH -p {mem}\n\n"
    file_contents += f"module load {soft}\n\n"
    file_contents += f"{command} {file}/\n"

    # 将作业脚本写入文件
    script_path = os.path.join(gaussian_dir, f"sub.sh")
    with open(script_path, 'w') as f:
        f.write(file_contents)

    # 提交作业并获取作业 ID
    submit_command = ['sbatch', script_path]
    result = subprocess.run(submit_command, capture_output=True, text=True)
    if result.returncode == 0:
        # 提取作业 ID
        output = result.stdout.strip()
        # 通常，sbatch 的输出格式为 "Submitted batch job <job_id>"
        job_id = output.split()[-1]
        return job_id
    else:
        print(f"提交作业失败：{result.stderr}")
        return None

def read_energy_from_gaussian(log_file_path):
    """
    从 Gaussian 输出文件中读取能量（自由能）
    """
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
    energy = None
    for line in lines:
        if 'Sum of electronic and thermal Free Energies=' in line:
            energy = float(line.strip().split()[-1])
    return energy

def read_final_structure_from_gaussian(log_file_path,):
    if not os.path.exists(log_file_path):
        print(f"File not found: {log_file_path}")
        return None

    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if 'Standard orientation:' in line:
            start_idx = i + 5  # 坐标数据从 'Standard orientation:' 后的第5行开始
            # 从 start_idx 开始寻找结束的分隔线
            for j in range(start_idx, len(lines)):
                if '---------------------------------------------------------------------' in lines[j]:
                    end_idx = j
                    break  # 找到当前块的结束位置

    if start_idx is not None and end_idx is not None and start_idx < end_idx:
        atoms = []
        for line in lines[start_idx:end_idx]:
            tokens = line.strip().split()
            if len(tokens) >= 6:
                atom_number = int(tokens[1])
                x, y, z = float(tokens[3]), float(tokens[4]), float(tokens[5])
                atom_symbol = Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), atom_number)
                atoms.append(f"{atom_symbol}   {x}   {y}   {z}")
        return atoms

    print(f"No valid atomic coordinates found between lines {start_idx} and {end_idx}")
    return None

def order_energy_gaussian(work_dir, numconf, output_file):

    data = []
    file_pattern = re.compile(r'^conf_\d+\.log$')
    # Traverse all files in the specified folder
    for file in os.listdir(work_dir):
        if file_pattern.match(file):
            log_file_path = os.path.join(work_dir, file)
            energy = read_energy_from_gaussian(log_file_path)
            atoms = read_final_structure_from_gaussian(log_file_path)
            if energy is not None and atoms is not None:
                data.append({"Energy": energy, "Atoms": atoms})

    # Check if data is not empty
    if data:
        # Sort the structures by energy
        sorted_data = sorted(data, key=lambda x: x['Energy'])
        selected_data = sorted_data[:numconf]
        # Write the sorted structures to an .xyz file
        with open(output_file, 'w') as outfile:
            for item in selected_data:
                num_atoms = len(item['Atoms'])
                outfile.write(f"{num_atoms}\n")
                outfile.write(f"Energy = {item['Energy']}\n")
                for atom_line in item['Atoms']:
                    outfile.write(f"{atom_line}\n")
        print(f"The lowest {numconf} energy structures have been saved to {output_file}")
    else:
        print(f"No successful Gaussian output files found in {work_dir}")


def lmptoxyz(work_dir, pdb_file):

    file_prefix, file_extension = os.path.splitext(pdb_file)
    data_filepath = os.path.join(work_dir, f'{file_prefix}_gaff2.lmp')
    input_filepath = os.path.join(work_dir, f'{file_prefix}_lmp.xyz')
    output_filepath = f'{file_prefix}_gmx.xyz'

    atom_map = parse_masses_from_lammps(data_filepath)

    with open(input_filepath, 'r') as fin, open(output_filepath, 'w') as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if i < 2:
                fout.write(line + '\n')
            else:
                parts = line.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    atom_id = int(parts[0])
                    if atom_id in atom_map:
                        parts[0] = atom_map[atom_id]
                    else:
                        print(f"Warning: Atom ID {atom_id} not found in atom_map.")
                fout.write(' '.join(parts) + '\n')

    print(f"the relaxed polymer chian has been written to {output_filepath}\n")

    return output_filepath


def parse_masses_from_lammps(data_filename):
    atom_map = {}
    with open(data_filename, 'r') as f:
        lines = f.readlines()

    masses_section = False
    for i, line in enumerate(lines):
        line_strip = line.strip()
        if line_strip == "Masses":
            masses_section = True
            start = i + 2  # Skip the next line (which is usually blank or comments)
            break

    if not masses_section:
        raise ValueError("Masses section not found in the LAMMPS data file.")

    # Now, parse the Masses section until an empty line or another section starts
    for line in lines[start:]:
        line_strip = line.strip()
        if line_strip == "" or any(line_strip.startswith(s) for s in ["Atoms", "Bonds", "Angles", "Dihedrals", "Impropers"]):
            break
        parts = line_strip.split()
        if len(parts) >= 2:
            atom_id = int(parts[0])
            mass = float(parts[1])
            atom_symbol = get_closest_element_by_mass(mass)
            atom_map[atom_id] = atom_symbol
    return atom_map


def get_closest_element_by_mass(target_mass, tolerance=0.5):
    import numpy as np

    # 手动定义一个包含元素质量的映射
    element_masses = {
        'H': 1.008,  # 氢
        'B': 10.81,  # 硼
        'C': 12.011,  # 碳
        'N': 14.007,  # 氮
        'O': 15.999,  # 氧
        'F': 18.998,  # 氟
        'Na': 22.990,  # 钠
        'Mg': 24.305,  # 镁
        'Al': 26.982,  # 铝
        'Si': 28.085,  # 硅
        'P': 30.974,  # 磷
        'S': 32.06,  # 硫
        'Cl': 35.45,  # 氯
        'K': 39.098,  # 钾
        'Ca': 40.078,  # 钙
        'Ti': 47.867,  # 钛
        'Cr': 51.996,  # 铬
        'Mn': 54.938,  # 锰
        'Fe': 55.845,  # 铁
        'Ni': 58.693,  # 镍
        'Cu': 63.546,  # 铜
        'Zn': 65.38,  # 锌
        'Br': 79.904,  # 溴
        'Ag': 107.87,  # 银
        'I': 126.90,  # 碘
        'Au': 196.97,  # 金
        # 可以根据需要添加更多元素
    }

    min_diff = np.inf
    closest_element = None

    for element, mass in element_masses.items():
        diff = abs(mass - target_mass)
        if diff < min_diff:
            min_diff = diff
            closest_element = element

    if min_diff > tolerance:
        print(f"Warning: No element found for mass {target_mass} within tolerance {tolerance}")
        closest_element = 'X'

    return closest_element

def apply_chg_to_poly(work_dir, itp_file, resp_chg_df, repeating_unit, end_repeating, corr_factor, target_sum_chg, ):

    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    nonH_df = resp_chg_df[resp_chg_df['atom'] != 'H']
    cleaned_smiles = repeating_unit.replace('[*]', '')
    left_non_matching, right_non_matching = count_non_matching_atoms(''.join(nonH_df['atom'].tolist()), cleaned_smiles)

    (
        top_N_noH_df,
        tail_N_noH_df,
        mid_ave_chg_noH_df,
        top_N_H_df,
        tail_N_H_df,
        mid_ave_chg_H_df
    ) = ave_chg_to_df(
        resp_chg_df,
        repeating_unit,
        end_repeating,
        left_non_matching,
        right_non_matching
    )

    # read the xyz file
    molecule_name = itp_file.replace("_bonded.itp", "")
    pdb_file = os.path.join(MD_dir, f'{molecule_name}.pdb')
    xyz_file = os.path.join(MD_dir, f'{molecule_name}.xyz')
    model_lib.convert_pdb_to_xyz(pdb_file, xyz_file)
    atoms_chg_df = xyz_to_df(xyz_file)

    # deal with the mid non-H atoms
    atoms_chg_noH_df = atoms_chg_df[atoms_chg_df['atom'] != 'H']

    cleaned_smiles = repeating_unit.replace('[*]', '')
    molecule = Chem.MolFromSmiles(cleaned_smiles)
    atom_count = molecule.GetNumAtoms()
    # N = atom_count * end_repeating + 1
    N_left = atom_count * end_repeating + 1*left_non_matching
    N_right = atom_count * end_repeating + 1*right_non_matching

    mid_atoms_chg_noH_df = atoms_chg_noH_df.drop(
        atoms_chg_noH_df.head(N_left).index.union(atoms_chg_noH_df.tail(N_right).index)).reset_index(drop=True)

    # 计算每个原子在重复单元中的位置
    positions_in_cycle = mid_atoms_chg_noH_df.index % atom_count
    # 更新电荷值
    mid_atoms_chg_noH_df['charge'] = mid_ave_chg_noH_df.iloc[positions_in_cycle]['charge'].values

    # deal with the mid H atoms
    atoms_chg_H_df = atoms_chg_df[atoms_chg_df['atom'] == 'H']

    molecule_with_h = Chem.AddHs(molecule)
    num_H_repeating = molecule_with_h.GetNumAtoms() - molecule.GetNumAtoms() - 2
    # N_H = num_H_repeating * end_repeating + 3
    N_H_left = num_H_repeating * end_repeating + 3*left_non_matching      # 多三个端集CH3中的H
    N_H_right = num_H_repeating * end_repeating + 3*right_non_matching    # 多三个端集CH3中的H

    mid_atoms_chg_H_df = atoms_chg_H_df.drop(
        atoms_chg_H_df.head(N_H_left).index.union(atoms_chg_H_df.tail(N_H_right).index)).reset_index(drop=True)

    # 计算每个原子在重复单元中的位置
    positions_in_cycle_H = mid_atoms_chg_H_df.index % num_H_repeating
    # 更新电荷值
    mid_atoms_chg_H_df['charge'] = mid_ave_chg_H_df.iloc[positions_in_cycle_H]['charge'].values

    charge_update_df = pd.concat([top_N_noH_df, mid_atoms_chg_noH_df, tail_N_noH_df, top_N_H_df, mid_atoms_chg_H_df,
                                tail_N_H_df], ignore_index=True)

    # charge neutralize and scale
    # corr_factor = model_info['polymer']['scale']
    charge_update_df_cor = charge_neutralize_scale(charge_update_df, corr_factor, target_sum_chg, )

    # update the itp file
    update_itp_file(MD_dir, itp_file, charge_update_df_cor)

def update_itp_file(MD_dir, itp_file, charge_update_df_cor):
    itp_filepath = os.path.join(MD_dir, itp_file)
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
    new_itp_filepath = os.path.join(MD_dir, itp_file)
    with open(new_itp_filepath, 'w') as file:
        file.writelines(lines)

def ave_end_chg(df, N):
    # 处理端部原子的电荷平均值
    top_N_df = df.head(N)
    tail_N_df = df.tail(N).iloc[::-1].reset_index(drop=True)
    average_charge = (top_N_df['charge'].reset_index(drop=True) + tail_N_df['charge']) / 2
    average_df = pd.DataFrame({
        'atom': top_N_df['atom'].reset_index(drop=True),  # 保持原子名称
        'charge': average_charge
    })
    return average_df

def ave_mid_chg(df, atom_count):
    # 处理中间原子的电荷平均值
    average_charges = []
    for i in range(atom_count):
        same_atoms = df[df.index % atom_count == i]
        avg_charge = same_atoms['charge'].mean()
        average_charges.append({'atom': same_atoms['atom'].iloc[0], 'charge': avg_charge})
    return pd.DataFrame(average_charges)

def xyz_to_df(xyz_file_path):
    # 初始化空列表来存储原子类型
    atoms = []

    # 读取XYZ文件
    with open(xyz_file_path, 'r') as file:
        next(file)  # 跳过第一行（原子总数）
        next(file)  # 跳过第二行（注释行）
        for line in file:
            atom_type = line.split()[0]  # 原子类型是每行的第一个元素
            atoms.append(atom_type)

    # 创建DataFrame
    df = pd.DataFrame(atoms, columns=['atom'])

    # 添加空的'charge'列
    df['charge'] = None  # 初始化为空值

    return df

def count_non_matching_atoms(sequence, repeating_unit_sequence):
    seq_len = len(sequence)
    unit_len = len(repeating_unit_sequence)

    # 初始化计数
    left_non_matching = 0
    right_non_matching = 0

    # 从左侧开始寻找重复单元的第一次出现
    first_occurrence = sequence.find(repeating_unit_sequence)
    if first_occurrence != -1:
        left_non_matching = first_occurrence
    else:
        # 未找到重复单元
        left_non_matching = seq_len
        right_non_matching = 0
        return left_non_matching, right_non_matching

    # 从右侧开始寻找重复单元的最后一次出现
    last_occurrence = sequence.rfind(repeating_unit_sequence)
    if last_occurrence != -1:
        right_non_matching = seq_len - (last_occurrence + unit_len)
    else:
        right_non_matching = seq_len

    return left_non_matching, right_non_matching

def ave_chg_to_df(resp_chg_df, repeating_unit, num_repeating, left_non_matching, right_non_matching):

    # 处理非氢原子
    nonH_df = resp_chg_df[resp_chg_df['atom'] != 'H']
    cleaned_smiles = repeating_unit.replace('[*]', '')

    molecule = Chem.MolFromSmiles(cleaned_smiles)
    atom_count = molecule.GetNumAtoms()
    N_left = atom_count * num_repeating + 1*left_non_matching
    N_right = atom_count * num_repeating + 1*right_non_matching

    # end_ave_chg_noH_df = ave_end_chg(nonH_df, N)
    top_N_noH_df = nonH_df.head(N_left)
    tail_N_noH_df = nonH_df.tail(N_right)
    mid_df_noH_df = nonH_df.drop(nonH_df.head(N_left).index.union(nonH_df.tail(N_right).index)).reset_index(drop=True)
    mid_ave_chg_noH_df = ave_mid_chg(mid_df_noH_df, atom_count)

    # 处理氢原子
    H_df = resp_chg_df[resp_chg_df['atom'] == 'H']

    molecule_with_h = Chem.AddHs(molecule)
    num_H_repeating = molecule_with_h.GetNumAtoms() - molecule.GetNumAtoms() - 2
    N_H_left = num_H_repeating * num_repeating + 3*left_non_matching      # 多三个端集CH3中的H
    N_H_right = num_H_repeating * num_repeating + 3*right_non_matching    # 多三个端集CH3中的H

    # end_ave_chg_H_df = ave_end_chg(H_df, N_H)
    top_N_H_df = H_df.head(N_H_left)
    tail_N_H_df = H_df.tail(N_H_right)
    mid_df_H_df = H_df.drop(H_df.head(N_H_left).index.union(H_df.tail(N_H_right).index)).reset_index(drop=True)
    mid_ave_chg_H_df = ave_mid_chg(mid_df_H_df, num_H_repeating)

    return top_N_noH_df, tail_N_noH_df, mid_ave_chg_noH_df, top_N_H_df, tail_N_H_df, mid_ave_chg_H_df

def charge_neutralize_scale(df, correction_factor=1, target_total_charge=0,):

    current_total_charge = df['charge'].sum()
    charge_difference = target_total_charge - current_total_charge
    charge_adjustment_per_atom = charge_difference / len(df)
    df['charge'] = (df['charge'] + charge_adjustment_per_atom) * correction_factor

    return df

def apply_chg_to_molecule(work_dir, itp_file, resp_chg_df, corr_factor, target_sum_chg,):

    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    # charge neutralize and scale
    charge_update_df_cor = charge_neutralize_scale(resp_chg_df , corr_factor, target_sum_chg, )

    # update the itp file
    update_itp_file(MD_dir, itp_file, charge_update_df_cor)

# work_dir, filename, corr_factor, target_sum_chg
def scale_chg_itp(work_dir, filename, corr_factor, target_sum_chg):

    filename = os.path.join(work_dir, filename)
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
    charge_update_df_cor = charge_neutralize_scale(df, corr_factor, target_sum_chg,)

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
    with open(filename, 'w') as file:
        file.writelines(lines)