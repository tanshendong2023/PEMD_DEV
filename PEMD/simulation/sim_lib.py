# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import os
import re
import time
import glob
import subprocess
import numpy as np
from rdkit import Chem
from openbabel import openbabel as ob
from simple_slurm import Slurm
from PEMD.model import model_lib
import PEMD.model.MD_lib as MDlib
from pysimm import system, lmps, forcefield

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
def order_energy_xtb(work_dir, xyz_file, numconf):

    sorted_xtb_file = os.path.join(work_dir, f'sorted_xtb_top{numconf}.xyz')

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

    # 处理最后一个结构
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

    # 根据能量排序
    structures.sort(key=lambda x: x[0])

    # 选择最低能量的结构
    selected_structures = structures[:numconf]

    # 写入到输出文件
    with open(sorted_xtb_file, 'w') as outfile:
        for energy, structure in selected_structures:
            for line_num, line in enumerate(structure):
                if line_num == 1:
                    outfile.write(f"Energy = {energy}\n")
                else:
                    outfile.write(f"{line}\n")

    print(f"The lowest {numconf} energy structures have been written to {sorted_xtb_file}")
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

def read_final_structure_from_gaussian(log_file_path):
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
    # 循环结束后，start_idx 和 end_idx 对应最后一个 'Standard orientation:' 块

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

def order_energy_gaussian(work_dir, numconf):

    gaussian_dir = os.path.join(work_dir, 'conformer_search', 'gaussian_work')
    os.makedirs(gaussian_dir, exist_ok=True)

    data = []
    file_pattern = re.compile(r'^conf_\d+\.log$')
    # Traverse all files in the specified folder
    for file in os.listdir(gaussian_dir):
        if file_pattern.match(file):
            log_file_path = os.path.join(gaussian_dir, file)
            energy = read_energy_from_gaussian(log_file_path)
            atoms = read_final_structure_from_gaussian(log_file_path)
            if energy is not None and atoms is not None:
                data.append({"Energy": energy, "Atoms": atoms})

    # Check if data is not empty
    output_file=f"sorted_gaussian_top{numconf}.xyz"
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
        print(f"No successful Gaussian output files found in {gaussian_dir}")

def get_gaff2(unit_name, length, relax_polymer_lmp_dir, mol, atom_typing='pysimm'):
    print("\nGenerating GAFF2 parameter file ...\n")
    # r = MDlib.get_coord_from_pdb(outfile_name + ".pdb")
    # from pysimm import system, forcefield

    file_base = relax_polymer_lmp_dir + '/' + f'{unit_name}_N{length}'

    obConversion.SetInAndOutFormats("pdb", "cml")
    if os.path.exists(file_base + '.pdb'):
        mol = ob.OBMol()
        obConversion.ReadFile(mol, file_base + '.pdb')
    else:
        try:
            count_atoms = mol.NumAtoms()
            if count_atoms > 0:
                pass
            else:
                print("ERROR: Number of atoms = ", count_atoms)
                print("Couldn't generate GAFF2 parameter file\n")
                return
        except BaseException:
            print("ERROR: pdb file not found; OBMol not provided")
            print("Couldn't generate GAFF2 parameter file\n")
            return

    obConversion.WriteFile(mol, file_base + '.cml')
    data_fname = file_base + '_gaff2.lmp'

    try:
        print("Pysimm working on {}".format(file_base + '.mol2'))
        s = system.read_cml(file_base + '.cml')

        f = forcefield.Gaff2()
        if atom_typing == 'pysimm':
            for b in s.bonds:
                if b.a.bonds.count == 3 and b.b.bonds.count == 3:
                    b.order = 4
            s.apply_forcefield(f, charges='gasteiger')
        elif atom_typing == 'antechamber':
            obConversion.SetOutFormat("mol2")
            obConversion.WriteFile(mol, file_base + '.mol2')
            print("Antechamber working on {}".format(file_base + '.mol2'))
            MDlib.get_type_from_antechamber(s, file_base + '.mol2', 'gaff2', f)
            s.pair_style = 'lj'
            s.apply_forcefield(f, charges='gasteiger', skip_ptypes=True)
        else:
            print('Invalid atom typing option, please select pysimm or antechamber.')
        s.write_lammps(data_fname)
        print("\nGAFF2 parameter file generated.")
    except BaseException:
        print('problem reading {} for Pysimm.'.format(file_base + '.cml'))


def relax_polymer_lmp(unit_name, length, relax_polymer_lmp_dir, core):
    origin_dir = os.getcwd()
    os.chdir(relax_polymer_lmp_dir)
    # 创建LAMMPS输入文件字符串
    file_base = f'{unit_name}_N{length}'
    lmp_commands = """
    units real
    boundary s s s
    dimension 3
    atom_style full
    bond_style harmonic
    angle_style harmonic
    dihedral_style fourier
    improper_style harmonic
    pair_style lj/cut 2.0
    read_data {0}_gaff2.lmp
    thermo 100
    thermo_style custom step temp pxx pyy pzz ebond eangle edihed eimp epair ecoul evdwl pe ke etotal lx ly lz vol density
    fix xwalls all wall/reflect xlo EDGE xhi EDGE
    fix ywalls all wall/reflect ylo EDGE yhi EDGE
    fix zwalls all wall/reflect zlo EDGE zhi EDGE
    velocity all create 300 3243242
    minimize 1e-8 1e-8 10000 10000
    dump 1 all custom 500 soft.lammpstrj id type mass mol x y z
    fix 1 all nvt temp 800 800 100 drag 2
    run 5000000
    write_data {0}_gaff2.data
    write_dump all xyz {0}_lmp.xyz
    """.format(file_base)

    # 将LAMMPS输入命令写入临时文件
    with open('lammps_input.in', "w") as f:
        f.write(lmp_commands)

    slurm = Slurm(J='lammps',
                  N=1,
                  n=f'{core}',
                  output=f'slurm.{Slurm.JOB_ARRAY_MASTER_ID}.out'
                  )

    slurm.add_cmd('module load LAMMPS')
    job_id = slurm.sbatch('mpirun lmp < lammps_input.in >out.lmp 2>lmp.err')
    while True:
        status = get_slurm_job_status(job_id)
        if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print("\n", unit_name, ": MD simulation normally terminated.\n")
            model_lib.toxyz_lammps(f'{file_base}_lmp.xyz', f'{file_base}_gmx.xyz', f'{file_base}_gaff2.lmp')
            os.chdir(origin_dir)
            break
        else:
            print("polymer relax not finish, waiting...")
            time.sleep(10)

