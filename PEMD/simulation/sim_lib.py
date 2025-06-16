# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import os
import re
import subprocess
import numpy as np
from rdkit import Chem


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


def read_energy_from_gaussian(log_file_path: str):
    # 自由能（优化+freq）行
    pattern_free = re.compile(
        r"Sum of electronic and thermal Free Energies\s*=\s*(-?\d+\.\d+)"
    )
    # SCF 单点能行
    pattern_scf = re.compile(
        r"SCF Done:\s+E\(\w+.*?\)\s+=\s+(-?\d+\.\d+)"
    )

    energy_free = None
    energy_scf = None

    with open(log_file_path, 'r') as f:
        for line in f:
            m_free = pattern_free.search(line)
            if m_free:
                energy_free = float(m_free.group(1))
            m_scf = pattern_scf.search(line)
            if m_scf:
                energy_scf = float(m_scf.group(1))

    # 优先返回自由能，否则返回 SCF 能量
    if energy_free is not None:
        return energy_free
    return energy_scf


def read_final_structure_from_gaussian(log_file_path):

    if not os.path.exists(log_file_path):
        print(f"File not found: {log_file_path}")
        return None

    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"Error reading file {log_file_path}: {e}")
        return None

    # Define the sections to search for
    orientation_sections = ['Standard orientation:', 'Input orientation:']
    start_idx = None
    end_idx = None

    # Iterate through the file to find the last occurrence of the orientation sections
    for i, line in enumerate(lines):
        for section in orientation_sections:
            if section in line:
                # Assume that coordinate data starts 5 lines after the section header
                current_start = i + 5
                # Search for the line that indicates the end of the coordinate block
                for j in range(current_start, len(lines)):
                    if '-----' in lines[j]:
                        current_end = j
                        break
                else:
                    # If no separator line is found, skip to the next section
                    continue
                # Update start and end indices to the latest found section
                start_idx, end_idx = current_start, current_end

    if start_idx is None or end_idx is None or start_idx >= end_idx:
        print(f"No valid atomic coordinates found in {log_file_path}")
        return None

    atoms = []
    periodic_table = Chem.GetPeriodicTable()

    for line in lines[start_idx:end_idx]:
        tokens = line.strip().split()
        if len(tokens) < 6:
            continue  # Skip lines that do not have enough tokens
        try:
            atom_number = int(tokens[1])  # Atomic number is the second token
            x = float(tokens[3])
            y = float(tokens[4])
            z = float(tokens[5])
            atom_symbol = periodic_table.GetElementSymbol(atom_number)
            atoms.append(f"{atom_symbol}   {x:.6f}   {y:.6f}   {z:.6f}")
        except ValueError:
            # Handle cases where conversion to int or float fails
            continue
        except Exception as e:
            print(f"Unexpected error parsing line: {line}\nError: {e}")
            continue

    if not atoms:
        print(f"No valid atomic coordinates extracted from {log_file_path}")
        return None

    return atoms

def order_energy_gaussian(work_dir, filename, numconf, output_file):

    data = []
    escaped = re.escape(filename)
    file_pattern = re.compile(rf'^{escaped}_\d+\.log$')
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
    output_filename = f'{file_prefix}_gmx.xyz'

    atom_map = parse_masses_from_lammps(data_filepath)

    with open(input_filepath, 'r') as fin, open(output_filename, 'w') as fout:
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

    print(f"the relaxed polymer chian has been written to {output_filename}\n")

    return output_filename

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

def smiles_to_atom_string(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    atom_list = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol != 'H':
            atom_list.append(symbol)
    atom_string = ''.join(atom_list)

    return atom_string