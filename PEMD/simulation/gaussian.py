# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import os
import subprocess
from PEMD.simulation.slurm import PEMDSlurm

class PEMDGaussian:
    def __init__(
        self,
        work_dir,
        filename,
        core = 32,
        mem = '64GB',
        chg = 0,
        mult = 1,
        function = 'B3LYP',
        basis_set = '6-31+g(d,p)',
        epsilon = 5.0,
    ):
        self.work_dir = work_dir
        self.filename = filename
        self.core = core
        self.mem = mem
        self.chg = chg
        self.mult = mult
        self.function = function
        self.basis_set = basis_set
        self.epsilon = epsilon

    def generate_input_file(self, structure=None, chk=None, gaucontinue=False):

        prefix, _ = os.path.splitext(self.filename)

        file_contents = f"%nprocshared={self.core}\n"
        file_contents += f"%mem={self.mem}\n"

        if chk:
            file_contents += f"%chk={self.work_dir}/{prefix}_opt.chk\n"

        if gaucontinue == True:
            file_contents += f"# opt(gdiis,maxstep=2,notrust,maxcyc=100,Cartesian) freq {self.function} {self.basis_set} em=GD3BJ scrf=(pcm,solvent=generic,read)\n\n"
        else:
            file_contents += f"# opt freq {self.function} {self.basis_set} em=GD3BJ scrf=(pcm,solvent=generic,read)\n\n"

        file_contents += 'qm calculation\n\n'
        file_contents += f'{self.chg} {self.mult}\n'  # 电荷和多重度
        for atom in structure['atoms']:
            atom_parts = atom.split()
            if len(atom_parts) >= 4:
                file_contents += f"{atom_parts[0]:<2} {float(atom_parts[1]):>15.6f} {float(atom_parts[2]):>15.6f} {float(atom_parts[3]):>15.6f}\n"
            else:
                print(f"Invalid atom line: {atom}")
                continue

        file_contents += '\n'
        file_contents += f"eps={self.epsilon}\n"
        file_contents += "epsinf=2.1\n"
        file_contents += '\n\n'

        # 创建 Gaussian 输入文件
        file_path = os.path.join(self.work_dir, self.filename)
        with open(file_path, 'w') as file:
            file.write(file_contents)

    def generate_input_file_resp(self, structure,):
        """
        Generates a Gaussian input file based on the given structure and other parameters.

        Args:
            gaussian_dir (str): The directory where the input file will be saved.
            structure (dict): The molecular structure containing atom names and their coordinates.
            filename (str): The name of the output Gaussian input file.
        """
        # Set Gaussian input file parameters.
        file_prefix, file_extension = os.path.splitext(self.filename)
        file_contents = f"%nprocshared={self.core}\n"
        file_contents += f"%mem={self.mem}\n"
        file_contents += f"%chk={self.work_dir}/resp_{file_prefix}.chk\n"
        file_contents += f"# opt B3LYP TZVP em=GD3BJ scrf=(pcm,solvent=generic,read)\n\n"
        file_contents += 'resp calculation\n\n'
        file_contents += f'{self.chg} {self.mult}\n'

        # Add atomic coordinates.
        for atom in structure['atoms']:
            atom_parts = atom.split()
            # Ensure each line contains the atom symbol and three coordinates
            if len(atom_parts) >= 4:
                file_contents += f"{atom_parts[0]:<2} {float(atom_parts[1]):>15.6f} {float(atom_parts[2]):>15.6f} {float(atom_parts[3]):>15.6f}\n"
            else:
                print(f"Invalid atom line: {atom}")
                continue

        # Settings for the implicit solvent model.
        file_contents += '\n'
        file_contents += f"eps={self.epsilon}\n"
        file_contents += '\n\n'

        file_contents += "--link1--\n"
        file_contents += f"nprocshared={self.core}\n"
        file_contents += f"%mem={self.mem}\n"
        file_contents += f"%oldchk={self.work_dir}/resp_{file_prefix}.chk\n"
        file_contents += f"%chk={self.work_dir}/SP_gas_{file_prefix}.chk\n"
        file_contents += f"# {self.function} {self.basis_set} em=GD3BJ geom=allcheck\n\n"

        file_contents += "--link1--\n"
        file_contents += f"nprocshared={self.core}\n"
        file_contents += f"%mem={self.mem}\n"
        file_contents += f"%oldchk={self.work_dir}/resp_{file_prefix}.chk\n"
        file_contents += f"%chk={self.work_dir}/SP_solv_{file_prefix}.chk\n"
        file_contents += f"# {self.function} {self.basis_set} em=GD3BJ scrf=(pcm,solvent=generic,read) geom=allcheck\n\n"
        file_contents += f"eps={self.epsilon}\n"
        file_contents += '\n\n'

        # Create and write the Gaussian input file
        file_path = os.path.join(self.work_dir, self.filename)
        with open(file_path, 'w') as file:
            file.write(file_contents)

    def execute_gaussian(self, log_file_path, command):
        with open(log_file_path, 'w') as log_file:
            subprocess.run(
                command,
                shell=True,
                check=True,
                text=True,
                stdout=log_file,
                stderr=None # 将 stderr 重定向到 stdout，即同一日志文件
            )

    def run_local(self,):

        prefix, _ = os.path.splitext(self.filename)
        log_file_path = f"{self.work_dir}/{prefix}.log"
        command = f"g16 {self.work_dir}/{self.filename}"

        try:
            self.execute_gaussian(log_file_path, command)
            print("Gaussian calculation completed successfully.")
            return 'success', f'{prefix}.log'
        except subprocess.CalledProcessError as e:

            print(f"Error executing Gaussian: {e}\n")

            with open(log_file_path, 'r') as log_file_read:
                lines = log_file_read.readlines()
                error_lines = [line for line in lines if "Error termination" in line]

                print("Gaussian error output:")
                for line in error_lines:
                    print(line.strip())

            return 'failed', f'{prefix}.log'

    def gen_slurm(self, script_name, job_name, nodes, ntasks_per_node, partition):
        slurm_script = PEMDSlurm(
            self.work_dir,
            script_name,
        )

        # slurm_script.add_command(f"module load conda && conda activate foyer")
        slurm_script.add_command(
            # f"xtb {self.xyz_filename} --opt --chrg={self.chg} --uhf={self.mult} --gfn {self.gfn}  --ceasefiles "
            # f"--namespace {self.work_dir}/{self.outfile_headname}"
            f"bash runGaussian.sh"
        )

        # Generate the SLURM script
        script_path = slurm_script.generate_script(
            job_name,
            nodes,
            ntasks_per_node,
            partition,
        )

        # print(f"XTB sybmit script generated successfually: {script_path}")
        return script_path



