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
        core=32,
        mem='64GB',
        chg=0,
        mult=1,
        function='B3LYP',
        basis_set='6-311+g(d,p)',
        epsilon=5.0,
    ):
        self.work_dir = work_dir
        self.core = core
        self.mem = mem
        self.chg = chg
        self.mult = mult
        self.function = function
        self.basis_set = basis_set
        self.epsilon = epsilon

    def generate_input_file(self, gaussian_dir, structure, filename):
        # 构建 Gaussian 输入文件内容
        file_contents = f"%nprocshared={self.core}\n"
        file_contents += f"%mem={self.mem}\n"
        file_contents += f"# opt freq {self.function} {self.basis_set} em=GD3BJ scrf=(pcm,solvent=generic,read)\n\n"
        file_contents += 'qm calculation\n\n'
        file_contents += f'{self.chg} {self.mult}\n'  # 电荷和多重度

        # 添加原子坐标
        for atom in structure['atoms']:
            atom_parts = atom.split()
            # 确保包含原子符号和三个坐标
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
        file_path = os.path.join(gaussian_dir, filename)
        with open(file_path, 'w') as file:
            file.write(file_contents)
        # print(f"Gaussian input file generated: {file_path}")

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



