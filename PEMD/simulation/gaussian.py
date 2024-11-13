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
        """
        Initializes the PEMDGaussian object with the given parameters.

        Args:
            work_dir (str): The directory where input/output files will be saved.
            core (int, optional): The number of CPU cores for the job. Default is 32.
            mem (str, optional): The memory allocation for the job. Default is '64GB'.
            chg (int, optional): The charge of the system. Default is 0.
            mult (int, optional): The multiplicity of the system. Default is 1.
            function (str, optional): The functional to use in the Gaussian calculation. Default is 'B3LYP'.
            basis_set (str, optional): The basis set for the calculation. Default is '6-311+g(d,p)'.
            epsilon (float, optional): The dielectric constant for the system. Default is 5.0.
        """
        self.work_dir = work_dir
        self.core = core
        self.mem = mem
        self.chg = chg
        self.mult = mult
        self.function = function
        self.basis_set = basis_set
        self.epsilon = epsilon

    def generate_input_file(self, gaussian_dir, structure, filename):
        """
        Generates a Gaussian input file based on the given structure and other parameters.

        Args:
            gaussian_dir (str): The directory where the input file will be saved.
            structure (dict): The molecular structure containing atom names and their coordinates.
            filename (str): The name of the output Gaussian input file.
        """
        # Set Gaussian input file parameters.
        self.gaussian_dir = gaussian_dir
        file_contents = f"%nprocshared={self.core}\n"
        file_contents += f"%mem={self.mem}\n"
        file_contents += f"# opt freq {self.function} {self.basis_set} em=GD3BJ scrf=(pcm,solvent=generic,read)\n\n"
        file_contents += 'qm calculation\n\n'
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
        file_contents += "epsinf=1.9\n"
        file_contents += '\n\n'

        # Create and write the Gaussian input file
        file_path = os.path.join(gaussian_dir, filename)
        with open(file_path, 'w') as file:
            file.write(file_contents)

    def generate_input_file_resp(self, gaussian_dir, structure, filename, idx):
        """
        Generates a Gaussian input file based on the given structure and other parameters.

        Args:
            gaussian_dir (str): The directory where the input file will be saved.
            structure (dict): The molecular structure containing atom names and their coordinates.
            filename (str): The name of the output Gaussian input file.
        """
        # Set Gaussian input file parameters.
        self.gaussian_dir = gaussian_dir
        file_contents = f"%nprocshared={self.core}\n"
        file_contents += f"%mem={self.mem}\n"
        file_contents += f"%chk={gaussian_dir}/resp_conf_{idx}.chk\n"
        file_contents += f"# opt {self.function} {self.basis_set} em=GD3BJ scrf=(pcm,solvent=generic,read)\n\n"
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
        file_contents += f"%oldchk={gaussian_dir}/resp_conf_{idx}.chk\n"
        file_contents += f"%chk={gaussian_dir}/SP_gas_conf_{idx}.chk\n"
        file_contents += f"# {self.function} {self.basis_set} em=GD3BJ geom=allcheck\n\n"

        file_contents += "--link1--\n"
        file_contents += f"nprocshared={self.core}\n"
        file_contents += f"%mem={self.mem}\n"
        file_contents += f"%oldchk={gaussian_dir}/resp_conf_{idx}.chk\n"
        file_contents += f"%chk={gaussian_dir}/SP_solv_conf_{idx}.chk\n"
        file_contents += f"# {self.function} {self.basis_set} em=GD3BJ scrf=(pcm,solvent=generic,read) geom=allcheck\n\n"
        file_contents += f"eps={self.epsilon}\n"
        file_contents += '\n\n'

        # Create and write the Gaussian input file
        file_path = os.path.join(gaussian_dir, filename)
        with open(file_path, 'w') as file:
            file.write(file_contents)

    def gen_slurm(self, script_name, job_name, nodes, ntasks_per_node, partition):
        """
        Generates a SLURM script to submit a job for Gaussian calculation.

        Args:
            script_name (str): The name of the SLURM script file.
            job_name (str): The name of the job for SLURM.
            nodes (int): The number of nodes for the SLURM job.
            ntasks_per_node (int): The number of tasks per node.
            partition (str): The SLURM partition to submit the job to.

        Returns:
            script_path (str): The path of the generated SLURM script.
        """
        slurm_script = PEMDSlurm(
            self.work_dir,
            script_name,
        )

        # Using script "runGaussian.sh" to perform Gaussian calculation.
        # Need to add its path as an environment variable firstly.
        slurm_script.add_command(
            f"bash runGaussian.sh {self.gaussian_dir}"
        )

        # Generate the SLURM script
        script_path = slurm_script.generate_script(
            job_name,
            nodes,
            ntasks_per_node,
            partition,
        )
        return script_path



