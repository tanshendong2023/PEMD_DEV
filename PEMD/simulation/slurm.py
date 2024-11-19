# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# simulation.slurm module
# ******************************************************************************

import os
import subprocess

class PEMDSlurm:
    def __init__(
            self,
            work_dir,
            script_name="sub.script",
    ):
        """
        Initializes an instance of the PEMDSlurm class.

        Args:
            work_dir (str): The directory where the script and output files will be stored.
            script_name (str, optional): The name of the SLURM script file to be generated. Defaults to "sub.script".

        Attributes:
            commands (list): A list to store job commands that will be included in the SLURM script.
        """
        self.work_dir = work_dir
        self.script_name = script_name
        self.commands = []

    def add_command(self, command):
        """
        Adds a command to the list of commands to be included in the SLURM script.

        Args:
            command (str): The command to be added to the SLURM script. This should be a string representing a shell command.

        Raises:
            ValueError: If the provided command is not a string.
        """
        if not isinstance(command, str):
            raise ValueError("Command must be a string.")
        self.commands.append(command)

    def gen_header(self, job_name, nodes, ntasks_per_node, partition):
        """
        Generates the header section of the SLURM script, which defines job parameters.

        Args:
            job_name (str): The name of the SLURM job.
            nodes (int): The number of nodes required for the job.
            ntasks_per_node (int): The number of tasks (processes) per node.
            partition (str): The partition (queue) on which the job should run.

        Returns:
            str: A string representing the header section of the SLURM script, with each parameter line separated by a newline.
        """
        header_lines = [
            "#!/bin/bash",
            f"#SBATCH -J {job_name}",
            f"#SBATCH -N {nodes}",
            f"#SBATCH -n {ntasks_per_node}",
            f"#SBATCH -p {partition}",
            f"#SBATCH -o {self.work_dir}/slurm.%A.out",
        ]

        return "\n".join(header_lines)

    def generate_script(self, job_name, nodes, ntasks_per_node, partition):
        """
        Generates the SLURM script file with the specified job configuration and commands.

        Args:
            job_name (str): The name of the SLURM job.
            nodes (int): The number of nodes required for the job.
            ntasks_per_node (int): The number of tasks (processes) per node.
            partition (str): The partition (queue) on which the job should run.

        Returns:
            str: The path to the directory containing the generated SLURM script.
        """
        # First generating a header with job settings
        slurm_script_content = self.gen_header(job_name, nodes, ntasks_per_node, partition)
        # Appending all commands in `self.commands`
        slurm_script_content += "\n\n" + "\n".join(self.commands)
        # finally writing the complete script to a file at `self.work_dir/script_name`.
        script_file = os.path.join(self.work_dir, self.script_name)
        with open(script_file, "w") as f:
            f.write(slurm_script_content)
        script_path = os.path.dirname(script_file)

        return script_path

    def submit_job(self):
        """
        Submits the SLURM job script to the SLURM scheduler.

        Returns:
            str: The job ID assigned by SLURM if the job is successfully submitted.
            None: If there is an error in submitting the job, returns None and prints an error message.

        Description:
            This method submits the generated SLURM script located at `self.work_dir/script_name` using the `sbatch` command.
            If the submission is successful, it extracts and returns the job ID from the output. In case of a failure,
            it captures the error and prints an error message.

        Raises:
            subprocess.CalledProcessError: If the `sbatch` command fails, with error details printed.
        """
        script_path = os.path.join(self.work_dir, self.script_name)
        try:
            result = subprocess.run(f"sbatch {script_path}", shell=True, check=True, text=True, capture_output=True)
            job_id = result.stdout.strip().split()[-1]
            print(f"SLURM job submitted: Job ID is {job_id}")
            return job_id
        except subprocess.CalledProcessError as e:
            print(f"Error submitting SLURM job: {e}")
            return None





