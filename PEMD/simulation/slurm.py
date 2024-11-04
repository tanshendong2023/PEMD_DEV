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

        self.work_dir = work_dir
        self.script_name = script_name
        self.commands = []

    def add_command(self, command):

        self.commands.append(command)

    def gen_header(self, job_name, nodes, ntasks_per_node, partition):

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

        slurm_script_content = self.gen_header(job_name, nodes, ntasks_per_node, partition)
        slurm_script_content += "\n\n" + "\n".join(self.commands)

        script_file = os.path.join(self.work_dir, self.script_name)
        with open(script_file, "w") as f:
            f.write(slurm_script_content)

        script_path = os.path.dirname(script_file)

        return script_path

    def submit_job(self):

        script_path = os.path.join(self.work_dir, self.script_name)
        try:
            result = subprocess.run(f"sbatch {script_path}", shell=True, check=True, text=True, capture_output=True)
            job_id = result.stdout.strip().split()[-1]
            print(f"SLURM job submitted: Job ID is {job_id}")
            return job_id
        except subprocess.CalledProcessError as e:
            print(f"Error submitting SLURM job: {e}")
            return None





