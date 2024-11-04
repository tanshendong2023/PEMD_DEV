# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import os
import subprocess
from PEMD.simulation.slurm import PEMDSlurm

class PEMDXtb:
    def __init__(
            self,
            work_dir,
            # xyz_filename,
            # outfile_headname,
            chg=0,
            mult=1,
            gfn=2
    ):

        self.work_dir = work_dir
        # self.xyz_filename = xyz_filename
        # self.xyz_filepath = os.path.join(work_dir, xyz_filename)
        self.chg = chg
        self.mult = mult
        self.gfn = gfn
        # self.outfile_headname = outfile_headname

    def run_local(self, xyz_filename, xtb_dir, outfile_headname):

        command = (
            f"xtb {xyz_filename} --opt --chrg={self.chg} --uhf={self.mult} --gfn {self.gfn}  --ceasefiles "
            f"--namespace {xtb_dir}/{outfile_headname}"
        )

        try:
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            return result.stdout  # Return the standard output from the XTB command
        except subprocess.CalledProcessError as e:
            print(f"Error executing XTB: {e}")
            return e.stderr  # Return the error output if the command fails

    def gen_slurm(self, script_name, job_name, nodes, ntasks_per_node, partition):

        slurm_script = PEMDSlurm(
            self.work_dir,
            script_name,
        )

        # slurm_script.add_command(f"module load conda && conda activate foyer")
        slurm_script.add_command(
            # f"xtb {self.xyz_filename} --opt --chrg={self.chg} --uhf={self.mult} --gfn {self.gfn}  --ceasefiles "
            # f"--namespace {self.work_dir}/{self.outfile_headname}"
            f"bash runxTB.sh {self.chg} {self.mult} {self.gfn}"
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

    # def run_slurm(self, script_name):
    #
    #     slurm_script = self.gen_slurm(script_name)
    #     job_id = slurm_script.submit_job()
    #
    #     return job_id