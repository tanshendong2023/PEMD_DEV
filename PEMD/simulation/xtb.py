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
            chg=0,
            mult=1,
            gfn=2,
    ):

        self.work_dir = work_dir
        self.chg = chg
        self.mult = mult
        self.gfn = gfn

    def run_local(self, xyz_filename, outfile_headname):

        uhf = self.mult - 1 # unpaired electron number

        command = (
            f"xtb {xyz_filename} --opt --chrg={self.chg} --uhf={uhf} --gfn {self.gfn}  --ceasefiles "
            f"--namespace {self.work_dir}/{outfile_headname}"
        )

        try:
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            return result.stdout  # Return the standard output from the XTB command
        except subprocess.CalledProcessError as e:
            print(f"Error executing XTB: {e}")
            return e.stderr  # Return the error output if the command fails


