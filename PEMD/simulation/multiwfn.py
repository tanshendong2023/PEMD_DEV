# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import subprocess
import os
import glob

class PEMDMultiwfn:
    def __init__(
            self,
            work_dir,
    ):
        """
        Initialize the PEMDMultiwfn object.

        Args:
            work_dir (str): The path to the working directory where XTB output files will be saved.
        """
        self.work_dir = work_dir

    def resp_run_local(self, method):
        """
        Execute Multiwfn locally to perform RESP fitting procedure locally.

        Args:
            method (str): The method to be used ('resp' or 'resp2').
                         'resp' method uses 'SP_solv_conf_*.fchk' files.
                         'resp2' method uses both 'SP_solv_conf*.fchk' and 'SP_gas_conf*.fchk' files.

        Raises:
            ValueError: If an unsupported method is provided.
        """
        # If using RESP method, calculate charges of 'SP_solv_conf_*.fchk' files.
        if method == 'resp':
            fchk_files = [os.path.abspath(path) for path in
                          glob.glob(os.path.join(self.work_dir, 'SP_solv_conf_*.fchk'))]
        # If using RESP2 method, calculate charges of 'SP_solv_conf_*.fchk' files and 'SP_gas_conf*.fchk' files.
        elif method == 'resp2':
            fchk_files = [os.path.abspath(path) for path in
                          glob.glob(os.path.join(self.work_dir, 'SP_solv_conf*.fchk')) + glob.glob(
                              os.path.join(self.work_dir, 'SP_gas_conf*.fchk'))]
        else:
            raise ValueError("Unsupported method. Please choose 'resp' or 'resp2'.")

        # Applying "calcRESP.sh" script to calculate charges.
        for idx, fchk_file in enumerate(fchk_files):
            command = ["bash", "calcRESP.sh", f"{fchk_file}"]
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # Output command execution results.
            if process.returncode == 0:
                print(f"RESP fitting for the {idx + 1}-th structure has been successfully completed.")
            else:
                print(f"RESP fitting for the {idx + 1}-th structure failed : {process.stderr}")

