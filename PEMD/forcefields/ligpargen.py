# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************


import subprocess


class PEMDLigpargen:
    def __init__(
            self,
            work_dir,
            poly_name,
            poly_resname,
            chg,
            chg_model,
            smiles=None,
            filename=None,
    ):
        """
        Initialize the PEMDLigpargen class.

        Parameters:
        - poly_name (str): Name of the molecule, used for output file naming.
        - poly_resname (str): Residue name of the molecule, used in output files.
        - chg (int): Net charge of the molecule.
        - chg_model (str): Charge model to use, 'CM1A' or 'CM1A-LBCC'.
        - smiles (str, optional): SMILES string of the molecule.
        - mol_file (str, optional): Path to the molecule file (PDB, MOL, MOL2, etc.).
        """
        self.work_dir = work_dir
        self.smiles = smiles
        self.filename = filename
        self.poly_name = poly_name
        self.poly_resname = poly_resname
        self.chg = chg
        self.chg_model = chg_model

    def run_local(self):

        # Determine the input option
        if self.smiles and self.filename:
            raise ValueError("Please provide only one of SMILES string or molecule file, not both.")
        elif self.smiles:
            input_option = f"-s '{self.smiles}'"
        elif self.filename:
            input_option = f"-i '{self.filename}'"
        else:
            raise ValueError("You must provide either a SMILES string or a molecule file as input.")

        # Build the command line
        command = (
            f"ligpargen {input_option} -n {self.poly_name} -p {self.work_dir} "
            f"-r {self.poly_resname} -c {self.chg} -cgen {self.chg_model}"
        )

        try:
            # Execute the command
            result = subprocess.run(
                command, shell=True, check=True, text=True, capture_output=True
            )
            print("LigParGen ran successfully.")
            return result.stdout  # Return the standard output
        except subprocess.CalledProcessError as e:
            print(f"Error executing LigParGen: {e.stderr}")
            return e.stderr  # Return the error output if the command fails


