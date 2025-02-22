# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# core.model Module
# ******************************************************************************


import os
import json
from rdkit import Chem

from PEMD.model import model_lib
from PEMD.model.packmol import PEMDPackmol
from PEMD.model.build import (
    gen_poly_smiles,
    gen_copoly_smiles,
    gen_poly_3D,
)


class PEMDModel:
    def __init__(self, work_dir, poly_name, poly_resname, repeating_unit, leftcap, rightcap, length_short, length_long, molecule_list):
        """
        Initialize a PEMDModel instance.

        Parameters:
        poly_name (str): The name of the polymer.
        repeating_unit (str): The structure of the polymer's repeating unit.
        leftcap (str): The left cap structure of the polymer.
        rightcap (str): The right cap structure of the polymer.
        length_short (int): The length of the short polymer.
        length_poly (int): The length of the long polymer.
        """

        self.work_dir = work_dir
        self.poly_name = poly_name
        self.poly_resname = poly_resname
        self.repeating_unit = repeating_unit
        self.leftcap = leftcap
        self.rightcap = rightcap
        self.length_short = length_short
        self.length_long = length_long
        self.molecule_list = molecule_list

    @classmethod
    def from_json(cls, work_dir, json_file):
        """
        Create a PEMDModel instance from a JSON file.

        Parameters:
        work_dir (str): The working directory where the JSON file is located.
        json_file (str): The name of the JSON file.

        Returns:
        PEMDModel: The created PEMDModel instance.
        """

        json_path = os.path.join(work_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as file:
            model_info = json.load(file)

        poly_name = model_info['polymer']['compound']
        poly_resname = model_info['polymer']['resname']
        repeating_unit = model_info['polymer']['repeating_unit']
        leftcap = model_info['polymer']['left_cap']
        rightcap = model_info['polymer']['right_cap']
        length_short = model_info['polymer']['length'][0]
        length_long = model_info['polymer']['length'][1]

        molecule_list = {}
        for category, details in model_info.items():
            compound = details.get('compound')
            numbers = details.get('numbers')
            molecule_list[compound] = numbers

        return cls(work_dir, poly_name, poly_resname, repeating_unit, leftcap, rightcap, length_short, length_long, molecule_list)

    def gen_oligomer_smiles(self,):
        """
        Generate the SMILES representation of the polymer.
        Returns:
        str: The generated SMILES string.
        """

        return gen_poly_smiles(
            self.poly_name,
            self.repeating_unit,
            self.length_short,
            self.leftcap,
            self.rightcap,
        )

    def gen_homopolymer(self):
        """
        Generate the structure of the homo polymer.

        Parameters:
            core: The number of kernels used for segmental relaxation using LAMMPS.

        Returns:
            str: The name of relaxed polymer structure file.
        """

        mol = gen_poly_3D(
            self.poly_name,
            self.repeating_unit,
            self.length_long,
        )

        pdb_filename = f"{self.poly_name}_N{self.length_long}.pdb"
        pdb_file = os.path.join(self.work_dir, pdb_filename)
        Chem.MolToXYZFile(mol, 'mid.xyz', confId=0)
        model_lib.convert_xyz_to_pdb('mid.xyz', pdb_file, self.poly_name, self.poly_resname)
        os.remove('mid.xyz')

        print(f"\nGenerated the pdb file {pdb_filename} successfully")

        return pdb_filename

    # def gen_flex_alter_copoly(self, max_retries=50):
    #     """
    #     Generate the SMILES representation of the alternating copolymer.
    #
    #     Parameters:
    #         core: The number of kernels used for segment relaxation using LAMMPS.
    #
    #     Returns:
    #         str: The name of relaxed polymer structure file.
    #     """
    #
    #     # Obtain smiles of copolymerized polymer segments.
    #     unit_smiles = gen_copoly_smiles(
    #         self.poly_name,
    #         self.repeating_unit,
    #         x_length = 1,
    #         y_length = 1,
    #     )
    #
    #     smiles = gen_poly_smiles(
    #         self.poly_name,
    #         unit_smiles,
    #         self.length_long,
    #         self.leftcap,
    #         self.rightcap,
    #     )
    #
    #     return gen_poly_3D(
    #         self.work_dir,
    #         self.poly_name,
    #         self.poly_resname,
    #         self.repeating_unit,
    #         self.length_long,
    #         smiles,
    #         max_retries
    #     )
    #
    # def gen_flex_block_copoly(self, x_length=1, y_length=1, max_retries=50):
    #     """
    #     Generate the SMILES representation of the block copolymer.
    #
    #     Parameters:
    #         core: The number of kernels used for segment relaxation using LAMMPS.
    #         x_length: The length of first type unit in one block.
    #         y_length: The length of second type unit in one block.
    #     Returns:
    #         str: The name of relaxed polymer structure file.
    #     """
    #
    #     # Obtain smiles of copolymerized polymer segments.
    #     unit_smiles = gen_copoly_smiles(
    #         self.poly_name,
    #         self.repeating_unit,
    #         x_length,
    #         y_length,
    #     )
    #
    #     smiles = gen_poly_smiles(
    #         self.poly_name,
    #         unit_smiles,
    #         self.length_long,
    #         self.leftcap,
    #         self.rightcap,
    #     )
    #
    #     return gen_poly_3D(
    #         self.work_dir,
    #         self.poly_name,
    #         self.poly_resname,
    #         self.repeating_unit,
    #         self.length_long,
    #         smiles,
    #         max_retries
    #     )

    def gen_amorphous_structure(self, density, add_length, packinp_name, packpdb_name,):
        MD_dir = os.path.join(self.work_dir, 'MD_dir')
        run = PEMDPackmol(
            MD_dir,
            self.molecule_list,
            density,
            add_length,
            packinp_name,
            packpdb_name
        )
        run.generate_input_file()
        run.run_local()







