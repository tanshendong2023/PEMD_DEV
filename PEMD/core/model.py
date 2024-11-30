# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# core.model Module
# ******************************************************************************


import os
import json
from PEMD.model.build import (
    gen_poly_smiles,
    gen_poly_3D,
)
from PEMD.simulation.md import (
    relax_poly_chain
)
from PEMD.model.packmol import PEMDPackmol


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

        Parameters:
        short (bool): If True, generate the SMILES for the short polymer; if False, generate the SMILES for the long polymer.

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

    def gen_flex_poly(self, core, atom_typing = 'pysimm'):

        smiles = gen_poly_smiles(
            self.poly_name,
            self.repeating_unit,
            self.length_long,
            self.leftcap,
            self.rightcap,
        )

        pdb_file = gen_poly_3D(
            self.work_dir,
            self.poly_name,
            self.poly_resname,
            self.length_long,
            smiles,
        )

        return relax_poly_chain(
            self.work_dir,
            pdb_file,
            core,
            atom_typing
        )

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







