# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# core.model Module
# ******************************************************************************


import os
import json
from PEMD.model.build import (
    gen_poly_smiles,
)

class PEMDModel:
    def __init__(self, poly_name, repeating_unit, leftcap, rightcap, length_short, length_poly, ):
        """
        Initialize a PEMDModel instance.

        Args:
            poly_name (str): The name of the polymer.
            repeating_unit (str): The structure of the polymer's repeating unit.
            leftcap (str): The left cap structure of the polymer.
            rightcap (str): The right cap structure of the polymer.
            length_short (int): The length of the short polymer.
            length_poly (int): The length of the long polymer.
        """
        self.poly_name = poly_name
        self.repeating_unit = repeating_unit
        self.leftcap = leftcap
        self.rightcap = rightcap
        self.length_short = length_short
        self.length_poly = length_poly

    @classmethod
    def from_json(cls, work_dir, json_file):
        """
        Create a PEMDModel instance from a JSON file.

        Args:
            work_dir (str): The working directory where the JSON file is located.
            json_file (str): The name of the JSON file.

        Returns:
            PEMDModel: The created PEMDModel instance.
        """
        json_path = os.path.join(work_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as file:
            model_info = json.load(file)

        poly_name = model_info['polymer']['compound']
        repeating_unit = model_info['polymer']['repeating_unit']
        leftcap = model_info['polymer']['left_cap']
        rightcap = model_info['polymer']['right_cap']
        length_short = model_info['polymer']['length'][0]
        length_poly = model_info['polymer']['length'][1]

        return cls(poly_name, repeating_unit, leftcap, rightcap, length_short, length_poly)

    def gen_poly_smiles(self, short=False):
        """
        Generate the SMILES representation of the polymer.

        Args:
            short (bool): If True, generate the SMILES for the short polymer; if False, generate the SMILES for the long polymer.

        Returns:
            str: The generated SMILES string.
        """
        if short:
            return gen_poly_smiles(
                self.poly_name,
                self.repeating_unit,
                self.length_short,
                self.leftcap,
                self.rightcap,
            )
        else:
            return gen_poly_smiles(
                self.poly_name,
                self.repeating_unit,
                self.length_poly,
                self.leftcap,
                self.rightcap,
            )

