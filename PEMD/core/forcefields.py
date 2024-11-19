# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# core.forcefields Module
# ******************************************************************************

import os
import json
from PEMD.forcefields.ff_lib import (
    get_gaff2,
    get_oplsaa_xml,
    get_oplsaa_ligpargen,
    gen_ff_from_data
)
from PEMD.simulation.sim_lib import (
    apply_chg_to_poly,
    apply_chg_to_molecule
)
from PEMD.model.build import (
    gen_poly_smiles,
)


class Forcefield:

    def __init__(self):
        self.work_dir = None
        self.name = None
        self.resname = None
        self.repeating_unit = None
        self.leftcap = None
        self.rightcap = None
        self.length = None
        self.scale = None
        self.charge = None
        self.smiles = None
        self.terminal_cap = None

    @classmethod
    def from_json(cls, work_dir, json_file, mol_type='polymer'):
        instance = cls()
        instance.work_dir = work_dir

        json_path = os.path.join(work_dir, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                model_info = json.load(file)
        except FileNotFoundError:
            print(f"Error: JSON file {json_file} not found in {work_dir}.")
            return None
        except json.JSONDecodeError:
            print(f"Error: JSON file {json_file} is not a valid JSON.")
            return None

        data = model_info.get(mol_type)
        if data is None:
            print(f"Error: '{mol_type}' section not found in JSON file.")
            return None

        instance.name = data.get('compound')
        instance.resname = data.get('resname')
        instance.scale = data.get('scale')
        instance.charge = data.get('charge')

        if mol_type == 'polymer':
            instance.repeating_unit = data.get('repeating_unit')
            instance.leftcap = data.get('left_cap')
            instance.rightcap = data.get('right_cap')
            length = data.get('length')
            if length is not None:
                if isinstance(length, list):
                    instance.length = length[0]
                else:
                    instance.length = length
        else:
            instance.smiles = data.get('smiles')

        return instance

    @staticmethod
    def get_gaff2(gaff_dir, pdb_file, atom_typing):
        return get_gaff2(
            gaff_dir,
            pdb_file,
            atom_typing
        )

    def get_oplsaa_xml(self, xyz_file):
        return get_oplsaa_xml(
            self.work_dir,
            self.name,
            self.resname,
            xyz_file,
        )

    def apply_chg_to_poly(self, itp_file, resp_chg_df, end_repeating, ):
        poly_smi = gen_poly_smiles(
            self.name,
            self.repeating_unit,
            self.length,
            self.leftcap,
            self.rightcap,
        )
        return apply_chg_to_poly(
            self.work_dir,
            poly_smi,
            itp_file,
            resp_chg_df,
            self.repeating_unit,
            end_repeating,
            self.scale,
            self.charge,
        )

    def get_ff_from_data(self, ):
        return gen_ff_from_data(
            self.work_dir,
            self.name,
            self.scale,
            self.charge,
        )

    def get_oplsaa_ligpargen(self, chg_model, ):
        return get_oplsaa_ligpargen(
            self.work_dir,
            self.name,
            self.resname,
            self.charge,
            chg_model,
            self.smiles,
        )

    def apply_chg_to_molecule(self, itp_file, resp_chg_df,):
        return apply_chg_to_molecule(
            self.work_dir,
            itp_file,
            resp_chg_df,
            self.scale,
            self.charge,
        )


















