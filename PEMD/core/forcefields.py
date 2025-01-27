# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# core.forcefields Module
# ******************************************************************************

import os
import json
from PEMD.forcefields.ff_lib import (
    get_oplsaa_xml,
    get_xml_ligpargen,
    get_oplsaa_ligpargen,
    gen_ff_from_data
)
from PEMD.forcefields.ff_lib import (
    apply_chg_to_poly,
    apply_chg_to_molecule
)
from PEMD.model.build import (
    gen_poly_3D,
    gen_poly_smiles
)


class Forcefield:

    def __init__(self):
        self.work_dir = None
        self.name = None
        self.resname = None
        self.repeating_unit = None
        self.leftcap = None
        self.rightcap = None
        self.length_short = None
        self.length_long = None
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
            instance.length_short = length[0]
            instance.length_long = length[1]
        else:
            instance.smiles = data.get('smiles')

        return instance

    def get_oplsaa_xml(self, xml, pdb_file, chg_model = 'CM1A'):

        if xml == "ligpargen":
            get_xml_ligpargen(
                self.work_dir,
                self.name,
                self.resname,
                self.repeating_unit,
                self.charge,
                chg_model,
            )

            return get_oplsaa_xml(
                self.work_dir,
                self.name,
                pdb_file,
                xml = "ligpargen",
            )

        else:
            return get_oplsaa_xml(
                self.work_dir,
                self.name,
                pdb_file,
                xml = "database",
            )

    def apply_chg_to_poly(self, itp_file, resp_chg_df, end_repeating, max_retries=500):

        # mol_short = gen_poly_3D(
        #     self.name,
        #     self.repeating_unit,
        #     self.length_short,
        #     max_retries
        # )
        #
        mol_long = gen_poly_3D(
            self.name,
            self.repeating_unit,
            self.length_long,
            max_retries
        )

        smiles_short = gen_poly_smiles(
            self.name,
            self.repeating_unit,
            self.length_short,
            self.leftcap,
            self.rightcap,
        )

        # smiles_long = gen_poly_smiles(
        #     self.name,
        #     self.repeating_unit,
        #     self.length_long,
        #     self.leftcap,
        #     self.rightcap,
        # )

        return apply_chg_to_poly(
            self.work_dir,
            # mol_short,
            smiles_short,
            mol_long,
            # smiles_long,
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

    def get_oplsaa_ligpargen(self, chg_model = 'CM1A', ):
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


















