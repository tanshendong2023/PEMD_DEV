# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# core.model Module
# ******************************************************************************


import os
import json

from pathlib import Path
from PEMD.model.packmol import PEMDPackmol
from dataclasses import dataclass, field
from PEMD.model.build import (
    gen_homopolymer_3D,
    gen_random_copolymer_3D,
    gen_alternating_copolymer_3D,
    gen_block_copolymer_3D,
    mol_to_pdb,
)


@dataclass
class PEMDModel:
    work_dir: Path
    poly_name: str
    poly_resname: str
    repeating_unit: str
    leftcap: str
    rightcap: str
    length_short: int
    length_long: int
    molecule_list: dict = field(default_factory=dict)


    @classmethod
    def from_json(cls, work_dir, json_file):
        json_path = os.path.join(work_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as file:
            model_info = json.load(file)

        polymer_info = model_info.get('polymer', {})

        poly_name = polymer_info.get('compound', '')
        poly_resname = polymer_info.get('resname', '')
        repeating_unit = polymer_info.get('repeating_unit', '')
        leftcap = polymer_info.get('left_cap', '')
        rightcap = polymer_info.get('right_cap', '')
        length_list = polymer_info.get('length', [0, 0])
        length_short = length_list[0] if len(length_list) > 0 else 0
        length_long = length_list[1] if len(length_list) > 1 else 0

        molecule_list = {}
        for category, details in model_info.items():
            if isinstance(details, dict):
                compound = details.get('compound')
                numbers = details.get('numbers')
                if compound is not None and numbers is not None:
                    molecule_list[compound] = numbers

        return cls(work_dir, poly_name, poly_resname, repeating_unit, leftcap, rightcap, length_short, length_long, molecule_list)


    @staticmethod
    def gen_homopolymer(
        work_dir: Path,
        poly_name: str,
        smiles: str,
        length: int,
        poly_resname: str
    ) -> str:

        mol = gen_homopolymer_3D(
            poly_name,
            smiles,
            length
        )

        pdb_filename = f"{poly_name}_N{length}.pdb"

        mol_to_pdb(
            work_dir=work_dir,
            mol=mol,
            poly_name=poly_name,
            poly_resname=poly_resname,
            pdb_filename=pdb_filename
        )
        print(f"\nGenerated the pdb file {pdb_filename} successfully")
        return pdb_filename


    def build_homopolymer(self) -> str:

        return PEMDModel.gen_homopolymer(
            work_dir=self.work_dir,
            poly_name=self.poly_name,
            smiles=self.repeating_unit,
            length=self.length_long,
            poly_resname=self.poly_resname
        )


    @staticmethod
    def gen_random_copolymer(
            work_dir: Path,
            poly_name_A: str,
            poly_name_B: str,
            smiles_A: str,
            smiles_B: str,
            length: int,
            frac_A: float
    ) -> str:

        mol = gen_random_copolymer_3D(
            poly_name_A,
            poly_name_B,
            smiles_A,
            smiles_B,
            length,
            frac_A,
        )

        poly_name = f"{poly_name_A}_{poly_name_B}"
        pdb_filename = f"{poly_name}_N{length}.pdb"

        mol_to_pdb(
            work_dir,
            mol,
            poly_name,
            poly_resname = "MOL",
            pdb_filename = pdb_filename,
        )

        print(f"\nGenerated the pdb file {pdb_filename} successfully")

        return pdb_filename


    @staticmethod
    def gen_alternating_copolymer(
            work_dir: Path,
            poly_name_A: str,
            poly_name_B: str,
            smiles_A: str,
            smiles_B: str,
            length: int
    ) -> str:

        mol = gen_alternating_copolymer_3D(
            poly_name_A,
            poly_name_B,
            smiles_A,
            smiles_B,
            length,
        )

        poly_name = f"{poly_name_A}_{poly_name_B}"
        pdb_filename = f"{poly_name}_N{length}.pdb"

        mol_to_pdb(
            work_dir,
            mol,
            poly_name,
            poly_resname = "MOL",
            pdb_filename = pdb_filename,
        )

        print(f"\nGenerated the pdb file {pdb_filename} successfully")

        return pdb_filename


    @staticmethod
    def gen_block_copolymer(
            work_dir: Path,
            poly_name_A: str,
            poly_name_B: str,
            smiles_A: str,
            smiles_B: str,
            block_sizes: list[int]
    ) -> str:

        mol = gen_block_copolymer_3D(
            poly_name_A,
            poly_name_B,
            smiles_A,
            smiles_B,
            block_sizes,
        )

        poly_name = f"{poly_name_A}_{poly_name_B}"
        length = sum(block_sizes)
        pdb_filename = f"{poly_name}_N{length}.pdb"

        mol_to_pdb(
            work_dir,
            mol,
            poly_name,
            poly_resname = "MOL",
            pdb_filename = pdb_filename,
        )

        print(f"\nGenerated the pdb file {pdb_filename} successfully")

        return pdb_filename

    def gen_amorphous_structure(
        self,
        density: float,
        add_length: int,
        packinp_name: str,
        packpdb_name: str,
    ) -> None:

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

        print("Amorphous structure generated.")







