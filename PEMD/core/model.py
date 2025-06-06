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
    gen_copolymer_3D,
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
    def gen_copolymer(
        work_dir: Path,
        poly_name_A: str,
        poly_name_B: str,
        smiles_A: str,
        smiles_B: str,
        *,
        mode: str | None = None,
        length: int | None = None,
        frac_A: float = 0.5,
        block_sizes: list[int] | None = None,
        sequence: list[str] | None = None,
        poly_resname: str = "MOL",
    ) -> str:
        """Generate a copolymer PDB file using a unified interface."""

        mol = gen_copolymer_3D(
            poly_name_A,
            poly_name_B,
            smiles_A,
            smiles_B,
            mode=mode,
            length=length,
            frac_A=frac_A,
            block_sizes=block_sizes,
            sequence=sequence,
        )

        poly_name = (
            poly_name_A if poly_name_A == poly_name_B else f"{poly_name_A}_{poly_name_B}"
        )

        if sequence is not None:
            seq_len = len(sequence)
        elif length is not None:
            seq_len = length
        elif block_sizes is not None:
            seq_len = sum(block_sizes)
        else:
            raise ValueError("length information missing")

        pdb_filename = f"{poly_name}_N{seq_len}.pdb"

        mol_to_pdb(
            work_dir=work_dir,
            mol=mol,
            poly_name=poly_name,
            poly_resname=poly_resname,
            pdb_filename=pdb_filename,
        )
        print(f"\nGenerated the pdb file {pdb_filename} successfully")
        return pdb_filename


    @staticmethod
    def gen_homopolymer(
        work_dir: Path,
        poly_name: str,
        smiles: str,
        length: int,
        poly_resname: str,
    ) -> str:

        return PEMDModel.gen_copolymer(
            work_dir=work_dir,
            poly_name_A=poly_name,
            poly_name_B=poly_name,
            smiles_A=smiles,
            smiles_B=smiles,
            mode="homopolymer",
            length=length,
            poly_resname=poly_resname,
        )


    def build_homopolymer(self) -> str:

        return PEMDModel.gen_homopolymer(
            work_dir=self.work_dir,
            poly_name=self.poly_name,
            smiles=self.repeating_unit,
            length=self.length_long,
            poly_resname=self.poly_resname
        )

    def build_copolymer(
        self,
        poly_name_A: str,
        poly_name_B: str,
        smiles_A: str,
        smiles_B: str,
        *,
        mode: str | None = None,
        length: int | None = None,
        frac_A: float = 0.5,
        block_sizes: list[int] | None = None,
        sequence: list[str] | None = None,
        poly_resname: str | None = None,
    ) -> str:
        """Build a copolymer specified by ``mode`` or ``sequence``."""

        if poly_resname is None:
            poly_resname = self.poly_resname

        return PEMDModel.gen_copolymer(
            work_dir=self.work_dir,
            poly_name_A=poly_name_A,
            poly_name_B=poly_name_B,
            smiles_A=smiles_A,
            smiles_B=smiles_B,
            mode=mode,
            length=length,
            frac_A=frac_A,
            block_sizes=block_sizes,
            sequence=sequence,
            poly_resname=poly_resname,
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

        return PEMDModel.gen_copolymer(
            work_dir=work_dir,
            poly_name_A=poly_name_A,
            poly_name_B=poly_name_B,
            smiles_A=smiles_A,
            smiles_B=smiles_B,
            mode="random",
            length=length,
            frac_A=frac_A,
        )


    @staticmethod
    def gen_alternating_copolymer(
            work_dir: Path,
            poly_name_A: str,
            poly_name_B: str,
            smiles_A: str,
            smiles_B: str,
            length: int
    ) -> str:

        return PEMDModel.gen_copolymer(
            work_dir=work_dir,
            poly_name_A=poly_name_A,
            poly_name_B=poly_name_B,
            smiles_A=smiles_A,
            smiles_B=smiles_B,
            mode="alternating",
            length=length,
        )


    @staticmethod
    def gen_block_copolymer(
            work_dir: Path,
            poly_name_A: str,
            poly_name_B: str,
            smiles_A: str,
            smiles_B: str,
            block_sizes: list[int]
    ) -> str:

        return PEMDModel.gen_copolymer(
            work_dir=work_dir,
            poly_name_A=poly_name_A,
            poly_name_B=poly_name_B,
            smiles_A=smiles_A,
            smiles_B=smiles_B,
            mode="block",
            block_sizes=block_sizes,
        )

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







