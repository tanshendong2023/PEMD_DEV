# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# core.model Module
# ******************************************************************************


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
    name: str
    resname: str
    repeating_unit: str
    leftcap: str
    rightcap: str
    length_short: int
    length_long: int
    smiles_A: str = ""
    smiles_B: str = ""
    mode: str | None = None
    block_sizes: list[int] | None = None
    frac_A: float = 0.5
    molecule_list: dict = field(default_factory=dict)


    @classmethod
    def from_json(cls, work_dir, json_file):
        work_dir = Path(work_dir)
        json_path = work_dir / json_file
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                model_info = json.load(file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"JSON file '{json_file}' not found in '{work_dir}'.") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON file '{json_file}' is not valid JSON: {exc}") from exc

        polymer_info = model_info.get('polymer', {})

        name = polymer_info.get('compound', '')
        resname = polymer_info.get('resname', '')
        repeating_unit = polymer_info.get('repeating_unit', '')
        leftcap = polymer_info.get('left_cap', '')
        rightcap = polymer_info.get('right_cap', '')
        length_list = polymer_info.get('length', [0, 0])
        length_short = length_list[0] if len(length_list) > 0 else 0
        length_long = length_list[1] if len(length_list) > 1 else 0
        smiles_A = polymer_info.get("smiles_A", None)
        smiles_B = polymer_info.get("smiles_B", None)
        mode = polymer_info.get("mode", None)
        block_sizes = polymer_info.get("block_sizes", None)
        frac_A = polymer_info.get("frac_A", 0.5)

        molecule_list = {}
        for category, details in model_info.items():
            if isinstance(details, dict):
                compound = details.get('compound')
                numbers = details.get('numbers')
                if compound is not None and numbers is not None:
                    molecule_list[compound] = numbers

        return cls(
            work_dir,
            name,
            resname,
            repeating_unit,
            leftcap,
            rightcap,
            length_short,
            length_long,
            smiles_A,
            smiles_B,
            mode,
            block_sizes,
            frac_A,
            molecule_list
        )


    @staticmethod
    def copolymer(
        work_dir: Path,
        smiles_A: str,
        smiles_B: str,
        *,
        mode: str | None = None,
        length: int | None = None,
        frac_A: float = 0.5,
        block_sizes: list[int] | None = None,
        sequence: list[str] | None = None,
        name: str = "PE",
        resname: str = "MOL",
    ) -> str:
        """Generate a copolymer PDB file using a unified interface."""

        mol = gen_copolymer_3D(
            smiles_A,
            smiles_B,
            name=name,
            mode=mode,
            length=length,
            frac_A=frac_A,
            block_sizes=block_sizes,
            sequence=sequence,
        )

        if sequence is not None:
            seq_len = len(sequence)
        elif length is not None:
            seq_len = length
        elif block_sizes is not None:
            seq_len = sum(block_sizes)
        else:
            raise ValueError("length information missing")

        pdb_filename = f"{name}_N{seq_len}.pdb"

        mol_to_pdb(
            work_dir=work_dir,
            mol=mol,
            name=name,
            resname=resname,
            pdb_filename=pdb_filename,
        )
        print(f"\nGenerated the pdb file {pdb_filename} successfully")
        return pdb_filename


    @classmethod
    def copolymer_from_json(
        cls,
        work_dir: Path,
        json_file: str
    ) -> str:
        instance = cls.from_json(work_dir, json_file)

        pdb_file = cls.copolymer(
            work_dir=instance.work_dir,
            smiles_A=instance.smiles_A,
            smiles_B=instance.smiles_B,
            mode=instance.mode,
            length=instance.length_long,
            frac_A=instance.frac_A,
            block_sizes=instance.block_sizes,
            name=instance.name,
            resname=instance.resname
        )
        return pdb_file


    @staticmethod
    def homopolymer(
        work_dir: Path,
        smiles: str,
        length: int,
        name: str = "PE",
        resname: str = "MOL",
    ) -> str:

        return PEMDModel.copolymer(
            work_dir=work_dir,
            name=name,
            smiles_A=smiles,
            smiles_B=smiles,
            mode="homopolymer",
            length=length,
            resname=resname,
        )

    @classmethod
    def homopolymer_from_json(
            cls,
            work_dir:
            Path,
            json_file: str
    ) -> str:

        instance = cls.from_json(work_dir, json_file)

        pdb_file = cls.homopolymer(
            work_dir=instance.work_dir,
            name=instance.name,
            smiles=instance.repeating_unit,
            length=instance.length_long,
            resname=instance.resname
        )

        return pdb_file

    @staticmethod
    def amorphous_cell(
        work_dir: Path,
        molecule_list: dict,
        density: float,
        add_length: int,
        packinp_name: str,
        packpdb_name: str,
    ) -> None:

        work_dir = Path(work_dir)
        MD_dir = work_dir / "MD_dir"
        run = PEMDPackmol(
            MD_dir,
            molecule_list,
            density,
            add_length,
            packinp_name,
            packpdb_name
        )

        run.generate_input_file()
        run.run_local()

        print("Amorphous structure generated.")

    @classmethod
    def amorphous_cell_from_json(
        cls,
        work_dir: Path,
        json_file: str,
        density: float,
        add_length: int,
        packinp_name: str,
        packpdb_name: str,
    ) -> None:

        work_dir = Path(work_dir)
        instance = cls.from_json(work_dir, json_file)
        MD_dir = work_dir / "MD_dir"
        run = PEMDPackmol(
            MD_dir,
            instance.molecule_list,
            density,
            add_length,
            packinp_name,
            packpdb_name
        )

        run.generate_input_file()
        run.run_local()

        print("Amorphous structure generated.")







