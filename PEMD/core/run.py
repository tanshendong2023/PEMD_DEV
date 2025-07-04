# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************


import os
import json

from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

from PEMD.simulation.qm import (
    gen_conf_rdkit,
    conf_xtb,
    qm_gaussian,
    calc_resp_gaussian,
    RESP_fit_Multiwfn,
)
from PEMD.simulation.md import (
    relax_poly_chain,
    annealing,
    run_gmx_prod
)


@dataclass
class QMRun:
    work_dir: Path
    name: str
    smiles: str

    @staticmethod
    def qm_gaussian(
        work_dir: Path | str,
        xyz_file: str,
        gjf_filename: str,
        *,
        charge: float = 0,
        mult: int = 1,
        function: str = 'B3LYP',
        basis_set: str = '6-31+g(d,p)',
        epsilon: float = 5.0,
        core: int = 64,
        memory: str = '128GB',
        chk: bool = False,
        optimize: bool = True,
        multi_step: bool = False,
        max_attempts: int = 1,
        toxyz: bool = True,
        top_n_qm: int = 4,
    ):
        return qm_gaussian(
            work_dir=work_dir,
            xyz_file=xyz_file,
            gjf_filename=gjf_filename,
            charge=charge,
            mult=mult,
            function=function,
            basis_set=basis_set,
            epsilon=epsilon,
            core=core,
            memory=memory,
            chk=chk,
            optimize=optimize,
            multi_step=multi_step,
            max_attempts=max_attempts,
            toxyz=toxyz,
            top_n_qm=top_n_qm
        )


    @staticmethod
    def conformer_search(
        work_dir: Path,
        *,
        smiles: str | None = None,
        pdb_file: str | None = None,
        max_conformers: int = 1000,
        top_n_MMFF: int = 100,
        top_n_xtb: int = 8,
        top_n_qm: int = 4,
        charge: float = 0,
        mult: int = 1,
        gfn: str = 'gfn2',
        function: str = 'b3lyp',
        basis_set: str = '6-31g*',
        epsilon: float = 2.0,
        core: int = 32,
        memory: str = '64GB',
    ):

        # Generate conformers using RDKit
        xyz_file_MMFF = gen_conf_rdkit(
            work_dir=work_dir,
            max_conformers=max_conformers,
            top_n_MMFF=top_n_MMFF,
            smiles=smiles,
            pdb_file=pdb_file,
        )

        # Optimize conformers using XTB
        xyz_file_xtb = conf_xtb(
            work_dir,
            xyz_file_MMFF,
            top_n_xtb=top_n_xtb,
            charge=charge,
            mult=mult,
            gfn=gfn,
            optimize=True
        )

        # Optimize conformers using Gaussian
        return qm_gaussian(
            work_dir,
            xyz_file_xtb,
            gjf_filename="conf",
            charge=charge,
            mult= mult,
            function=function,
            basis_set=basis_set,
            epsilon=epsilon,
            core=core,
            memory=memory,
            optimize=True,
            multi_step=True,
            max_attempts=2,
            toxyz=True,
            top_n_qm=top_n_qm,
        )


    @staticmethod
    def resp_chg_fitting(
        work_dir: Path,
        xyz_file: str,
        charge: float = 0,
        mult: int = 1,
        function: str = 'b3lyp',
        basis_set: str = '6-311+g(d,p)',
        epsilon: float = 5.0,
        core: int = 32,
        memory: str = '64GB',
        method: str = 'resp2',
    ):

        calc_resp_gaussian(
            work_dir,
            xyz_file,
            charge,
            mult,
            function,
            basis_set,
            epsilon,
            core,
            memory,
        )

        return RESP_fit_Multiwfn(
            work_dir,
            method,
            delta=0.5
        )


@dataclass
class MDRun:
    work_dir: Path
    molecules: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_json(cls, work_dir, json_file):

        json_path = os.path.join(work_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as file:
            model_info = json.load(file)

        molecules = []
        for key, value in model_info.items():
            name = value["name"]
            number = value["numbers"]
            resname = value["resname"]

            molecule = {
                "name": name,
                "number": number,
                "resname": resname,
            }
            molecules.append(molecule)

        return cls(work_dir, molecules,)


    @staticmethod
    def relax_poly_chain(
        work_dir: Path,
        name: str,
        resname: str,
        pdb_file: str,
        temperature: int = 1000,
        gpu: bool = False,
    ):

        return relax_poly_chain(
            work_dir,
            name,
            resname,
            pdb_file,
            temperature,
            gpu,
        )


    @staticmethod
    def annealing(
        work_dir: Path,
        molecules: List[Dict[str, Any]],
        temperature: int = 298,
        T_high_increase: int = 300,
        anneal_rate: float = 0.05,
        anneal_npoints: int = 5,
        packmol_pdb: str = "pack_cell.pdb",
        density: float = 0.8,
        add_length: int = 10,
        gpu: bool = False,
    ):

        annealing(
            work_dir,
            molecules,
            temperature,
            T_high_increase,
            anneal_rate,
            anneal_npoints,
            packmol_pdb,
            density,
            add_length,
            gpu,
        )


    @classmethod
    def annealing_from_json(
        cls,
        work_dir: Path,
        json_file: str,
        temperature: int = 298,
        T_high_increase: int = 300,
        anneal_rate: float = 0.05,
        anneal_npoints: int = 5,
        packmol_pdb: str = "pack_cell.pdb",
        density: float = 0.8,
        add_length: int = 10,
        gpu: bool = False,
    ):

        work_dir = Path(work_dir)
        instance = cls.from_json(work_dir, json_file)

        annealing(
            work_dir,
            instance.molecules,
            temperature,
            T_high_increase,
            anneal_rate,
            anneal_npoints,
            packmol_pdb,
            density,
            add_length,
            gpu,
        )


    @staticmethod
    def production(
        work_dir: Path,
        molecules: List[Dict[str, Any]],
        temperature: int = 298,
        nstep_ns: int = 200,   # 200 ns
        gpu=False
    ):

        run_gmx_prod(
            work_dir,
            molecules,
            temperature,
            nstep_ns,
            gpu
        )


    @classmethod
    def production_from_json(
        cls,
        work_dir: Path,
        json_file: str,
        temperature: int = 298,
        nstep_ns: int = 200,   # 200 ns
        gpu=False
    ):

        work_dir = Path(work_dir)
        instance = cls.from_json(work_dir, json_file)

        run_gmx_prod(
            instance.work_dir,
            instance.molecules,
            temperature,
            nstep_ns,
            gpu
        )
















