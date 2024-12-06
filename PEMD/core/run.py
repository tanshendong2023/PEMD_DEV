# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import os
import json
from PEMD.simulation.qm import (
    gen_conf_rdkit,
    opt_conf_xtb,
    opt_conf_gaussian,
    calc_resp_gaussian,
    RESP_fit_Multiwfn,
)
from PEMD.simulation.md import (
    relax_poly_chain,
    anneal_amorph_poly,
    run_gmx_prod
)


class QMRun:

    def __init__(self):
        self.work_dir = None
        self.name = None
        self.smiles = None

    @classmethod
    def from_json(cls, work_dir, json_file, mol_type='polymer', external_smiles=None):
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

        if mol_type == 'polymer':
            # 当 mol_type 是 'polymer'，从外部传入 smiles
            if external_smiles is not None:
                instance.smiles = external_smiles
            else:
                print(f"No external SMILES provided for polymer type '{mol_type}'.")
        else:
            # 当 mol_type 不是 'polymer'，从 JSON 文件中读取 smiles
            instance.smiles = data.get('smiles', '')
            if not instance.smiles:
                print(f"Warning: SMILES not found for molecule type '{mol_type}'.")

        return instance

    def conformer_search(self, max_conformers, top_n_MMFF, top_n_xtb, top_n_qm, chg, mult, gfn, function, basis_set, epsilon,
                         core, memory, gaucontinue=False):

        # Generate conformers using RDKit
        xyz_file_MMFF = gen_conf_rdkit(
            self.work_dir,
            self.name,
            self.smiles,
            max_conformers,
            top_n_MMFF,
        )

        # Optimize conformers using XTB
        xyz_file_xtb = opt_conf_xtb(
            self.work_dir,
            xyz_file_MMFF,
            self.name,
            top_n_xtb,
            chg,
            mult,
            gfn,
        )

        # Optimize conformers using Gaussian
        return opt_conf_gaussian(
            self.work_dir,
            self.name,
            xyz_file_xtb,
            top_n_qm,
            chg,
            mult,
            function,
            basis_set,
            epsilon,
            core,
            memory,
            gaucontinue
        )


    def resp_chg_fitting(self, xyz_file, chg, mult, function, basis_set, epsilon, core, mem, method):
        calc_resp_gaussian(
            self.work_dir,
            self.name,
            xyz_file,
            chg,
            mult,
            function,
            basis_set,
            epsilon,
            core,
            mem,
        )

        return RESP_fit_Multiwfn(
            self.work_dir,
            self.name,
            method,
            delta=0.5
        )


class MDRun:
    def __init__(self, work_dir, molecules):
        self.work_dir = work_dir
        self.molecules = molecules

    @classmethod
    def from_json(cls, work_dir, json_file):

        json_path = os.path.join(work_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as file:
            model_info = json.load(file)

        molecules = []
        for key, value in model_info.items():
            name = value["compound"]
            number = value["numbers"]
            resname = value["resname"]

            molecule = {
                "name": name,
                "number": number,
                "resname": resname,
            }
            molecules.append(molecule)

        return cls(work_dir, molecules)

    @staticmethod
    def relax_poly_chain(work_dir, pdb_file, core, atom_typing = 'pysimm'):
        return relax_poly_chain(
            work_dir,
            pdb_file,
            core,
            atom_typing
        )

    def anneal_amorph_poly(self, temperature, T_high_increase, anneal_rate, anneal_npoints, packmol_pdb, density, add_length, gpu=False):

        anneal_amorph_poly(
            self.work_dir,
            self.molecules,
            temperature,
            T_high_increase,
            anneal_rate,
            anneal_npoints,
            packmol_pdb,
            density,
            add_length,
            gpu,
        )

    def run_gmx_prod(self, temperature, nstep_ns, gpu=False):

        run_gmx_prod(
            self.work_dir,
            self.molecules,
            temperature,
            nstep_ns,
            gpu
        )


















