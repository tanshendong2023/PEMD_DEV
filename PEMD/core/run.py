# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

from PEMD.core.model import PEMDModel
from PEMD.model.build import (
    gen_poly_smiles,
    gen_poly_3D,
)
from PEMD.simulation.qm import (
    gen_conf_rdkit,
    opt_conf_xtb,
    opt_conf_gaussian,
    calc_resp_gaussian
)
from PEMD.simulation.sim_lib import (
    order_energy_xtb,
    order_energy_gaussian,
    read_xyz_file
)

from PEMD.simulation.md import (
    gen_poly_gmx_oplsaa,
)

class ConformerSearch:

    def __init__(self, work_dir, smiles,):
        self.work_dir = work_dir
        self.smiles = smiles

    def gen_conf_rdkit(self, max_conformers, top_n_MMFF, ):

        return gen_conf_rdkit(
            self.work_dir,
            self.smiles,
            max_conformers,
            top_n_MMFF,
        )

    def opt_conf_xtb(self, xyz_file, chg=0, mult=1, gfn=2, slurm=False, job_name='xtb', nodes=1, ntasks_per_node=64,
                     partition='standard',):

        return opt_conf_xtb(
            self.work_dir,
            xyz_file,
            chg,
            mult,
            gfn,
            slurm,
            job_name,
            nodes,
            ntasks_per_node,
            partition,
        )

    def order_energy_xtb(self, xyz_file, numconf):

            return order_energy_xtb(
                self.work_dir,
                xyz_file,
                numconf,
            )

    def opt_conf_gaussian(self, xyz_file, chg, mult, function, basis_set, epsilon, memory, job_name, nodes,
                          ntasks_per_node, partition,):

        return opt_conf_gaussian(
            self.work_dir,
            xyz_file,
            chg,
            mult,
            function,
            basis_set,
            epsilon,
            memory,
            job_name,
            nodes,
            ntasks_per_node,
            partition,
        )

    def order_energy_gaussian(self, numconf):

            return order_energy_gaussian(
                self.work_dir,
                numconf,
            )












    # def calc_resp_charge(
    #         self,
    #         epsilon,
    #         core,
    #         memory,
    #         function,
    #         basis_set,
    #         method, # resp1 or resp2
    # ):
    #     sorted_df = self.conformer_search(
    #         epsilon,
    #         core,
    #         memory,
    #         function,
    #         basis_set,
    #     )
    #
    #     return calc_resp_gaussian(
    #         sorted_df,
    #         epsilon,
    #         core,
    #         memory,
    #         method,
    #     )
    #
    # def build_polymer(self,):
    #
    #     return  gen_poly_3D(
    #         self.poly_name,
    #         self.length,
    #         self.gen_poly_smiles(),
    #     )
    #
    # def gen_polymer_force_field(self,):
    #
    #     gen_poly_gmx_oplsaa(
    #         self.poly_name,
    #         self.poly_resname,
    #         self.poly_scale,
    #         self.poly_charge,
    #         self.length,
    #     )








