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
        """
        Initialize an instance of the ConformerSearch class.

        Args:
            work_dir (str): The working directory where generated files and results will be stored.
            smiles (str): The SMILES representation of the molecule used for conformer generation.
        """
        self.work_dir = work_dir
        self.smiles = smiles

    def gen_conf_rdkit(self, max_conformers, top_n_MMFF, ):
        """
        Generate max_conformers conformers and select top_n_MMFF conformers to save their coordinates into seperate and merged XYZ files.

        Args:
            max_conformers (int): Maximum number of conformers to generate.
            top_n_MMFF (int): Number of  conformers to save.
        """
        return gen_conf_rdkit(
            self.work_dir,
            self.smiles,
            max_conformers,
            top_n_MMFF,
        )

    def opt_conf_xtb(self, xyz_file, chg=0, mult=1, gfn=2, slurm=False, job_name='xtb', nodes=1, ntasks_per_node=64,
                     partition='standard',):
        """
        This function is used to optimize conformers using xTB calculation to obtain their energy.

        Args:
            work_dir (str): Path to the working directory.
            xyz_file (str): Input XYZ file containing conformer structures.
            chg (int): Molecular charge. Default is 0.
            mult (int): Spin multiplicity. Default is 1.
            gfn (int): Parametrization of the xTB method. Default is GFN2.
            slurm (bool): If True, generates a SLURM submission script instead of running locally. Default is False.
            job_name (str): Job name for SLURM submission. Default is 'xtb'.
            nodes (int): Number of nodes for SLURM. Default is 1.
            ntasks_per_node (int): Number of tasks per node for SLURM. Default is 64.
            partition (str): SLURM partition to use. Default is 'standard'.

        Description:
            The function performs xTB energy optimization.
            If `slurm` is False, it runs locally and saves conformers in a merged XYZ file.
            If `slurm` is True, it generates a SLURM job script for batch submission.
        """
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
        """
        This function reads an XYZ file containing molecular structures and their energies,
        sorts the structures based on their energy values, and writes the lowest `numconf`
        energy structures to a new XYZ file.

        Args:
            work_dir (str): The directory where the output sorted XYZ file will be saved.
            xyz_file (str): The path to the input XYZ file containing molecular structures.
            numconf (int): The number of lowest energy structures to be selected and saved.
        """
        return order_energy_xtb(
            self.work_dir,
            xyz_file,
            numconf,
        )

    def opt_conf_gaussian(self, xyz_file, chg, mult, function, basis_set, epsilon, memory, job_name, nodes,
                          ntasks_per_node, partition,):
        """
        The  function is used to read molecular configurations from an input xyz file
        and generates the corresponding Gaussian input files and a submission script.

        Args:
            xyz_file: Path to the input xyz file containing molecular structures.
            chg: Charge of the molecule, default is 0.
            mult: Spin multiplicity of the molecule, default is 1.
            function: DFT functional to use, default is 'B3LYP'.
            basis_set: Basis set for Gaussian, default is '6-311+g(d,p)'.
            epsilon: Dielectric constant for the implicit solvation model, default is 5.0.
            mem: Memory allocation for Gaussian, default is '64GB'.
            job_name: Job name, default is 'g16'.
            nodes: Number of compute nodes allocated, default is 1.
            ntasks_per_node: Number of tasks per node, default is 64.
            partition: Name of the Slurm job queue to submit the job, default is 'standard'.

        Output:
            Prints the path to the generated Gaussian submission script.
        """
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
        """
        The function processes Gaussian log files to extract optimized energies and final atomic structures
        from multiple conformations. It sorts the conformations by energy, selects the numconf lowest-energy structures,
        and writes them  to an output .xyz file.

        Args:
            work_dir (str): The directory where the output sorted XYZ file will be saved.
            numconf (int): The number of lowest energy structures to be selected and saved.
        """
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








