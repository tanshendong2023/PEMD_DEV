# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# core.analysis module
# ******************************************************************************


import os
import pandas as pd
import numpy as np
import MDAnalysis as mda

from tqdm.auto import tqdm
from functools import partial
from functools import lru_cache
from multiprocessing import Pool

from PEMD.analysis.conductivity import calc_cond_msd, calc_conductivity
from PEMD.analysis.transfer_number import calc_transfer_number
from PEMD.analysis.msd import (
    calc_slope_msd,
    create_position_arrays,
    calc_Lii_self,
    calc_Lii,
    calc_self_diffusion_coeff,
    plot_msd
)
from PEMD.analysis.polymer_ion_dynamics import (
    process_traj,
    calc_tau3,
    calc_delta_n_square,
    calc_tau1,
    ms_endtoend_distance,
    fit_rouse_model,
    calc_msd_M2
)
from PEMD.analysis.coordination import (
    calc_rdf_coord,
    obtain_rdf_coord,
    plot_rdf_coordination,
    num_of_neighbor,
    pdb2mol,
    get_cluster_index,
    find_poly_match_subindex,
    get_cluster_withcap
)
from PEMD.analysis.esw import esw


class PEMDAnalysis:

    def __init__(
        self,
        run_wrap,
        run_unwrap,
        cation_name,
        anion_name,
        polymer_name,
        run_start,
        run_end,
        dt,
        dt_collection,
        temperature,
    ):

        self.run_wrap = run_wrap
        self.run_unwrap = run_unwrap
        self.run_start = run_start
        self.run_end = run_end
        self.dt = dt
        self.dt_collection = dt_collection
        self.temp = temperature
        self.times = self.times_range(self.run_end)
        self.cation_name = cation_name
        self.anion_name = anion_name
        self.polymer_name = polymer_name
        self.cations_unwrap = run_unwrap.select_atoms(self.cation_name)
        self.anions_unwrap = run_unwrap.select_atoms(self.anion_name)
        self.polymers_unwrap = run_unwrap.select_atoms(self.polymer_name)
        self.volume = self.run_unwrap.coord.volume
        self.box_size = self.run_unwrap.dimensions[0]
        self.num_cation = len(self.cations_unwrap)
        self.num_o_polymer = len(self.polymers_unwrap)
        self.num_chain = len(np.unique(self.polymers_unwrap.resids))
        self.num_o_chain = int(self.num_o_polymer // self.num_chain)


    @classmethod
    def from_gromacs(
        cls,
        work_dir,
        tpr_file,
        wrap_xtc_file,
        unwrap_xtc_file,
        cation_name,
        anion_name,
        polymer_name,
        run_start,
        run_end,
        dt,
        dt_collection,
        temperature,
    ):

        tpr_path = os.path.join(work_dir, tpr_file)
        wrap_xtc_path = os.path.join(work_dir, wrap_xtc_file)
        unwrap_xtc_path = os.path.join(work_dir, unwrap_xtc_file)

        run_wrap = mda.Universe(tpr_path, wrap_xtc_path)
        run_unwrap = mda.Universe(tpr_path, unwrap_xtc_path)

        return cls(
            run_wrap,
            run_unwrap,
            cation_name,
            anion_name,
            polymer_name,
            run_start,
            run_end,
            dt,
            dt_collection,
            temperature,
        )


    def times_range(self, end_time):

        t_total = end_time - self.run_start
        return np.arange(
            0,
            t_total * self.dt * self.dt_collection,
            self.dt * self.dt_collection,
            dtype=int
        )


    def get_cond_array(self):

        return calc_cond_msd(
            self.run_unwrap,
            self.cations_unwrap,
            self.anions_unwrap,
            self.run_start,
        )


    def get_slope_msd(self, msd_array, interval_time=10000, step_size=10):

        slope, time_range = calc_slope_msd(
            self.times,
            msd_array,
            self.dt_collection,
            self.dt,
            interval_time,
            step_size
        )
        return slope, time_range


    @lru_cache(maxsize=128)
    def conductivity(self, save_csv=False, plot=False, ):

        msd_array = self.get_cond_array()
        slope, time_ranges = self.get_slope_msd(msd_array)

        if save_csv:
            df = pd.DataFrame({
                "time": self.times,
                "msd": msd_array
            })
            df.to_csv('msd.csv', index=False)

        if plot:
            plot_msd(
                msd_array,
                self.times,
                self.dt_collection,
                self.dt,
                time_ranges=time_ranges,
            )

        return calc_conductivity(
            slope,
            self.volume,
            self.temp
        )


    @lru_cache(maxsize=128)
    def get_ions_positions_array(self, mols_unwrap=None):

        mols_positions = create_position_arrays(
            self.run_unwrap,
            mols_unwrap,
            self.times,
            self.run_start,
        )
        return mols_positions


    def get_Lii_self_array(self, atom_positions):

        n_atoms = np.shape(atom_positions)[1]
        return calc_Lii_self(atom_positions, self.times) / n_atoms


    # calculate the self diffusion coefficient
    def diffusion_coefficient(self, atoms, save_csv=False, plot=False, ):

        atoms_unwrap = self.run_unwrap.select_atoms(atoms)
        atoms_positions = self.get_ions_positions_array(atoms_unwrap)

        msd_array = self.get_Lii_self_array(atoms_positions)
        slope, time_ranges = self.get_slope_msd(msd_array)

        if save_csv:
            df = pd.DataFrame({
                "time": self.times,
                "msd": msd_array
            })
            df.to_csv('msd.csv', index=False)

        if plot:
            plot_msd(
                msd_array,
                self.times,
                self.dt_collection,
                self.dt,
                time_ranges=time_ranges,
            )

        D = calc_self_diffusion_coeff(slope)

        return D


    # calculate the transfer number
    def transfer_number(self):

        cations_positions = self.get_ions_positions_array(self.cations_unwrap)
        anions_positions = self.get_ions_positions_array(self.anions_unwrap)

        slope_plusplus, time_range_plusplus = self.get_slope_msd(calc_Lii(cations_positions))
        slope_minusminus, time_range_minusminus = self.get_slope_msd(calc_Lii(anions_positions))

        return calc_transfer_number(
            slope_plusplus,
            slope_minusminus,
            self.temp,
            self.volume,
            self.conductivity()
        )


    def get_rdf_coordination_array(self, group1_name, group2_name):

        group1 = self.run_wrap.select_atoms(group1_name)
        group2 = self.run_wrap.select_atoms(group2_name)
        bins, rdf, coord_number = calc_rdf_coord(
            group1,
            group2,
            self.volume
        )
        return bins, rdf, coord_number


    def coordination_number(self, group1_name, group2_name, save_csv=False, plot=False):

        bins, rdf, coord_number = self.get_rdf_coordination_array(group1_name, group2_name)

        if save_csv:
            df = pd.DataFrame({
                "bins": bins,
                "rdf": rdf,
                "coordination_number": coord_number
            })
            df.to_csv('coord.csv', index=False)

        if plot:
            plot_rdf_coordination(
                bins,
                rdf,
                coord_number,
            )

        y_coord = obtain_rdf_coord(bins, rdf, coord_number)[1]

        return y_coord


    @lru_cache(maxsize=128)
    def get_cutoff_radius(self, group1_name, group2_name):

        bins, rdf, coord_number = self.get_rdf_coordination_array(group1_name, group2_name)
        x_val = obtain_rdf_coord(bins, rdf, coord_number)[0]

        return x_val


    @lru_cache(maxsize=128)
    def get_poly_array(self):

        return process_traj(
            self.run_unwrap,
            self.times,
            self.run_start,
            self.run_end,
            self.num_cation,
            self.num_o_polymer,
            self.get_cutoff_radius(self.cation_name, self.polymer_name),
            self.cations_unwrap,
            self.polymers_unwrap,
            self.box_size,
            self.num_chain,
            self.num_o_chain
        )


    # calculate the tau3
    @lru_cache(maxsize=128)
    def tau3(self):

        return calc_tau3(
            self.dt,
            self.dt_collection,
            self.num_cation,
            self.run_start,
            self.run_end,
            self.get_poly_array()[1]
        )


    def get_delta_n_square_array(self, time_window,):

        # msd = []
        poly_o_n = self.get_poly_array()[0]
        poly_n = self.get_poly_array()[1]

        partial_calc_delta_n_square = partial(calc_delta_n_square, poly_o_n=poly_o_n, poly_n=poly_n,
                                              run_start=self.run_start, run_end=self.run_end)

        with Pool() as pool:
            msd = list(tqdm(pool.imap(partial_calc_delta_n_square, range(time_window)), total=time_window))
        return np.array(msd)


    # calculate the tau1
    def tau1(self, time_window, save_csv=False, plot=False):

        times = self.times_range(time_window)
        msd_array = self.get_delta_n_square_array(time_window)

        if save_csv:
            df = pd.DataFrame({
                "time": times,
                "msd": msd_array
            })
            df.to_csv('msd.csv', index=False)

        if plot:
            plot_msd(
                msd_array,
                times,
                self.dt_collection,
                self.dt,
            )

        return calc_tau1(
            self.tau3(),
            times,
            msd_array,
            self.num_o_chain
        )


    @lru_cache(maxsize=128)
    def get_ms_endtoend_distance_array(self):

        return ms_endtoend_distance(
            self.run_unwrap,
            self.num_chain,
            self.polymers_unwrap,
            self.box_size,
            self.run_start,
            self.run_end,
        )


    def get_oe_msd_array(self):

        return self.get_Lii_self_array(
            self.get_poly_array()[3]
        )


    # calculate the tauR
    def tauR(self, save_csv=False, plot=False):

        msd_array = self.get_oe_msd_array()

        if save_csv:
            df = pd.DataFrame({
                "time": self.times,
                "msd": msd_array
            })
            df.to_csv('msd.csv', index=False)

        if plot:
            plot_msd(
                msd_array,
                self.times,
                self.dt_collection,
                self.dt,
            )

        return fit_rouse_model(
            self.get_ms_endtoend_distance_array(),
            self.times,
            msd_array,
            self.num_o_chain
        )


    def get_msd_M2_array(self, time_window):

        poly_o_n = self.get_poly_array()[0]
        bound_o_n = self.get_poly_array()[2]
        poly_o_positions = self.get_poly_array()[3]

        partial_calc_msd_M2 = partial(calc_msd_M2, poly_o_positions=poly_o_positions, poly_o_n=poly_o_n,
                                      bound_o_n=bound_o_n, run_start=self.run_start, run_end=self.run_end)

        with Pool() as pool:
            msd = list(tqdm(pool.imap(partial_calc_msd_M2, range(time_window)), total=time_window))
        return np.array(msd)


    # calculate the tau2
    def tau2(self, time_window, save_csv=False, plot=False):

        times = self.times_range(time_window)
        msd_array = self.get_msd_M2_array(time_window)

        if save_csv:
            df = pd.DataFrame({
                "time": times,
                "msd": msd_array
            })
            df.to_csv('msd.csv', index=False)

        if plot:
            plot_msd(
                msd_array,
                times,
                self.dt_collection,
                self.dt,
            )

        return fit_rouse_model(
            self.get_ms_endtoend_distance_array(),
            times,
            msd_array,
            self.num_o_chain
        )


    @staticmethod
    def write_cluster(
        work_dir: str,
        tpr_file: str,
        wrap_xtc_file: str,
        center_atom_name: str,
        distance_dict: dict[str, float],
        select_dict: dict[str, str],
        run_start: int = 0,
        run_end: int = 80000,
        structure_code: int = 1,
        max_number: int = 100,
        write: bool = True,
        write_freq: float = 0.01,
    ):

        tpr_path = os.path.join(work_dir, tpr_file)
        wrap_xtc_path = os.path.join(work_dir, wrap_xtc_file)
        nvt_run = mda.Universe(tpr_path, wrap_xtc_path)

        num_of_neighbor(
            work_dir,
            nvt_run,
            center_atom_name,
            distance_dict,
            select_dict,
            run_start,
            run_end,
            write,
            structure_code,
            write_freq,
            max_number
        )


    @staticmethod
    def write_cluster_polymer(
        work_dir: str,
        tpr_file: str,
        wrap_xtc_file: str,
        center_atom_name: str,
        distance_dict: dict[str, float],
        select_dict: dict[str, str],
        poly_name: str,
        repeating_unit: str,
        length: int = 3,
        run_start: int = 0,
        run_end: int = 80000,
        structure_code: int = 1,
        max_number: int = 100,
        write_freq: float = 0.01,
    ):

        PEMDAnalysis.write_cluster(
            work_dir,
            tpr_file,
            wrap_xtc_file,
            center_atom_name,
            distance_dict,
            select_dict,
            run_start,
            run_end,
            structure_code,
            max_number,
            write = True,
            write_freq = write_freq,
        )

        cluster_dir = os.path.join(work_dir, "cluster_dir")
        pdb_files = sorted([f for f in os.listdir(cluster_dir) if f.endswith(".pdb")])

        for i, pdb_file in enumerate(pdb_files):

            try:
                mol = pdb2mol(
                    cluster_dir,
                    pdb_file
                )

                (
                    c_idx,
                    center_atom_idx,
                    selected_atom_idx,
                    other_atom_idx,
                ) = get_cluster_index(
                    mol,
                    center_atom_name,
                    select_dict,
                    distance_dict,
                    poly_name,
                    repeating_unit,
                    length,
                )

                (
                    match_list,
                    start_atom,
                    end_atom
                ) = find_poly_match_subindex(
                    poly_name,
                    repeating_unit,
                    length,
                    mol,
                    selected_atom_idx,
                    c_idx,
                )

                outxyz_file = f'num_{i}_frag.xyz'
                get_cluster_withcap(
                    cluster_dir,
                    mol,
                    match_list,
                    center_atom_idx,
                    other_atom_idx,
                    start_atom,
                    end_atom,
                    outxyz_file
                )
            except Exception as e:
                print(f"Error processing file {pdb_file}: {e}")
                continue


    @staticmethod
    def esw(
        G_init: float,
        *,
        G_oxid: float = None,
        G_red: float = None,
        output: str = 'all',
        unit: str = 'hartree',  # hartree, eV, V, mV to kcal/mol
    ):

        return esw(
            G_init,
            G_oxid=G_oxid,
            G_red=G_red,
            output=output,
            unit=unit
        )









