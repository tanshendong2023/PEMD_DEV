# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# core.analysis module
# ******************************************************************************


import os
import numpy as np
import MDAnalysis as mda
from typing import Tuple
from tqdm.auto import tqdm
from functools import partial
from functools import lru_cache
from multiprocessing import Pool
from PEMD.analysis.conductivity import calc_cond_msd, calc_conductivity
from PEMD.analysis.transfer_number import calc_transfer_number
from PEMD.analysis.coordination import calc_rdf_coord, obtain_rdf_coord
from PEMD.analysis.msd import (
    calc_slope_msd,
    create_position_arrays,
    calc_Lii_self,
    calc_Lii,
    calc_self_diffusion_coeff,
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
from PEMD.analysis.analysis_lib import (
    num_of_neighbor,
    pdb2mol,
    get_cluster_index,
    find_poly_match_subindex,
    get_cluster_withcap
)


class PEMDAnalysis:

    def __init__(self, run_wrap, run_unwrap, cation_name, anion_name, polymer_name, run_start, run_end, dt,
                 dt_collection, temperature):
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
    def from_gromacs(cls, work_dir, tpr_file, wrap_xtc_file, unwrap_xtc_file, cation_name, anion_name,
                     polymer_name, run_start, run_end, dt, dt_collection, temperature):

        tpr_path = os.path.join(work_dir, tpr_file)
        wrap_xtc_path = os.path.join(work_dir, wrap_xtc_file)
        unwrap_xtc_path = os.path.join(work_dir, unwrap_xtc_file)

        run_wrap = mda.Universe(tpr_path, wrap_xtc_path)
        run_unwrap = mda.Universe(tpr_path, unwrap_xtc_path)

        return cls(run_wrap, run_unwrap, cation_name, anion_name, polymer_name, run_start, run_end, dt, dt_collection,
                   temperature,)

    def times_range(self, end_time):

        t_total = end_time - self.run_start
        return np.arange(0, t_total * self.dt * self.dt_collection, self.dt * self.dt_collection, dtype=int)

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

    # calculate the conductivity
    @lru_cache(maxsize=128)
    def get_conductivity(self):

        return calc_conductivity(
            self.get_slope_msd(self.get_cond_array())[0],
            self.volume,
            self.temp
        )

    @lru_cache(maxsize=128)
    def get_ions_positions_array(self):

        cations_positions, anions_positions = create_position_arrays(
            self.run_unwrap,
            self.cations_unwrap,
            self.anions_unwrap,
            self.times,
            self.run_start,
        )
        return cations_positions, anions_positions

    def get_Lii_self_array(self, atom_positions):

        n_atoms = np.shape(atom_positions)[1]
        return calc_Lii_self(atom_positions, self.times) / n_atoms

    # calculate the self diffusion coefficient
    def get_self_diffusion_coefficient(self) -> Tuple[float, float]:

        cations_positions, anions_positions = self.get_ions_positions_array()
        slope_cations, time_range_cations = self.get_slope_msd(self.get_Lii_self_array(cations_positions))
        slope_anions, time_range_anions = self.get_slope_msd(self.get_Lii_self_array(anions_positions))

        D_cations = calc_self_diffusion_coeff(slope_cations)
        D_anions = calc_self_diffusion_coeff(slope_anions)
        return D_cations, D_anions

    # calculate the transfer number
    def get_transfer_number(self):

        cations_positions, anions_positions = self.get_ions_positions_array()
        slope_plusplus, time_range_plusplus = self.get_slope_msd(calc_Lii(cations_positions))
        slope_minusminus, time_range_minusminus = self.get_slope_msd(calc_Lii(anions_positions))
        return calc_transfer_number(
            slope_plusplus,
            slope_minusminus,
            self.temp,
            self.volume,
            self.get_conductivity()
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

    def get_coordination_number(self, group1_name, group2_name):

        bins, rdf, coord_number = self.get_rdf_coordination_array(group1_name, group2_name)
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
    def get_tau3(self):

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
    def get_tau1(self, time_window):

        return calc_tau1(
            self.get_tau3(),
            self.times_range(time_window),
            self.get_delta_n_square_array(time_window),
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
    def get_tauR(self):

        return fit_rouse_model(
            self.get_ms_endtoend_distance_array(),
            self.times,
            self.get_oe_msd_array(),
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
    def get_tau2(self, time_window):

        return fit_rouse_model(
            self.get_ms_endtoend_distance_array(),
            self.times_range(time_window),
            self.get_msd_M2_array(time_window),
            self.num_o_chain
        )

    @staticmethod
    def write_cluster(work_dir, tpr_file, wrap_xtc_file, center_atom_name, distance_dict, select_dict,run_start, run_end,
                      structure_code, max_number, write=True, write_freq=0.01, ):

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
    def write_poly_cluster_fragment(work_dir, pdb_filename, center_atom_name, poly_atom_name, poly_name, repeating_unit,
                                    length, outxyz_filename):

        mol = pdb2mol(
            work_dir,
            pdb_filename
        )

        (
            c_idx,
            center_atom_idx,
            selected_atom_idx
        ) = get_cluster_index(
            mol,
            center_atom_name,
            poly_atom_name,
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

        get_cluster_withcap(
            work_dir,
            mol,
            match_list,
            center_atom_idx,
            start_atom,
            end_atom,
            outxyz_filename
        )






























