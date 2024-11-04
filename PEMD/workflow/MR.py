#!/usr/bin/env python

import os
from PEMD.analysis import residence_time, coordination

work_dir = './'
data_tpr_file = 'nvt_prod.tpr'
dcd_xtc_file = 'nvt_prod_unwrap.xtc'

select_atoms = {
    'cation': 'resname LIP and name Li',
    'anion': 'resname NSC and name OBT',
    'polymer': 'resname MOL and name O',
}

# Load the trajectory
u = coordination.load_md_trajectory(work_dir, 'nvt_prod.tpr', 'nvt_prod.xtc')

# Select the atoms of interest
li_atoms = u.select_atoms('resname LIP and name Li')
peo_atoms = u.select_atoms('resname MOL and name O')
tfsi_atoms = u.select_atoms('resname NSC and name OBT')

# Perform RDF and coordination number calculation
bins_peo, rdf_peo, coord_num_peo = coordination.calculate_rdf_and_coordination(u, li_atoms, peo_atoms)
bins_tfsi, rdf_tfsi, coord_num_tfsi = coordination.calculate_rdf_and_coordination(u, li_atoms, tfsi_atoms)

# obtain the coordination number and first solvation shell distance
r_li_peo, y_rdf_peo, y_coord_peo = coordination.obtain_rdf_coord(bins_peo, rdf_peo, coord_num_peo)
r_li_tfsi, y_rdf_tfsi, y_coord_tfsi = coordination.obtain_rdf_coord(bins_tfsi, rdf_tfsi, coord_num_tfsi)

# setting the first solvation shell distance
cutoff_radii = {
    'PEO': r_li_peo,
    'TFSI': r_li_tfsi,
}

run_start = 0
run_end = 80001  # step
dt = 0.001
dt_collection = 5000 # step
num_li = 50
num_oe = 50
chains = 20

# calculate the mean square end to end distance
re_all = residence_time.ms_endtoend_distance(work_dir, data_tpr_file, dcd_xtc_file, run_start, dt_collection,
                                             chains, select_atoms, )

# obtain the ether oxygen position
atom_position, times = residence_time.get_ether_oxygen_position(work_dir, data_tpr_file, dcd_xtc_file,
                                                                select_atoms, run_start, run_end, dt,
                                                                dt_collection )

# calculate the MSD of ether oxygen
msd_oe = residence_time.compute_oe_msd(atom_position, times,)

# fit the tR by the rouse model
tauR, fit_curve = residence_time.compute_tR(re_all, times, num_oe, msd_oe)

# Define the path for the results file
results_file_path = os.path.join(work_dir, 'tauR.txt')

# Write results to a text file
with open(results_file_path, 'w') as file:
    file.write(f"Calculated τR: {tauR:.2f} ns\n")

print(f"Results saved to {results_file_path}")
