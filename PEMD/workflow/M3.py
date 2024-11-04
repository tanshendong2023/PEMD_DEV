# Import necessary libraries
from PEMD.analysis import residence_time
import os
import numpy as np
import MDAnalysis as mda
from tqdm.auto import tqdm


# Define constants and settings
work_dir = './'
data_tpr_file = 'nvt_prod.tpr'
dcd_xtc_file = 'unwrapped_traj.xtc'
select_atoms = {
    'cation': 'resname LIP and name Li',
    'anion': 'resname NSC and name OBT',
    'polymer': 'resname MOL and name O',
}
cutoff_radii = {
    'PEO': 3.575,
    'TFSI': 3.125,
}
run_start = 0
run_end = 80001  # step
dt = 0.001
dt_collection = 5000  # step
num_li = 50

# Define functions
def distance(x0, x1, box_length):
    delta = x1 - x0
    delta = np.where(delta > 0.5 * box_length, delta - box_length, delta)
    delta = np.where(delta < -0.5 * box_length, delta + box_length, delta)
    return delta


def load_data_traj(work_dir, data_tpr_file, dcd_xtc_file, select_atoms, run_start, run_end, dt, dt_collection, cutoff_radii):
    data_tpr_file_path = os.path.join(work_dir, data_tpr_file)
    dcd_xtc_file_path = os.path.join(work_dir, dcd_xtc_file)
    u = mda.Universe(data_tpr_file_path, dcd_xtc_file_path)
    li_atoms = u.select_atoms(select_atoms['cation'])
    oe_atoms = u.select_atoms(select_atoms['polymer'])
    poly_n = np.zeros((run_end - run_start, len(li_atoms)))
    for ts in tqdm(u.trajectory[run_start: run_end], desc='Processing trajectory'):
        box_size = ts.dimensions[0]
        for n, li in enumerate(li_atoms):
            distances_oe_vec = distance(oe_atoms.positions, li.position, box_size)
            distances_oe = np.linalg.norm(distances_oe_vec, axis=1)
            close_oe_index = np.where(distances_oe < cutoff_radii['PEO'])[0]
            if len(close_oe_index) > 0:
                o_resids = oe_atoms[close_oe_index].resids
                if np.all(o_resids == o_resids[0]):
                    poly_n[ts.frame, n] = o_resids[0]
                else:
                    poly_n[ts.frame, n] = -1
    times = np.arange(0, (run_end - run_start) * dt * dt_collection, dt * dt_collection, dtype=int)
    return poly_n, times


# Main function to load and process data
if __name__ == "__main__":
    poly_n, times = load_data_traj(work_dir, data_tpr_file, dcd_xtc_file, select_atoms, run_start, run_end, dt, dt_collection, cutoff_radii)
    total_hops, tau3 = residence_time.compute_tau3(poly_n, run_start, run_end, dt, dt_collection, num_li)
    # Define the path for the results file
    results_file_path = os.path.join(work_dir, 'tau3.txt')

    # Write results to a text file
    with open(results_file_path, 'w') as file:
        file.write(f"Calculated τ3: {tau3:.2f} ns\n")

    print(f"Results saved to {results_file_path}")

