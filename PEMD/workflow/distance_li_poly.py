import numpy as np
from tqdm.auto import tqdm
from PEMD.analysis import coordination


def calculate_average_min_distance(work_dir, tpr_file, xtc_file, run_start, run_end, output_file, cutoff):

    run = coordination.load_md_trajectory(work_dir, tpr_file, xtc_file)
    cations = run.select_atoms('resname LIP and name Li')
    polymers = run.select_atoms('resname MOL and name O')

    def distance(x0, x1, box_length):
        delta = x1 - x0
        delta -= np.round(delta / box_length) * box_length
        return np.linalg.norm(delta, axis=1)

    min_distance_list = []

    # Iterate over the specified trajectory range
    for ts in tqdm(run.trajectory[run_start: run_end], desc='Processing trajectory'):
        box_size = ts.dimensions[0:3]
        min_distances = []

        all_o_positions = polymers.positions
        all_o_resids = polymers.resids

        for li in cations:
            distances_oe = distance(all_o_positions, li.position, box_size)
            close_oe_index = np.where(distances_oe < cutoff)[0]
            bound_poly_resids = set(all_o_resids[close_oe_index])

            unbound_indices = np.where(~np.isin(all_o_resids, list(bound_poly_resids)))[0]

            if unbound_indices.size > 0:
                unbound_o_positions = all_o_positions[unbound_indices]
                distances_oe = distance(unbound_o_positions, li.position, box_size)
                min_distance = np.min(distances_oe)
                min_distances.append(min_distance)

        if min_distances:
            ave_min_distance = np.mean(min_distances)
            min_distance_list.append(ave_min_distance)

    # Calculate the overall average minimum distance
    if min_distance_list:
        overall_ave_min_distance = np.mean(min_distance_list)
        with open(output_file, 'w') as f:
            f.write(f"Overall average minimum distance: {overall_ave_min_distance}\n")
    else:
        with open(output_file, 'w') as f:
            f.write("No relevant unbound oxygens found during the simulation period.\n")
        print("No relevant unbound oxygens found during the simulation period.")


# Example usage:
work_dir = './'
tpr_file = 'nvt_prod.tpr'
xtc_file = 'nvt_prod_unwrap.xtc'
xtc_wrap_file = 'nvt_prod.xtc'
run_start = 0
run_end = 80001
output_file = 'ave_min_distance.txt'
run_wrap = coordination.load_md_trajectory(work_dir, tpr_file, xtc_wrap_file)
volume = run_wrap.coord.volume

cations_wrap = run_wrap.select_atoms('resname LIP and name Li')
polymers_wrap = run_wrap.select_atoms('resname MOL and name O')

bins_peo, rdf_peo, coord_num_peo = coordination.calc_rdf_coord(cations_wrap, polymers_wrap, volume)
r_li_peo, y_coord_peo = coordination.obtain_rdf_coord(bins_peo, rdf_peo, coord_num_peo)
calculate_average_min_distance(work_dir, tpr_file, xtc_file, run_start, run_end, output_file, r_li_peo)

