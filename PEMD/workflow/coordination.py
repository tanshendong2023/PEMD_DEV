
import numpy as np
from PEMD.analysis import coordination

work_dir = './'
data_tpr_file='nvt_prod.tpr'
dcd_xtc_file='nvt_prod.xtc'

# Load the trajectory
u = coordination.load_md_trajectory(work_dir, data_tpr_file, dcd_xtc_file)
volume = u.coord.volume

# Select the atoms of interest
li_atoms = u.select_atoms('resname LIP and name Li')
peo_atoms = u.select_atoms('resname MOL and name O')
tfsi_atoms = u.select_atoms('resname NSC and name OBT')
sn_atoms = u.select_atoms('resname SN and name N')

# Perform RDF and coordination number calculation
bins_peo, rdf_peo, coord_num_peo = coordination.calc_rdf_coord(li_atoms, peo_atoms, volume)
bins_tfsi, rdf_tfsi, coord_num_tfsi = coordination.calc_rdf_coord(li_atoms, tfsi_atoms, volume)
bins_sn, rdf_sn, coord_num_sn = coordination.calc_rdf_coord(li_atoms, sn_atoms, volume)

# obtain the coordination number and first solvation shell distance
r_li_peo, y_coord_peo = coordination.obtain_rdf_coord(bins_peo, rdf_peo, coord_num_peo)
r_li_tfsi, y_coord_tfsi = coordination.obtain_rdf_coord(bins_tfsi, rdf_tfsi, coord_num_tfsi)
r_li_sn, y_coord_sn = coordination.obtain_rdf_coord(bins_sn, rdf_sn, coord_num_sn)

# Example usage of the function
run_start = 0
run_end = 80001

# Select the atoms of interest
target_groups = {
    'PEO': peo_atoms,
    'TFSI': tfsi_atoms,
    'SN': sn_atoms,
}

# setting the first solvation shell distance
cutoff_radii = {
    'PEO': r_li_peo,
    'TFSI': r_li_tfsi,
    'SN': r_li_sn,
}

# analysis the coordination for targed compound
coord = coordination.analyze_coordination(u, li_atoms, target_groups, cutoff_radii, run_start, run_end)

# Calculate the distribution as percentages
unique, counts = np.unique(coord, return_counts=True)
total_counts = (run_end - run_start) * len(li_atoms)
percentages = np.round((counts / total_counts) * 100, 2)

distribution = dict(zip(unique, percentages))

# 使用repr()将字典转换为字符串
frequencies_str = repr(distribution)

# 写入文件
file_path = 'coordination.txt'
with open(file_path, 'w') as file:
    file.write(frequencies_str)



