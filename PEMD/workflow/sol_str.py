import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rdf

work_dir = './'

data_tpr_file = os.path.join(work_dir, 'nvt_prod.tpr')
dcd_xtc_file = os.path.join(work_dir, 'nvt_prod.xtc')

# Load the trajectory
u = mda.Universe(data_tpr_file, dcd_xtc_file)

# Select the atoms of interest
LI = u.select_atoms('resname LIP and name Li')
PEO = u.select_atoms('resname MOL and name O')
TFSI = u.select_atoms('resname NSC and name OBT')
SN = u.select_atoms('resname SN and name N')

# Initialize RDF analysis
# LI_PEO
rdf_LI_PEO = rdf.InterRDF(LI, PEO, nbins=200, range=(0.0, 10.0))
rdf_LI_PEO.run()
# LI_TFSI
rdf_LI_TFSI = rdf.InterRDF(LI, TFSI, nbins=200, range=(0.0, 10.0))
rdf_LI_TFSI.run()
# LI_SN
rdf_LI_SN = rdf.InterRDF(LI, SN, nbins=200, range=(0.0, 10.0))
rdf_LI_SN.run()

# Calculate coordination numbers
volume = u.coord.volume
rho = PEO.n_atoms / volume
# LI_PEO
bins1 = rdf_LI_PEO.results.bins
rdf1 = rdf_LI_PEO.results.rdf
coord_numbers1 = np.cumsum(4 * np.pi * bins1**2 * rdf1 * np.diff(np.append(0, bins1)) * rho)
# LI_TFSI
bins2 = rdf_LI_TFSI.results.bins
rdf2 = rdf_LI_TFSI.results.rdf
coord_numbers2 = np.cumsum(4 * np.pi * bins2**2 * rdf2 * np.diff(np.append(0, bins2)) * rho)
# LI_SN
bins3 = rdf_LI_SN.results.bins
rdf3 = rdf_LI_SN.results.rdf
coord_numbers3 = np.cumsum(4 * np.pi * bins3**2 * rdf3 * np.diff(np.append(0, bins3)) * rho)


def obtain_rdf_coord(rdf, bins, coord_numbers):
    # 计算一阶导数的符号变化
    deriv_sign_changes = np.diff(np.sign(np.diff(rdf)))
    # 寻找第一个峰值：导数从正变负
    peak_index = np.where(deriv_sign_changes < 0)[0] + 1  # +1 是因为diff的结果比原数组短1
    if len(peak_index) == 0:
        raise ValueError("No peak found in RDF data.")
    first_peak_index = peak_index[0]

    # 寻找第一个峰值后的第一个极小点
    min_after_peak_index = np.where(deriv_sign_changes[first_peak_index:] > 0)[0] + first_peak_index + 1
    if len(min_after_peak_index) == 0:
        raise ValueError("No minimum found after the first peak in RDF data.")
    first_min_index = min_after_peak_index[0]

    # 极小点处的坐标和配位数
    x_val = bins[first_min_index]
    y_rdf = rdf[first_min_index]
    y_coord = np.interp(x_val, bins, coord_numbers)
    return x_val, y_rdf, y_coord


x_val1, y_rdf1, y_coord1 = obtain_rdf_coord(rdf1, bins1, coord_numbers1)
x_val2, y_rdf2, y_coord2 = obtain_rdf_coord(rdf2, bins2, coord_numbers2)
x_val3, y_rdf3, y_coord3 = obtain_rdf_coord(rdf3, bins3, coord_numbers3)
print(f'{y_coord1:.3f}')
print(f'{y_coord2:.3f}')
print(f'{y_coord3:.3f}')
