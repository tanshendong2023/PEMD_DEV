import os
from PEMD.core.analysis import PEMDAnalysis

work_dir = './'
tpr_file = "nvt_prod.tpr"
xtc_unwrap_file = "nvt_prod_unwrap.xtc"
xtc_wrap_file = "nvt_prod.xtc"
cation_name = 'resname LIP and name Li'
anion_name = 'resname NSC and name OBT'
polymer_name = 'resname MOL and name O'
plasticizer_name = 'resname SN and name N'


dt = 0.001
dt_collection = 5e3
run_start=0
run_end=80001
temperature=450

analysis = PEMDAnalysis.from_gromacs(
    work_dir,
    tpr_file,
    xtc_wrap_file,
    xtc_unwrap_file,
    cation_name,
    anion_name,
    polymer_name,
    run_start,
    run_end,
    dt,
    dt_collection,
    temperature,
)

cond = analysis.get_conductivity()
D_cations, D_anions = analysis.get_self_diffusion_coefficient()
t = analysis.get_transfer_number()
coord_Li_PEO = analysis.get_coordination_number(cation_name, polymer_name)
coord_Li_TFSI = analysis.get_coordination_number(cation_name, anion_name)
coord_LI_SN = analysis.get_coordination_number(cation_name, plasticizer_name)
tau3 = analysis.get_tau3()
tau1 = analysis.get_tau1(time_window=2001)
tauR = analysis.get_tauR()
tau2 = analysis.get_tau2(time_window=501)

# Define the path for the results file
results_file_path = os.path.join(work_dir, 'results.txt')

# Write results to a text file
with open(results_file_path, 'w') as file:
    file.write(f"Calculated conductivity: {cond:.2f} ms/cm\n")
    file.write(f"Calculated D_cations: {D_cations:.2f} cm2/s\n")
    file.write(f"Calculated D_anions: {D_anions:.2f} cm2/s\n")
    file.write(f"Calculated transfer number: {t:.2f} \n")
    file.write(f"Calculated coordination of Li-PEO: {coord_Li_PEO:.2f} \n")
    file.write(f"Calculated coordination of Li-TFSI: {coord_Li_TFSI:.2f} \n")
    file.write(f"Calculated coordination of Li-SN: {coord_LI_SN:.2f} \n")
    file.write(f"Calculated τ1: {tau1:.2f} ns\n")
    file.write(f"Calculated τR: {tauR:.2f} ns\n")
    file.write(f"Calculated τ2: {tau2:.2f} ns\n")
    file.write(f"Calculated τ3: {tau3:.2f} ns\n")
print(f"Results saved to {results_file_path}")



