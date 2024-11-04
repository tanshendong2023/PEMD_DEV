"""
PEMD code library.

Developed by: Tan Shendong
Date: 2024.03.17
"""


import pandas as pd


def homo_lumo_energy(sorted_df, unit_name, out_dir, length):
    homo_energy, lumo_energy = None, None
    found_homo = False
    log_file_path = sorted_df.iloc[0]['File_Path']

    with open(log_file_path, 'r') as f:
        for line in f:
            if "Alpha  occ. eigenvalues" in line:
                # Convert the last value to float and assign to HOMO energy
                homo_energy = float(line.split()[-1])
                found_homo = True
            elif "Alpha virt. eigenvalues" in line and found_homo:
                # Convert the fifth value to float and assign to LUMO energy
                lumo_energy = float(line.split()[4])
                # No need to break here, as we want the last occurrence
                found_homo = False

    result_df = pd.DataFrame({
        'out_dir': [out_dir],
        'HOMO_Energy_eV': [homo_energy],
        'LUMO_Energy_eV': [lumo_energy]
    })

    # to csv file
    csv_filepath = f'{out_dir}/{unit_name}_N{length}_HOMO_LUMO.csv'

    # 将DataFrame保存为CSV文件
    result_df.to_csv(csv_filepath, index=False)

    return result_df


def dipole_moment(sorted_df, unit_name, out_dir, length):
    found_dipole_moment = None
    log_file_path = sorted_df.iloc[0]['File_Path']

    with open(log_file_path, 'r') as file:
        for line in file:
            if "Dipole moment (field-independent basis, Debye):" in line:
                found_dipole_moment = next(file)  # Keep updating until the last occurrence

    if  found_dipole_moment:
        parts =  found_dipole_moment.split()
        # Extracting the X, Y, Z components and the total dipole moment
        dipole_moment = parts[7]
        dipole_moment_df = pd.DataFrame({'dipole_moment': [dipole_moment]})

        # to csv file
        csv_filepath = f'{out_dir}/{unit_name}_N{length}_dipole_moment.csv'

        # 将DataFrame保存为CSV文件
        dipole_moment_df.to_csv(csv_filepath, index=False)

        return dipole_moment_df





















