import os
import subprocess
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def dens_temp(out_dir, tpr_file, edr_file, module_soft='GROMACS', initial_time=500, time_gap=4000, duration=1000,
              temp_initial=600, temp_decrement=20, max_time=102000, summary_file="dens_tem.csv"):
    # go to dir
    current_path = os.getcwd()
    MD_dir = os.path.join(current_path, out_dir)
    os.chdir(MD_dir)

    # Load GROMACS module before starting the loop
    subprocess.run(f"module load {module_soft}", shell=True)

    # Initialize a list to store the data
    results = []

    # Loop until time exceeds max_time ps
    time = initial_time
    temp = temp_initial

    while time <= max_time:
        start_time = time
        end_time = time + duration

        print(f"Processing temperature: {temp}K, time interval: {start_time} to {end_time}ps")

        # Use gmx_mpi energy to extract density data and extract the average density value from the output
        command = f"echo Density | gmx_mpi energy -f {edr_file} -s {tpr_file} -b {start_time} -e {end_time}"
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        density_lines = [line for line in result.stdout.split('\n') if "Density" in line]
        density = round(float(density_lines[0].split()[1]) / 1000, 4) if density_lines else "N/A"

        # Append the extracted density value and corresponding temperature to the results list
        results.append({"Temperature": temp, "Density": density})

        # Update time and temperature for the next loop iteration
        time += time_gap
        temp -= temp_decrement

    # Convert the list of results to a DataFrame
    df = pd.DataFrame(results)
    # Save the DataFrame to a CSV file
    df.to_csv(summary_file, index=False)

    os.chdir(current_path)
    print("Density extraction and summary for all temperature points completed.")

    return df


def fit_tg(df, param_file="fitting_tg.csv"):
    # Define the fitting function
    def fit_function(T, a, b, c, Tg, xi):
        return a*T + b - c*(T - Tg) * (1 + (T - Tg) / np.sqrt((T - Tg)**2 + xi**2))

    # Extract temperature and density from the DataFrame
    temperatures = df['Temperature'].to_numpy()
    densities = df['Density'].to_numpy()

    # Initial guess for the fitting parameters
    initial_guess = [1, 1, 1, 300, 1]

    # Set parameter bounds
    bounds = ([-np.inf, -np.inf, -np.inf, 100, 0], [np.inf, np.inf, np.inf, 600, np.inf])

    # Perform the curve fitting
    popt, pcov = curve_fit(
        fit_function,
        temperatures,
        densities,
        p0=initial_guess,
        maxfev=5000,
        bounds=bounds
    )

    # Extracting fitted parameters
    a_fit, b_fit, c_fit, Tg_fit, xi_fit = popt
    print(f"Fitted parameters: a = {a_fit}, b = {b_fit}, c = {c_fit}, Tg = {Tg_fit}, xi = {xi_fit}")
    print(f"Estimated Tg from fit: {Tg_fit} K")

    # Save fitting parameters to CSV file
    param_df = pd.DataFrame({
        'Parameter': ['a', 'b', 'c', 'Tg', 'xi'],
        'Value': [a_fit, b_fit, c_fit, Tg_fit, xi_fit]
    })
    param_df.to_csv(param_file, index=False)

    return param_df