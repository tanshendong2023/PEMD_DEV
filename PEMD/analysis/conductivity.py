
# ****************************************************************************** #
#      The module implements functions to calculate the ionic conductivity       #
# ****************************************************************************** #


import numpy as np
from tqdm.auto import tqdm
from PEMD.analysis import msd


def calc_cond_msd(run, cations, anions, run_start, ):

    # Split atoms into lists by residue for cations and anions
    cations_list = cations.atoms.split("residue")
    anions_list = anions.atoms.split("residue")

    # compute sum over all charges and positions
    qr = []
    for _ts in tqdm(run.trajectory[run_start:], desc='Calculating conductivity'):
        qr_temp = np.zeros(3)
        for cation in cations_list:
            qr_temp += cation.center_of_mass() * int(1)
        for anion in anions_list:
            qr_temp += anion.center_of_mass() * int(-1)
        qr.append(qr_temp)
    return msd.msd_fft(np.array(qr))

def calc_conductivity(slope, v, T):
    # Calculate conductivity from the slope
    A2cm = 1e-8  # Angstroms to cm
    ps2s = 1e-12  # picoseconds to seconds
    e2c = 1.60217662e-19  # elementary charge to Coulomb
    kb = 1.38064852e-23  # Boltzmann Constant, J/K
    convert = e2c * e2c / ps2s / A2cm * 1000

    cond = slope / 6 / kb / T / v * convert   # "mS/cm"

    return cond






