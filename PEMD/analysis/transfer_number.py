
# ****************************************************************************** #
#      The module implements functions to calculate the transfer number          #
# ****************************************************************************** #


def calc_transfer_number(slope_plusplus, slope_minusminus, T, v, cond):

    A2cm = 1e-8  # Angstroms to cm
    ps2s = 1e-12  # picoseconds to seconds
    e2c = 1.60217662e-19  # elementary charge to Coulomb
    kb = 1.38064852e-23  # Boltzmann Constant, J/K
    convert = e2c * e2c / ps2s / A2cm * 1000

    slope_plusminus = (cond / convert * 6 * kb * T * v - slope_plusplus - slope_minusminus) / -2

    t = (slope_plusplus - slope_plusminus) / (slope_plusplus + slope_minusminus - 2 * slope_plusminus)   # mS/cm

    return t
