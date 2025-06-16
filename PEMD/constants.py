# PEMD/constants.py

"""
Global constants for the PEMD code library.
"""

import numpy as np

# Default direction vector and threshold for validation
DEFAULT_DIRECTION = np.array([1.0, 0.0, 0.0])
MIN_DIRECTION_NORM = 0.1

# Periodic table: atomic number -> element symbol
PERIODIC_TABLE = [
    'H',   'He',  'Li',  'Be',  'B',   'C',   'N',   'O',   'F',   'Ne',
    'Na',  'Mg',  'Al',  'Si',  'P',   'S',   'Cl',  'Ar',  'K',   'Ca',
    'Sc',  'Ti',  'V',   'Cr',  'Mn',  'Fe',  'Co',  'Ni',  'Cu',  'Zn',
    'Ga',  'Ge',  'As',  'Se',  'Br',  'Kr',  'Rb',  'Sr',  'Y',   'Zr',
    'Nb',  'Mo',  'Tc',  'Ru',  'Rh',  'Pd',  'Ag',  'Cd',  'In',  'Sn',
    'Sb',  'Te',  'I',   'Xe',  'Cs',  'Ba',  'La',  'Ce',  'Pr',  'Nd',
    'Pm',  'Sm',  'Eu',  'Gd',  'Tb',  'Dy',  'Ho',  'Er',  'Tm',  'Yb',
    'Lu',  'Hf',  'Ta',  'W',   'Re',  'Os',  'Ir',  'Pt',  'Au',  'Hg',
    'Tl',  'Pb',  'Bi',  'Po',  'At',  'Rn',  'Fr',  'Ra',  'Ac',  'Th',
    'Pa',  'U',   'Np',  'Pu',  'Am',  'Cm',  'Bk',  'Cf',  'Es',  'Fm',
    'Md',  'No',  'Lr',  'Rf',  'Db',  'Sg',  'Bh',  'Hs',  'Mt',  'Ds',
    'Rg',  'Cn',  'Fl',  'Lv',  'Ts',  'Og'
]

FARADAY_CONSTANT = 96485.3329  # C/mol, Faraday constant


