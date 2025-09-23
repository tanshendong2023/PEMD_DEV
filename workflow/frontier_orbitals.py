# Author: Shendong Tan
# Date: 2025-09-19
# Description: Workflow for frontier orbital energies of polymer electrolytes

from PEMD.core.run import QMRun
from PEMD.core.model import PEMDModel

work_dir = '/'

pdb_filename = PEMDModel.homopolymer(
    work_dir,
    smiles = '*CCO*',
    length = 3,
    name = 'PEO',
    resname = 'MOL',
    left_cap = '',
    right_cap = ''
)

xyz_file = QMRun.conformer_search(
    work_dir,
    pdb_file = pdb_filename,
    max_conformers = 1000,
    top_n_MMFF = 100,
    top_n_xtb = 10,
    top_n_qm = 5,
    charge = 0,
    mult = 1,
    gfn = 'gfn2',
    function = 'b3lyp',
    basis_set = '6-31g*',
    epsilon = 5.0,
    core = 32,
    memory = '64GB',
)

QMRun.qm_gaussian(
    work_dir,
    xyz_file = xyz_file,
    gjf_filename = 'PEO',
    charge = 0,
    mult = 1,
    function = 'b3lyp',
    basis_set = 'def2tzvp',
    epsilon = 4.07673,
    core = 32,
    memory = '64GB',
    chk = True,
    optimize = True,
    multi_step = True,
    max_attempts = 4,
)
