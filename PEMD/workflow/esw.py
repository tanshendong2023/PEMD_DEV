# Author: Shendong Tan
# Date: 2025-09-19
# Description: Workflow for Oxidative stability window of polymer electrolytes

from PEMD.core.analysis import PEMDAnalysis
from PEMD.core.run import QMRun


work_dir = './'
poly_name = 'PEO'

center_atom_name = 'resname NSC and name NBT'

distance_dict = {
    "PEO": 2.8,
}

select_dict = {
    "PEO": 'resname MOL and name H',
}

xyzfile = PEMDAnalysis.write_cluster_polymer(
    work_dir = './',
    tpr_file = 'nvt_prod.tpr',
    wrap_xtc_file = 'nvt_prod.xtc',
    center_atom_name = center_atom_name,
    distance_dict = distance_dict,
    select_dict = select_dict,
    poly_name = 'PEO',
    repeating_unit = '*CCO*',
    length = 3,
    run_start = 0,
    run_end = 80000,
    structure_code = 1,
    max_number = 100,
    write_freq = 0.10,
)

QMRun.qm_gaussian(
   work_dir = './',
   xyz_file = xyzfile,
   gjf_filename = 'conf',
   charge = -1,
   mult = 1,
   function = 'b3lyp',
   basis_set = '6-31+g*',
   epsilon = 4.07673,
   core = 32,
   memory = '64GB',
   optimize = False,
   toxyz = True,
   top_n_qm = 5
)

QMRun.qm_gaussian(
    work_dir = './',
    xyz_file = 'gaussian_top5.xyz',
    gjf_filename = f'{poly_name}_init',
    charge = -1,
    mult = 1,
    function = 'b3lyp',
    basis_set = 'def2tzvp',
    epsilon = 4.07673,
    core = 32,
    memory = '64GB',
    multi_step = True,
    max_attempts = 4,
)

QMRun.qm_gaussian(
    work_dir = work_dir,
    xyz_file = 'gaussian_top5.xyz',
    gjf_filename = f'{poly_name}_oxid',
    charge = 0,
    mult = 2,
    function = 'b3lyp',
    basis_set = 'def2tzvp',
    epsilon = 4.07673,
    core = 32,
    memory = '64GB',
    multi_step = True,
    max_attempts = 4,
)

