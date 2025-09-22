# Author: Shendong Tan
# Date: 2025-09-19
# Description: Workflow for Molecular Dynamics Simulation of Polymer Electrolytes

from PEMD.core.run import QMRun
from PEMD.core.model import PEMDModel
from PEMD.core.run import MDRun
from PEMD.core.forcefields import Forcefield

work_dir = './'
json_file = 'md.json'

pdb_file_short, pdb_file_long = PEMDModel.homopolymer_from_json(
   work_dir,
   json_file
)

xyz_file = QMRun.conformer_search(
    work_dir,
    pdb_file = pdb_file_short,
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

resp_df = QMRun.resp_chg_fitting(
   work_dir,
   xyz_file = xyz_file,
   charge = 0,
   mult = 1,
   function = 'B3LYP',
   basis_set = 'def2TZVP',
   epsilon = 5.0,
   core = 32,
   memory = '64GB',
   method = 'resp2'
)

Forcefield.oplsaa_from_json(
    work_dir,
    json_file,
    mol_type = "polymer",
    resp_df = resp_df,
    ff_source = "ligpargen",
    pdb_file = pdb_file_long,
)

Forcefield.oplsaa_from_json(
    work_dir,
    json_file,
    mol_type = "Li_cation",
    ff_source = "database",
)

Forcefield.oplsaa_from_json(
    work_dir,
    json_file,
    mol_type = "salt_anion",
    ff_source = "database",
)

PEMDModel.amorphous_cell_from_json(
  work_dir,
  json_file,
  density = 0.8,
  add_length = 25,
  packinp_name = 'pack.inp',
  packpdb_name = 'pack_cell.pdb',
)

MDRun.annealing_from_json(
    work_dir,
    json_file,
    temperature = 298,
    T_high_increase = 300,
    anneal_rate = 0.05,
    anneal_npoints = 5,
    packmol_pdb = "pack_cell.pdb",
)

MDRun.production_from_json(
    work_dir,
    json_file,
    temperature = 298,
    nstep_ns = 200,
)