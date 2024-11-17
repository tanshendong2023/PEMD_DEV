"""
Visualization Toolkit for Molecular Structures

This script serves as a visualization tool within a broader computational suite aimed at analyzing molecular and crystalline structures pertinent to materials science and chemistry. It leverages advanced visualization techniques to render three-dimensional representations of molecular geometries, facilitating an intuitive understanding of complex structural data. The tool is particularly geared towards enhancing the interpretability of molecular dynamics simulations, crystallography studies, and other computational chemistry applications by providing clear, detailed visualizations of atomic arrangements and bonds in polymers, electrolytes, and other compounds.

Key features include support for various file formats (e.g., CIF, PDB, VASP), the ability to replicate unit cells for extended structures, and customizable rendering styles to suit different analytical needs. This aids researchers and scientists in the rapid assessment of molecular configurations, interactions, and the spatial orientation of atoms, contributing to the accelerated discovery and characterization of novel materials with potential applications in energy, pharmaceuticals, and nanotechnology.

Developed by: Tan Shendong
Date: 2024.02.23
"""

import py3Dmol
import subprocess
import datamol as dm
from IPython.display import display

def vis_3D(input_file, supercell=[1,1,1]):
  if input_file.split(".")[-1] == 'vasp':
    subprocess.run(['obabel', '-iposcar', input_file, '-ocif', '-Omodel_wCell.cif'])
    input_file = "model_wCell.cif"
  with open(input_file) as ifile:
      print(input_file)
      system = "".join([x for x in ifile])
  view = py3Dmol.view(width=400, height=300)
  view.addModelsAsFrames(system)
  view.setStyle({"stick":{}}) # 'colorscheme':'greenCarbon'
  view.addUnitCell()
  view.replicateUnitCell(supercell[0],supercell[1],supercell[2])
  view.zoomTo()
  view.show()

def vis_2Dsmiles(smiles, mol_size=(350, 150)):
  img = dm.to_image(smiles, mol_size=mol_size)
  display(img)