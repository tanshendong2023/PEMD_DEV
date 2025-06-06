# PolyElecMD

PolyElecMD is a Python package for molecular dynamics simulations of polymers with charged end groups.

## Features
- Molecular dynamics simulations of polymers with charged end groups
- Analysis of polymer properties
- Visualization of simulation results
- Easy-to-use command line interface

## Requirements
- Python 3.8 or higher
- NumPy
- Pandas
- Matplotlib
- Click
- PyYAML
- MDAnalysis
- OpenMM
- NGLView
- TQDM

## Installation
To install PolyElecMD, clone the repository and install the required packages using the following commands:
```bash
git clone
cd PEMD
conda env create -f environment.yml
conda activate PEMD
pip install -e .
```

## Polymer Building

You can construct different polymers directly using
`PEMDModel.build_copolymer` by specifying a building `mode`:

```python
from pathlib import Path
from PEMD.core.model import PEMDModel

model = PEMDModel(Path("./"), "A", "MOL", "[*]C[*]", "", "", 5, 10)
model.build_copolymer(
    poly_name_A="A",
    poly_name_B="B",
    smiles_A="[*]C[*]",
    smiles_B="[*]N[*]",
    mode="random",
    length=20,
    frac_A=0.5,
)
```

The ``mode`` argument supports ``homopolymer``, ``random``, ``alternating`` and
``block``. If ``poly_resname`` is omitted, the value stored on the model
instance is used. Functions ``gen_homopolymer_3D`` and friends remain as
wrappers for backward compatibility.





