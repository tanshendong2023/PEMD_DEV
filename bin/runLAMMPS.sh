#!/bin/bash

# Load the LAMMPS module (adjust according to your environment)
module load LAMMPS/2Aug2023_update1

mpirun lmp < "lammps.in" >"out.lmp" 2> "lmp.err"

echo "Completed LAMMPS simulation"



