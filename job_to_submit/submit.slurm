#!/bin/bash

#SBATCH --job-name="NN"
#SBATCH --partition=normal
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=32
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --time=10:00:00

module load intel/psxe-2020

python case_prediction.py
