#!/bin/bash
#SBATCH --job-name=gbsa_locality
#SBATCH --nodelist=compute-cpu-5
#SBATCH --time=30:00
#SBATCH --output=python/tests/locality_test_%j.out
#SBATCH --error=python/tests/locality_test_%j.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate openmmgridforce312

cd /home/jtufts/src/p312/openmmgridforce
python python/tests/test_receptor_locality_cutoff.py
