#!/bin/bash
#BSUB -J building_stats_aarhus
#BSUB -q hpc
#BSUB -W 48:00
#BSUB -R "rusage[mem=64GB]"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
##BSUB -w download
#BSUB -o batch_output/building_stats_aarhus_%J.out
#BSUB -e batch_output/building_stats_aarhus_%J.err
#BSUB -B
#BSUB -N

source /zhome/00/f/156462/miniconda3/etc/profile.d/conda.sh
conda activate gis

# prevent oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python 1_building_stats.py aarhus