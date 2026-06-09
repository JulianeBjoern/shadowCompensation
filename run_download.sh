#!/bin/bash
#BSUB -J download
#BSUB -q hpc
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -o batch_output/download_%J.out
#BSUB -e batch_output/download_%J.err
#BSUB -B
#BSUB -N

source /zhome/00/f/156462/miniconda3/etc/profile.d/conda.sh
conda activate gis

# prevent oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python 0_download.py nordvest