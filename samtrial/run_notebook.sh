#!/bin/bash
### General options
#BSUB -q gpuv100
#BSUB -J notebook
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 06:00
#BSUB -o batch_output/notebook_%J.out
#BSUB -e batch_output/notebook_%J.err
#BSUB -B
#BSUB -N
#BSUB -u s203971@student.dtu.dk

mkdir -p batch_output

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate sam-aerial

nvidia-smi

# --- SAM2 checkpoint ---
mkdir -p checkpoints
if [ ! -f "checkpoints/sam2_hiera_large.pt" ]; then
    echo "Downloading SAM2 checkpoint..."
    wget -q -O checkpoints/sam2_hiera_large.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
fi

# --- Grounding DINO config ---
DINO_CONFIG_DIR="/work3/s203971/samtrial/GroundingDINO/groundingdino/config"
mkdir -p "$DINO_CONFIG_DIR"
if [ ! -f "$DINO_CONFIG_DIR/GroundingDINO_SwinT_OGC.py" ]; then
    echo "Downloading GroundingDINO config..."
    wget -q -O "$DINO_CONFIG_DIR/GroundingDINO_SwinT_OGC.py" \
        https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
fi

# --- Grounding DINO weights ---
if [ ! -f "checkpoints/groundingdino_swint_ogc.pth" ]; then
    echo "Downloading GroundingDINO weights (~694MB)..."
    wget -q -O checkpoints/groundingdino_swint_ogc.pth \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
fi

echo "All checkpoints ready."

jupyter nbconvert \
    --to notebook \
    --execute samtrial.ipynb \
    --output samtrial_executed.ipynb \
    --ExecutePreprocessor.timeout=3600
