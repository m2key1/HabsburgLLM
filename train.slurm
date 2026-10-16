#!/bin/bash
#SBATCH --job-name=olmo-habsburg
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=14-00:00:00
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p runs

unset LD_LIBRARY_PATH

.venv/bin/torchrun --standalone --nproc_per_node=8 scripts/train.py
