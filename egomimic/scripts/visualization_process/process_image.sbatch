#!/bin/bash
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/process_image/%j.out
cd /nethome/paphiwetsa3/flash/projects/EgoVerse

# Ensure non-interactive Slurm shell has your functions/aliases (e.g. `actenv`)
# available. `sbatch` does not source ~/.bashrc by default.
source .venv/bin/activate

srun --cpus-per-task=12 python3 -u egomimic/scripts/visualization_process/process_image.py --embodiment aria_bimanual --out-dir egomimic/scripts/visualization_process/data
