#!/bin/bash
#SBATCH --job-name=mps_local
#SBATCH --output=sbatch_logs/mps_local.out
#SBATCH --error=sbatch_logs/mps_local.err
#SBATCH --partition="rl2-lab"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"

# --- Credentials ---
export MPS_USER="georgiat_zb658p"
export MPS_PASSWORD="georgiat0001"

# require "projectaria-tools==1.7.1"
# aria_mps --help
source /coc/flash7/bli678/miniconda3/etc/profile.d/conda.sh
conda activate egowm

cd /coc/flash7/scratch/egowm/raw/aria_raw
aria_mps single -i . --no-ui --retry-failed -u "$MPS_USER" -p "$MPS_PASSWORD" --no-save-token

