#!/bin/bash
#SBATCH --job-name=T_cup_BC_delta_cam
#SBATCH --account=a144
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/zeta/50hz/BC_delta/cup/slurm-cup-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/zeta/50hz/BC_delta/cup/slurm-cup-%j.err
#SBATCH --partition=normal
#SBATCH --environment=/users/jiaqchen/.edf/faive2lerobot.toml
#SBATCH --requeue
#SBATCH --signal=USR1@600
##################### SBATCH RESOURCES #####################
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
############################################################

##################### EXPERIMENT CONFIG ####################
export VARIANT="BC_delta"
export DATA_CONFIG="cup/multi_data_BC"
# Data config: /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/egomimic/hydra_configs/data/cup/multi_data_BC.yaml
export CONFIG_SUFFIX="_BC"  # _BC (EVE-only) or _BC_aria (EVE + Aria)
# train config: /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/egomimic/hydra_configs/train_eth_bimanual_BC.yaml
export RLDB_WORKERS=10
export frame_type="cam_frame"  # base_frame, cam_frame, or ee_frame
export delta=true  # Use delta/relative actions
############################################################

# Parse command-line arguments (--debug, --new-wandb, --skip-preflight)
for arg in "$@"; do
    case $arg in
        --debug) export debug=true ;;
        --new-wandb) export new_wandb=true ;;
        --skip-preflight) export skip_preflight=true ;;
    esac
done

# Source common logic
SCRIPT_DIR="/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/eth_clariden/cup/"
source "${SCRIPT_DIR}/common.sh"