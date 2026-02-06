#!/bin/bash
#SBATCH --job-name=T_cup_BC+0ID+8EV_cam
#SBATCH --account=a144
#SBATCH --output=/iopsstor/scratch/cscs/jiaqchen/egomim_out/zeta/50hz/BC+0ID+8EV/cup/slurm-cup-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jiaqchen/egomim_out/zeta/50hz/BC+0ID+8EV/cup/slurm-cup-%j.err
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
export VARIANT="BC+0ID+8EV"
export DATA_CONFIG="cup/multi_data_BC+0ID+8EV"
# Data config: /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/egomimic/hydra_configs/data/cup/multi_data_BC+0ID+8EV.yaml
export CONFIG_SUFFIX="_BC_aria"  # _BC (EVE-only) or _BC_aria (EVE + Aria)
# train config: /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/egomimic/hydra_configs/train_eth_bimanual_BC_aria.yaml
export RLDB_WORKERS=10
export frame_type="cam_frame"  # base_frame, cam_frame, or ee_frame
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
