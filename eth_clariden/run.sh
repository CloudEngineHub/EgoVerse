#!/bin/bash
#SBATCH --job-name=ego_ee
#SBATCH --account=a144
#SBATCH --output=slurm-ego_ee-%j.out
#SBATCH --error=slurm-ego_ee-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --environment=/users/jiaqchen/.edf/faive2lerobot.toml



# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

# Check some specs
free -h
nvidia-smi --query-gpu=memory.total --format=csv

# Source and target paths
# SRC=/cluster/work/cvg/jiaqchen/EGOMIM/data/cupOnSAUZZ/cup_lerobot_camframe
# DEST=$TMPDIR/cup_camframe

# Copy to local scratch
# echo "Copying data to $DEST ..."
# rsync -a --info=progress2 $SRC/ $DEST/

# # Change some paths so I have enough storage
# export HF_HOME="$TMPDIR/hf"
# export HF_DATASETS_CACHE="$TMPDIR/hf/datasets"
# export HF_HUB_CACHE="$TMPDIR/hf/hub"
# export TRANSFORMERS_CACHE="$TMPDIR/hf/transformers"
# export HF_HUB_OFFLINE=1
# mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"


# Use srun to launch all 4 processes simultaneously for DDP
# Similar to the example script, use bash -c to ensure venv is activated in each process
# SLURM allocates 4 tasks (--ntasks-per-node=4), and srun launches one process per task
# Each process will source clariden.sh to activate venv and set up environment variables
CMD="
source /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/eth_clariden/clariden.sh
cd /iopsstor/scratch/cscs/jiaqchen/egomim_out
python /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/egomimic/trainHydra.py $@
"
srun bash -c "$CMD"

# Copy the results to capstor


# Print completion information
echo "Job finished at: $(date)"
