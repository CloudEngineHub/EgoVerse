#!/bin/bash
#SBATCH --job-name=motion_diversity_multi_scene_12_3_75
#SBATCH --output=sbatch_logs/motion_diversity_multi_scene_12_3_75.out
#SBATCH --error=sbatch_logs/motion_diversity_multi_scene_12_3_75.err
#SBATCH --partition="overcap"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

source /coc/flash7/bli678/Shared/emimic/bin/activate

# Extract number of GPUs from SLURM_GPUS_PER_NODE (format: "l40s:4" -> 4)
NUM_GPUS_PER_NODE=$(echo ${SLURM_GPUS_PER_NODE} | cut -d: -f2)
export SLURM_GPUS=$((NUM_GPUS_PER_NODE * SLURM_NNODES))
echo "Using node: $SLURM_NODELIST, GPUs per node: $NUM_GPUS_PER_NODE, total GPUs: $SLURM_GPUS"

# Set PyTorch memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# python egomimic/trainHydra.py \
#     --config-name=train.yaml \
#     data=cup_saucer/motion_diversity/motion_diversity_multi_scene_12_3_75 \
#     logger.wandb.project=everse_motion_diversity_multi_scene_cup_saucer \
#     name=cup_saucer \
#     description=12-3_75

python egomimic/trainHydra.py \
    --config-name=train.yaml \
    data=cup_saucer/motion_diversity/motion_eval \
    logger.wandb.project=everse_motion_diversity_multi_scene_cup_saucer \
    name=eval-cup-saucer-motion \
    description=12-3_75 \
    train=false \
    validate=true \
    ckpt_path="/coc/cedarp-dxu345-0/bli678/EgoVerse/logs/cup_saucer_motion/12-3_75_2026-02-03_12-02-47/checkpoints/epoch_epoch\=1399.ckpt"

