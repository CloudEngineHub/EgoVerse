#!/bin/bash
#SBATCH --job-name=scene_diversity_1_60
#SBATCH --output=sbatch_logs/scene_diversity_1_60.out
#SBATCH --error=sbatch_logs/scene_diversity_1_60.err
#SBATCH --partition="rl2-lab"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
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
#     data=cup_saucer/scene_diversity/scene_diversity_1_60 \
#     logger.wandb.project=everse_scene_diversity_cup_saucer \
#     name=cup_saucer_scene \
#     description=1-60

python egomimic/trainHydra.py \
    --config-name=train.yaml \
    data=cup_saucer/scene_diversity/scene_eval \
    logger.wandb.project=everse_scene_diversity_cup_saucer \
    name=eval-cup-saucer-scene \
    description=1-60 \
    train=false \
    validate=true \
    ckpt_path="/coc/cedarp-dxu345-0/bli678/EgoVerse/logs/cup_saucer_scene/1-60_2026-02-04_22-30-18/checkpoints/epoch_epoch\=1399.ckpt"
