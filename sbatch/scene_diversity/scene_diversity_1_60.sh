#!/bin/bash
#SBATCH --job-name=scene_diversity_1_60
#SBATCH --output=sbatch_logs/scene_diversity_1_60.out
#SBATCH --error=sbatch_logs/scene_diversity_1_60.err
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
#     data=scene_diversity/scene_diversity_1_60 \
#     logger.wandb.project=everse_scenes_diveristy_fold_clothes \
#     name=fold-clothes \
#     description=scenes-1-time-60

python egomimic/trainHydra.py \
    --config-name=train.yaml \
    data=scene_diversity/scene_eval \
    logger.wandb.project=everse_scene_diversity_fold_clothes \
    name=eval-fold-clothes-scene-diversity \
    description=scenes-1-time-60 \
    train=false \
    validate=true \
    ckpt_path="/coc/cedarp-dxu345-0/bli678/EgoVerse/logs/fold_clothes/scene_diversity/scenes-1-time-60_2026-01-21_22-01-40/everse_scenes_diveristy_fold_clothes/fold-clothes_scenes-1-time-60_2026-01-21_22-01-40/checkpoints/last.ckpt"