#!/bin/bash
#SBATCH --job-name=motion_diversity_multi_scene_4_2
#SBATCH --output=sbatch_logs/motion_diversity_multi_scene_4_2.out
#SBATCH --error=sbatch_logs/motion_diversity_multi_scene_4_2.err
#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node="l40s:1"
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

source /coc/flash7/bli678/Shared/emimic/bin/activate

# Extract number of GPUs from SLURM_GPUS_PER_NODE (format: "l40s:4" -> 4)
NUM_GPUS_PER_NODE=$(echo ${SLURM_GPUS_PER_NODE} | cut -d: -f2)
export SLURM_GPUS=$((NUM_GPUS_PER_NODE * SLURM_NNODES))
echo "Using node: $SLURM_NODELIST, GPUs per node: $NUM_GPUS_PER_NODE, total GPUs: $SLURM_GPUS"

# Set PyTorch memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python egomimic/trainHydra.py \
    --config-name=train.yaml \
    data=motion_diversity_multi_scene_4_2 \
    logger.wandb.project=everse_offline_eval_motion_diversity_multi_scene \
    name=fold-clothes \
    description=operator-4-eps-2