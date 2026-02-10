#!/usr/bin/env python3
"""
Generate SBATCH files for all cup_saucer scene diversity configs.
"""

import os
from pathlib import Path

# Template for sbatch file
template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=sbatch_logs/{job_name}.out
#SBATCH --error=sbatch_logs/{job_name}.err
#SBATCH --partition="rl2-lab"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

source /coc/flash7/bli678/Shared/emimic/bin/activate

# Extract number of GPUs from SLURM_GPUS_PER_NODE (format: "l40s:4" -> 4)
NUM_GPUS_PER_NODE=$(echo ${{SLURM_GPUS_PER_NODE}} | cut -d: -f2)
export SLURM_GPUS=$((NUM_GPUS_PER_NODE * SLURM_NNODES))
echo "Using node: $SLURM_NODELIST, GPUs per node: $NUM_GPUS_PER_NODE, total GPUs: $SLURM_GPUS"

# Set PyTorch memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python egomimic/trainHydra.py \\
    --config-name=train.yaml \\
    data=cup_saucer/scene_diversity/{config_name} \\
    logger.wandb.project=everse_scene_diversity_cup_saucer \\
    name=cup_saucer_scene \\
    description={description}
"""

# Create output directory
output_dir = Path('sbatch/cup_saucer/scene_diversity')
output_dir.mkdir(parents=True, exist_ok=True)

# Configurations: (num_scenes, minutes_per_scene)
configs = [
    # 1 scene configs
    (1, 3.75),
    (1, 7.5),
    (1, 15),
    (1, 30),
    (1, 60),
    # 2 scene configs
    (2, 3.75),
    (2, 7.5),
    (2, 15),
    (2, 30),
    (2, 60),
    # 4 scene configs
    (4, 3.75),
    (4, 7.5),
    (4, 15),
    (4, 30),
    (4, 60),
    # 8 scene configs
    (8, 3.75),
    (8, 7.5),
    (8, 15),
    (8, 30),
    (8, 60),
    # 16 scene configs
    (16, 3.75),
    (16, 7.5),
    (16, 15),
    (16, 30),
    (16, 60),
]

print('Generating SBATCH files for cup_saucer scene_diversity configs...')
for num_scenes, minutes in configs:
    # Generate filename format
    if minutes == int(minutes):
        minutes_str = str(int(minutes))
    else:
        minutes_str = str(minutes).replace('.', '_')
    
    job_name = f'scene_diversity_{num_scenes}_{minutes_str}'
    config_name = job_name
    description = f'{num_scenes}-{minutes_str}'
    
    content = template.format(
        job_name=job_name,
        config_name=config_name,
        description=description
    )
    
    filename = output_dir / f"{job_name}.sh"
    with open(filename, 'w') as f:
        f.write(content)
    
    # Make executable
    os.chmod(filename, 0o755)
    
    print(f'Created: {filename}')

print('\nAll SBATCH files created!')

