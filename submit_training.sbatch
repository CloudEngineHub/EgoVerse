#!/bin/bash
#SBATCH -J egoverse_RBY1         # Job name
#SBATCH --partition=overcap
#SBATCH --nodes=1   # NOTE: try to use GPUs on the same node to reduce communication overhead
#SBATCH --gres=gpu:a40:1               # Request how many H100 GPUs per node
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-gpu=32G               # RAM per GPU Memory per core
#SBATCH -t 24:00:00                     # Duration of the job
#SBATCH --requeue
#SBATCH -o ./logs/%A/task_%a/out.log
#SBATCH -e ./logs/%A/task_%a/err.log
#SBATCH --exclude='xaea-12,dendrite,synapse,uniblab,puma'

cd /coc/flash7/zhenyang/EgoVerse

source emimic/bin/activate

export TMPDIR=/tmp

# python egomimic/trainHydra.py model=hpt_bc_flow_rby1 name=RBY1_test_resume_0301_2 description=resume_last_ckpt

# python egomimic/trainHydra.py model=hpt_bc_flow_eva name=EgoWM_test

# python egomimic/trainHydra.py model=hpt_bc_flow_rby1 name=RBY1_test_0305_lowdim description=debug 

# use the test_2 dataset
# Test1: image + proprio → bi arm action
# Test2:  proprio → bi arm action
# actions_biarm: 14-dim (vs actions 49-dim). Override schematic, delta_timestamps, and act_dim.
python egomimic/trainHydra.py \
        model=hpt_bc_flow_rby1 \
        name=RBY1_test_biarm_action_image_0305 \
        description=RBY1_test_biarm_action_image_0305 \
        data_schematic.schematic_dict.rby1.actions.lerobot_key=actions_biarm \
        model.robomimic_model.head_specs.rby1.infer_ac_dims.rby1=14 \
        model.robomimic_model.head_specs.rby1.model.act_dim=14