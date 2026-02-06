#!/bin/bash
# Common functions and logic for all cup training scripts
# This file is sourced by the variant-specific run_*.sh scripts

# Stop the script if a command fails or if an undefined variable is used
set -eo pipefail

ulimit -c 0

# Validate required variables from variant config
: "${VARIANT:?VARIANT must be set}"
: "${DATA_CONFIG:?DATA_CONFIG must be set}"
: "${CONFIG_SUFFIX:?CONFIG_SUFFIX must be set (use _BC or _BC_aria)}"
: "${RLDB_WORKERS:?RLDB_WORKERS must be set}"
: "${frame_type:?frame_type must be set (base_frame, cam_frame, or ee_frame)}"

# Resolve data config path for clickable link
export DATA_CONFIG_PATH="/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/egomimic/hydra_configs/data/${DATA_CONFIG}.yaml"

export PG=${PG:-1}
export CL=${CL:-75}
export CL_OUT=${CL_OUT:-100}
echo $PG # POINT_GAP_ACT
echo $CL # CHUNK_LENGTH_ACT, need to change action horizon
echo $CL_OUT # CHUNK_LENGTH_ACT_OUT, need to change action horizon
if [ -z "$CL_OUT" ] || [ "$CL_OUT" = "None" ]; then
    export PG_CL_EXPERIMENT=pg${PG}_cl${CL}
else
    export PG_CL_EXPERIMENT=pg${PG}_cl${CL}_clout${CL_OUT}
fi
echo $PG_CL_EXPERIMENT

# The sbatch script is executed by only one node.
echo "[sbatch-master] running on $(hostname)"

echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
echo "[sbatch-master] SLURM_NNODES: $SLURM_NNODES"
echo "[sbatch-master] SLURM_NODEID: $SLURM_NODEID"

echo "[sbatch-master] define some env vars that will be passed to the compute nodes"

# The defined environment vars will be shared with the other compute nodes.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=12345   # Choose an unused port
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
export NCCL_NET="AWS Libfabric"

# Speed up dataset loading
export RLDB_LOAD_WORKERS=${RLDB_WORKERS}
export HF_HUB_DISABLE_PROGRESS_BARS=1

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

# Check some specs
free -h
nvidia-smi --query-gpu=memory.total --format=csv

##################### SET THESE VARIABLES #####################
export task=cup
export arm=bimanual # right_arm, left_arm, bimanual
# frame_type is set in run_*.sh (base_frame, cam_frame, or ee_frame)
# debug is set via --debug flag in the variant script

# Defaults (can be overridden in run_*.sh before sourcing common.sh)
export quat=${quat:-false}
export actions_for_qpos=${actions_for_qpos:-false}
export delta=${delta:-false}
###############################################################

# Set ee_pose_dim based on arm type and quat
if [ "$arm" = "bimanual" ]; then
    if [ "$quat" = "true" ]; then
        export ee_pose_dim=14
    else
        export ee_pose_dim=12
    fi
else
    if [ "$quat" = "true" ]; then
        export ee_pose_dim=7
    else
        export ee_pose_dim=6
    fi
fi
export joint_dim=34

# Define EXPERIMENT: combines VARIANT (data config) with PG_CL_EXPERIMENT (training params)
# e.g., BC+2ID+8EV_pg1_cl75_clout100
export EXPERIMENT=${VARIANT}_${PG_CL_EXPERIMENT}

if [ "$debug" = true ]; then
    export trainer=debug
    export logger=debug
else
    export trainer=ddp
    export logger=wandb
fi

##################### PATHS #####################
if [ "$debug" = true ]; then
    export debug_folder='DEBUG/'
else
    export debug_folder=''
fi
export hydra_run_dir=/iopsstor/scratch/cscs/jiaqchen/egomim_out/zeta/50hz_outputs/${debug_folder}${EXPERIMENT}/${task}/${task}_${frame_type}
##################################################

export config_name=train_eth_${arm}${CONFIG_SUFFIX}

########################## CHECKPOINT PATH & RESUME VALIDATION #####################
# Default checkpoint path
default_ckpt_path=${hydra_run_dir}/checkpoints/last.ckpt
export ckpt_path=${default_ckpt_path}
# Uncomment below to use a specific checkpoint:
# export ckpt_path='/iopsstor/scratch/cscs/jiaqchen/egomim_out/multi_node_v2/50hz/pg1_cl75_clout100_recheck_split/cup/cup_base_frame/checkpoints/epoch_epoch=499.ckpt'

# If using non-default checkpoint, require RESUME_JOB_ID for W&B continuity (unless --new-wandb flag is set)
if [ "${new_wandb:-false}" = true ]; then
    job_id_for_wandb=$SLURM_JOB_ID
    echo "Using --new-wandb flag, forcing new W&B run with SLURM_JOB_ID: $SLURM_JOB_ID"
elif [ "$ckpt_path" != "$default_ckpt_path" ]; then
    if [ -z "$RESUME_JOB_ID" ]; then
        echo "ERROR: RESUME_JOB_ID must be set when resuming from a non-default checkpoint"
        echo "This ensures W&B logging continues in the original run."
        echo "Usage: RESUME_JOB_ID=<original_slurm_job_id> sbatch ... OR use --new-wandb to start fresh"
        exit 1
    fi
    job_id_for_wandb=$RESUME_JOB_ID
    echo "Resuming from custom checkpoint, using RESUME_JOB_ID: $RESUME_JOB_ID"
else
    job_id_for_wandb=$SLURM_JOB_ID
fi

# Check if checkpoint file exists, set to null if not
ckpt_path=$( [[ -f "$ckpt_path" ]] && echo "\\\"$ckpt_path\\\"" || echo null )
echo "CHECKPOINT PATH! ckpt_path: $ckpt_path"
##############################################################################

################ WANDB RUN ID #####################
# Use job_id_for_wandb (either SLURM_JOB_ID or RESUME_JOB_ID) for W&B run continuity
export WANDB_RUN_ID=${task}_${frame_type}_${EXPERIMENT}_${arm}_${job_id_for_wandb}
echo "WANDB_RUN_ID: $WANDB_RUN_ID"
###############################################################

if [ -z "$CL_OUT" ] || [ "$CL_OUT" = "None" ]; then
    export CL_param=${CL}
else
    export CL_param=${CL_OUT}
fi

##################### PREFLIGHT CHECKLIST #####################
if [ "${skip_preflight:-false}" = false ]; then
    echo ""
    echo "============================================================"
    echo "                  PREFLIGHT CHECKLIST"
    echo "============================================================"
    echo "Script: $(basename $0)"
    echo "Variant: ${VARIANT}"
    echo "------------------------------------------------------------"

    # Checkpoint status
    echo ""
    echo "[CHECKPOINT]"
    if [ "$ckpt_path" = "null" ]; then
        echo "  Status: FRESH START (no checkpoint found)"
    elif [ "$ckpt_path" != "\\\"${default_ckpt_path}\\\"" ]; then
        echo "  Status: CUSTOM CHECKPOINT"
        echo "  Path: ${ckpt_path}"
    else
        echo "  Status: RESUMING from last.ckpt"
        echo "  Path: ${default_ckpt_path}"
    fi

    # WandB status
    echo ""
    echo "[WANDB]"
    echo "  Run ID: ${WANDB_RUN_ID}"
    if [ "${new_wandb:-false}" = true ]; then
        echo "  Mode: NEW RUN (--new-wandb flag)"
    elif [ "$job_id_for_wandb" = "$SLURM_JOB_ID" ]; then
        echo "  Mode: NEW RUN (using SLURM_JOB_ID)"
    else
        echo "  Mode: RESUME (continuing job ${job_id_for_wandb})"
    fi

    # Debug mode
    echo ""
    echo "[MODE]"
    echo "  Debug: $([ "$debug" = true ] && echo 'ENABLED' || echo 'DISABLED')"
    echo "  Trainer: ${trainer}"
    echo "  Logger: ${logger}"

    # Experiment params
    echo ""
    echo "[EXPERIMENT]"
    echo "  Task: ${task}"
    echo "  Arm: ${arm}"
    echo "  PG=${PG}, CL=${CL}, CL_OUT=${CL_OUT:-None}"
    echo "  Experiment: ${EXPERIMENT}"

    # Paths
    echo ""
    echo "[PATHS]"
    echo "  Hydra Dir: ${hydra_run_dir}"
    echo "  Config: ${config_name}"
    echo "  Data Config: ${DATA_CONFIG_PATH}"

    echo ""
    echo "============================================================"
    echo ""
fi
###############################################################

CMD="
source /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/eth_clariden/clariden.sh
source /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/eth_clariden/aws_env.sh
cd /iopsstor/scratch/cscs/jiaqchen/egomim_out
python /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/egomimic/trainHydra.py \
    --config-name=${config_name} \
    trainer=${trainer} \
    logger=${logger} \
    data=${DATA_CONFIG} \
    name=${task}_${frame_type} \
    description=${task}_${frame_type} \
    chosen_frame=${frame_type} \
    ckpt_path=${ckpt_path} \
    wandb_run_id=${WANDB_RUN_ID} \
    hydra.run.dir=${hydra_run_dir} \
    trainer.num_nodes=${SLURM_NNODES} \
    model.robomimic_model.head_specs.eve_${arm}.action_horizon=${CL_param} \
    model.robomimic_model.head_specs.eve_${arm}.model.act_seq=${CL_param} \
    model.robomimic_model.head_specs.eve_${arm}_actions_joints.action_horizon=${CL_param} \
    model.robomimic_model.head_specs.eve_${arm}_actions_joints.model.act_seq=${CL_param} \
    model.robomimic_model.stem_specs.eve_${arm}.state_ee_pose.input_dim=${ee_pose_dim} \
    model.robomimic_model.stem_specs.eve_${arm}.state_joint_positions.input_dim=${joint_dim} \
    model.robomimic_model.head_specs.eve_${arm}.infer_ac_dims.eve_${arm}=${ee_pose_dim} \
    model.robomimic_model.head_specs.eve_${arm}.model.act_dim=${ee_pose_dim} \
    model.robomimic_model.head_specs.eve_${arm}_actions_joints.infer_ac_dims.eve_${arm}=${joint_dim} \
    model.robomimic_model.head_specs.eve_${arm}_actions_joints.model.act_dim=${joint_dim} \
    model.robomimic_model.use_quat=${quat} \
    model.robomimic_model.use_delta=${delta} \
    \$@
"
srun bash -lc "$CMD"

echo "Job finished at: $(date)"
