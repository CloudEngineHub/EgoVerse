# Cup Experiments

## Pipeline Overview

```
1. faive2lerobot conversion   →  produces lerobot-format datasets
2. configure data YAMLs       →  point to converted data, set splits/filters
3. configure run_*.sh          →  set variant, SBATCH params, flags
4. launch experiments          →  sbatch run_*.sh
```

---

## Step 1: faive2lerobot Conversion

The conversion script `faive2lerobot/convert_faive_to_lerobot.py` converts FAIVE H5 trajectory data to LeRobot parquet format.

### Input/Output Paths

**Input (raw FAIVE H5 data):**
- Original cup data: `/capstor/store/cscs/swissai/a144/agavryus/datasets/egoverse/put_cup_on_saucer_processed_50hz`
- Additional cup data: `/iopsstor/scratch/cscs/jiaqchen/50hz/additional_data/cup_on_saucer_robot_raw_processed_50hz`

**Output (LeRobot format):**
- EVE: `/iopsstor/scratch/cscs/jiaqchen/data/EGOMIM/srl_data/output/release_2_0/50hz/pg1_cl75_clout100/`
- Aria: S3 bucket `rldb` (accessed via `S3RLDBDataset`)

### Conversion Scripts

Located in `faive2lerobot/run_conversion_scripts/`:
- `run_conversion_cup_on_saucer_original.sh` - Original cup dataset
- `run_conversion_cup_on_saucer_additional.sh` - Additional cup data
- `run_conversion.sh` - Small debug dataset

### Key Conversion Parameters

| Parameter | Value (cup) | Description |
|---|---|---|
| `--name` | `cup_lerobot_base_frame` | Output dataset name |
| `--arm` | `both` | Bimanual (left+right) |
| `--fps` | `50` | Data frequency in Hz |
| `--base-frame` | (set) | Use robot base frame (identity extrinsics) |
| `--extrinsics-key` | `best_calib` | Camera calibration (auto-set to `identity` for base-frame) |
| `--prestack` | (set) | Pre-compute action chunks |
| `--POINT_GAP_ACT` | `1` | Gap between points in action chunk |
| `--CHUNK_LENGTH_ACT` | `75` | Action chunk length (75 steps @ 50Hz = 1.5s) |
| `--CHUNK_LENGTH_ACT_OUT` | `100` | Resample chunks to this length |

### What the Conversion Does

1. **Reads H5 files** containing:
   - `observations/images/<camera>/color` - RGB images
   - `observations/qpos_arm_left`, `observations/qpos_arm_right` - Arm cartesian positions (xyz + quaternion)
   - `observations/qpos_hand_left`, `observations/qpos_hand_right` - Hand joint angles
   - `actions_arm_left`, `actions_arm_right` - Target arm positions
   - `actions_hand_left`, `actions_hand_right` - Target hand joint angles

2. **Transforms coordinates:**
   - `--base-frame`: Uses identity extrinsics (data stays in robot base frame)
   - `--cam-frame`: Transforms to camera frame using `best_calib` extrinsics
   - `--ee-frame`: End-effector relative frame (observations=zeros, actions=deltas)

3. **Processes orientations:**
   - Default: Euler angles (yaw-pitch-roll) - 6D per arm (xyz + ypr)
   - `--quat`: Quaternions - 7D per arm (xyz + xyzw)

4. **Creates action chunks:**
   - Samples future actions at `POINT_GAP_ACT` intervals
   - Creates chunks of length `CHUNK_LENGTH_ACT`
   - Interpolates to `CHUNK_LENGTH_ACT_OUT` (uses SLERP for quaternions, unwrap/wrap for Euler)

5. **Output structure:**
   - `observations.images.<camera>` - RGB images
   - `observations.state.cartesian_arm` - Raw arm pose in robot frame
   - `observations.state.ee_pose` - Transformed pose (used as qpos during training)
   - `observations.state.joints_hand` - Hand joint angles
   - `actions_cartesian` - Prestacked arm action chunks
   - `actions_joints` - Prestacked hand action chunks
   - `metadata.embodiment` - Robot embodiment ID

### Optional Flags

| Flag | Description |
|---|---|
| `--quat` | Use quaternions instead of Euler angles |
| `--delta` | Compute actions as deltas from first value of each chunk |
| `--actions-for-qpos` | Use actions instead of qpos for state observations |
| `--debug` | Process only 3 episodes for testing |
| `--overwrite` | Overwrite existing output |

### Running Conversion

```bash
cd /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/faive2lerobot
sbatch run_conversion_scripts/run_conversion_cup_on_saucer_original.sh
```

---

## Step 2: Data Configs

Hydra data configs live in `egomimic/hydra_configs/data/cup/`.

Each experiment variant has a matching `multi_data_<VARIANT>.yaml`:

| Config file | EVE data | Aria in-domain | Aria EgoVerse | Notes |
|---|---|---|---|---|
| `multi_data_BC.yaml` | original | - | - | Baseline, EVE-only |
| `multi_data_BC_delta.yaml` | TODO | - | - | Delta actions variant |
| `multi_data_BC+1ID.yaml` | original + additional | 1 (eth in-domain) | - | |
| `multi_data_BC+2ID.yaml` | original + additional | 2 (eth in-domain) | - | |
| `multi_data_BC+0ID+8EV.yaml` | original + additional | - | 8 (song, wang, eth, ...) | EgoVerse only, no in-domain |
| `multi_data_BC+2ID+8EV.yaml` | original + additional | 2 (eth in-domain) | 8 (song, wang, eth) | Full mix |
| `multi_data_BC+2ID+2EVeth.yaml` | original + additional | 2 (eth in-domain) | 2 (eth EgoVerse) | ETH-only EgoVerse |

**Key things to verify in data configs:**
- `root` paths point to the correct converted datasets
- `mode` is `train`/`valid` correctly
- `batch_size` and `num_workers` match node resources
- Aria `filters` (task, lab, operator) select the right subsets

---

## Step 3: Run Scripts

All run scripts share `common.sh` for training logic. Each `run_*.sh` sets:
- **SBATCH resources** (time, nodes, gpus) directly in `#SBATCH` directives
- **Experiment config** (VARIANT, DATA_CONFIG, etc.) in the script body

Each script includes a **clickable path comment** to the data config file for easy navigation.

### Script Structure

```bash
#!/bin/bash
#SBATCH --job-name=T_cup_<VARIANT>
#SBATCH --time=10:00:00          # Edit these SBATCH params as needed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
...

##################### EXPERIMENT CONFIG ####################
export VARIANT="BC+2ID+8EV"
export DATA_CONFIG="cup/multi_data_BC+2ID+8EV"
# Data config: /capstor/.../hydra_configs/data/cup/multi_data_BC+2ID+8EV.yaml  <- clickable!
export CONFIG_SUFFIX="_BC_aria"
export RLDB_WORKERS=32
############################################################
```

### Experiment Config Variables

| Variable | Description |
|---|---|
| `VARIANT` | Data variant name (e.g., `BC`, `BC+2ID+8EV`, `BC_delta`) |
| `DATA_CONFIG` | Hydra data config path (e.g. `cup/multi_data_BC+2ID+8EV`) |
| `CONFIG_SUFFIX` | `_BC` (EVE-only) or `_BC_aria` (EVE + Aria) |
| `RLDB_WORKERS` | Number of dataset loading workers |
| `delta` | (optional) Set to `true` for delta actions |

### Global Parameters (set in common.sh)

| Variable | Value | Description |
|---|---|---|
| `task` | `cup` | Task name |
| `frame_type` | `base_frame` | Coordinate frame |
| `arm` | `bimanual` | Robot arm config |
| `quat` | `false` | Use quaternion rotations |
| `actions_for_qpos` | `false` | Use actions for qpos |
| `delta` | `false` | Use delta/relative actions (can be overridden in run_*.sh) |

### Computed Variables (in common.sh)

| Variable | Example | Description |
|---|---|---|
| `PG_CL_EXPERIMENT` | `pg1_cl75_clout100` | Built from PG, CL, CL_OUT |
| `EXPERIMENT` | `BC+2ID+8EV_pg1_cl75_clout100` | `${VARIANT}_${PG_CL_EXPERIMENT}` - used in paths and W&B |

### Environment Variables (passed at sbatch time)

| Variable | Example | Description |
|---|---|---|
| `PG` | `1` | Point gap for action sampling |
| `CL` | `75` | Chunk length for actions |
| `CL_OUT` | `100` or `None` | Output chunk length (None = same as CL) |

### Flags

| Flag | Description |
|---|---|
| `--debug` | Use debug trainer/logger, skip W&B |
| `--new-wandb` | Force new W&B run instead of resuming |
| `--skip-preflight` | Skip the preflight checklist printout |

---

## Step 4: Launch Experiments

### Single experiment

```bash
PG=1 CL=75 CL_OUT=100 sbatch run_BC.sh
```

### All main experiments

```bash
# Baseline
PG=1 CL=75 CL_OUT=100 sbatch run_BC.sh

# Delta actions
PG=1 CL=75 CL_OUT=100 sbatch run_BC_delta.sh

# In-domain scaling
PG=1 CL=75 CL_OUT=100 sbatch run_BC+1ID.sh
PG=1 CL=75 CL_OUT=100 sbatch run_BC+2ID.sh

# EgoVerse scaling
PG=1 CL=75 CL_OUT=100 sbatch run_BC+0ID+8EV.sh
PG=1 CL=75 CL_OUT=100 sbatch run_BC+2ID+8EV.sh
PG=1 CL=75 CL_OUT=100 sbatch run_BC+2ID+2EVeth.sh
```

### Debug run

```bash
# run_BC_debug.sh has debug=true and uses debug partition
PG=1 CL=75 CL_OUT=100 sbatch run_BC_debug.sh
```

### Resume from checkpoint

```bash
# Resumes from last.ckpt automatically if it exists
PG=1 CL=75 CL_OUT=100 sbatch run_BC.sh

# Resume from specific checkpoint (needs original job ID for W&B continuity)
RESUME_JOB_ID=12345 PG=1 CL=75 CL_OUT=100 sbatch run_BC.sh
```

---

## Output Paths

- Checkpoints: `/iopsstor/scratch/cscs/jiaqchen/egomim_out/multi_node_v2/50hz/<EXPERIMENT>/cup/cup_base_frame/checkpoints/`
  - Example: `.../50hz/BC+2ID+8EV_pg1_cl75_clout100/cup/cup_base_frame/checkpoints/`
- SLURM logs: `/iopsstor/scratch/cscs/jiaqchen/egomim_out/multi_node_slurm_out_v2/50hz/<VARIANT>/cup/`
- W&B run ID: `cup_base_frame_<EXPERIMENT>_bimanual_<SLURM_JOB_ID>`
  - Example: `cup_base_frame_BC+2ID+8EV_pg1_cl75_clout100_bimanual_12345`