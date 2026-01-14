#!/bin/bash
#SBATCH --job-name=egomimic_train_git
#SBATCH --account=a144
#SBATCH --output=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/logs/slurm-%j.out
#SBATCH --error=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --environment=/users/jiaqchen/.edf/faive2lerobot.toml

set -euo pipefail

REPO_URL="https://github.com/GaTech-RL2/EgoVerse.git"
COMMIT="${1:-main}"            # sbatch run_github.sh <commit-ish> [-- extra args to python]
shift || true                  # shift only if a commit arg was provided

LOG_DIR="/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/logs"
mkdir -p "$LOG_DIR"

WORKDIR=$(mktemp -d -p "${TMPDIR:-/tmp}" egoverserun-XXXXXX)
cleanup() {
  if [[ -d "$WORKDIR" ]]; then
    rm -rf "$WORKDIR"
  fi
}
trap cleanup EXIT

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Working directory (temp clone): $WORKDIR"
echo "Using commit: $COMMIT"
echo "Logs: $LOG_DIR/slurm-${SLURM_JOB_ID:-unknown}.out"

echo "Cloning repository to temp dir..."
git clone --no-checkout "$REPO_URL" "$WORKDIR"
cd "$WORKDIR"
git fetch --depth=1 origin "$COMMIT"
git checkout FETCH_HEAD
git submodule update --init --recursive

# Activate environment and train from the cloned copy
CMD="
source \"$WORKDIR/eth_clariden/clariden.sh\"
cd \"$WORKDIR\"
python \"$WORKDIR/egomimic/trainHydra.py\" \"$@\"
"

srun bash -c "$CMD"

echo "Job finished at: $(date)"
