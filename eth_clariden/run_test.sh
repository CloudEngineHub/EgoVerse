#!/bin/bash
#SBATCH --job-name=test_requeue_counter
#SBATCH --account=a144
#SBATCH --output=counter-%j.out
#SBATCH --error=counter-%j.err
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --requeue
# Send SIGUSR1 10s before time limit so we can requeue cleanly
#SBATCH --signal=USR1@10

set -euo pipefail

# The sbatch script is executed by only one node (master)
echo "[sbatch-master] running on $(hostname)"
echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
echo "[sbatch-master] SLURM_NNODES: $SLURM_NNODES"
echo "[sbatch-master] SLURM_NODEID: $SLURM_NODEID"

# Function to run counter on each node
run_counter() {
  # Get node name for per-node log file
  NODE_NAME=$(hostname)
  LOG_FILE="${SLURM_SUBMIT_DIR:-$PWD}/counter_log_${NODE_NAME}.txt"
  
  # Start after the last written number if the log already exists
  start_from=1
  if [[ -f "$LOG_FILE" ]]; then
    last_line=$(tail -n 1 "$LOG_FILE" || true)
    if [[ "$last_line" =~ ^[0-9]+$ ]]; then
      start_from=$((last_line + 1))
    fi
  fi
  
  echo "[$NODE_NAME] Starting at number: $start_from"
  echo "[$NODE_NAME] Logging to: $LOG_FILE"
  
  # On SIGUSR1, request requeue and exit gracefully
  trap 'echo "[$NODE_NAME] $(date -Ins) SIGUSR1 received; requeuing..."; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
  
  # Run for ~60 seconds per allocation
  deadline=$((SECONDS + 60))
  n=$start_from
  while (( SECONDS < deadline )); do
    echo "$n" >> "$LOG_FILE"
    ((n++))
    sleep 1
  done
  
  # If we reach here without SIGUSR1, proactively requeue
  echo "[$NODE_NAME] $(date -Ins) Time slice ending; requeuing..."
  scontrol requeue "$SLURM_JOB_ID"
}

# Use srun to launch counter on each node
# Each node will run the counter function independently
srun bash -c "$(declare -f run_counter); run_counter"
