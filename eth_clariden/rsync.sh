#!/bin/bash
#SBATCH --job-name=egomim_sync
#SBATCH --account=a144
#SBATCH --output=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/rsync_logs/egomim_sync-%j.out
#SBATCH --error=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/rsync_logs/egomim_sync-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=normal
#SBATCH --nice=10000

set -euo pipefail

SRC="/iopsstor/scratch/cscs/jiaqchen/egomim_out/logs/"
DST="/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/egomim_out/logs/"
LOGDIR="/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/rsync_logs"
LOCKFILE="${LOGDIR}/egomim_sync.lock"

mkdir -p "$DST" "$LOGDIR"

# prevent overlap
exec 9>"$LOCKFILE"
flock -n 9 || exit 0

echo "=== $(date) ==="
echo "Syncing everything from $SRC to $DST"

rsync -a --partial --numeric-ids "$SRC" "$DST"

echo "Done."

# run again in 5 hours
sbatch --begin=now+5hours "$(realpath "$0")" >/dev/null