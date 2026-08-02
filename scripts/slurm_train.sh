#!/bin/bash
#SBATCH --job-name=bpd
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j.out
#
# SLURM launcher for Bellman Path Diffusion on a GPU node.
#
# Usage:
#   sbatch scripts/slurm_train.sh --config configs/hopper_medium.yaml \
#       --max_horizon 8 --steps_per_horizon 100000 --batch_size 2048
#
# The training script keeps the transition dataset resident on the GPU and uses
# AMP autocast + cuDNN/TF32; --cpus-per-task feeds the multi-threaded CPU-side
# work (indexing, replay bookkeeping) so the GPU does not wait on the host.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

# Route D4RL/MuJoCo caches under the repo (never $HOME).
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-$REPO_ROOT/data/d4rl}"
mkdir -p "$D4RL_DATASET_DIR"

# Match CPU thread pools to the allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "== node: $(hostname) =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Log GPU utilization every 10s in the background for the whole job.
( nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
    --format=csv -l 10 > "logs/gpu_util_${SLURM_JOB_ID:-local}.csv" 2>/dev/null ) &
UTIL_PID=$!
trap 'kill $UTIL_PID 2>/dev/null || true' EXIT

python3 -u scripts/train.py --device cuda --num_threads "${SLURM_CPUS_PER_TASK:-8}" "$@"
