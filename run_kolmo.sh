#!/bin/bash
# Kolmogorov Flow — adaptive schedule training.
# All N_SEEDS explorations launch in parallel; once done, all greedy
# schedule trainings launch in parallel.
#
# Usage:
#   bash run_kolmo.sh
#   GPUS="0 1 2" bash run_kolmo.sh
#
# Environment overrides:
#   N_SEEDS=3          number of independent seeds
#   TAU=1.05           exploration threshold
#   GPUS="0 1 2"       pin seeds to specific GPUs (round-robin); empty = DEVICE
#   DEVICE=cuda        fallback device when GPUS is unset

set -euo pipefail

N_SEEDS="${N_SEEDS:-3}"
TAU="${TAU:-1.05}"
GPUS="${GPUS:-}"
DEVICE="${DEVICE:-cuda}"
BASE="./checkpoints/KolmogorovFlow/exploration"
LOGDIR="./logs/kolmo"

mkdir -p "$LOGDIR"

log() { echo "[$(date '+%H:%M:%S')] [Kolmo] $*"; }

device_for() {
    local idx=$1
    if [ -n "$GPUS" ]; then
        local arr=($GPUS)
        echo "cuda:${arr[$((idx % ${#arr[@]}))]}"
    else
        echo "$DEVICE"
    fi
}

next_run() {
    python3 - "$1" <<'EOF'
import os, sys
base = sys.argv[1]
os.makedirs(base, exist_ok=True)
i = 0
while os.path.exists(os.path.join(base, f'run_{i}')):
    i += 1
print(os.path.join(base, f'run_{i}'))
EOF
}

log "N_SEEDS=$N_SEEDS  TAU=$TAU  GPUS=${GPUS:-<all use $DEVICE>}"

# ── Pre-allocate run dirs ─────────────────────────────────────────────────────
declare -A run_dirs
for ((i=0; i<N_SEEDS; i++)); do
    run_dirs[$i]="$(next_run "$BASE")"
    log "seed $((i+1)) -> ${run_dirs[$i]}"
done

# ── Phase 1: explorations in parallel ────────────────────────────────────────
log "--- Phase 1: launching $N_SEEDS explorations in parallel ---"
declare -A exp_pids
for ((i=0; i<N_SEEDS; i++)); do
    dev="$(device_for $i)"
    logf="$LOGDIR/exploration_seed$((i+1)).log"
    log "[bg] exploration seed $((i+1)) dev=$dev -> $logf"
    python train_exploration.py \
        +experiment=kolmo_exploration \
        training.tau="$TAU" \
        training.device="$dev" \
        checkpoint_dir="$BASE" \
        >"$logf" 2>&1 &
    exp_pids[$i]=$!
done

exp_ok=0
for ((i=0; i<N_SEEDS; i++)); do
    if wait "${exp_pids[$i]}"; then
        log "[done] exploration seed $((i+1))"
    else
        log "[FAIL] exploration seed $((i+1)) — see $LOGDIR/exploration_seed$((i+1)).log"
        exp_ok=1
    fi
done
[ "$exp_ok" -eq 0 ] || { log "Exploration failed — aborting."; exit 1; }

# ── Phase 2+3: greedy schedules in parallel ───────────────────────────────────
log "--- Phase 2+3: launching $N_SEEDS greedy trainings in parallel ---"
declare -A greedy_pids
for ((i=0; i<N_SEEDS; i++)); do
    dev="$(device_for $i)"
    run_dir="${run_dirs[$i]}"
    logf="$LOGDIR/greedy_seed$((i+1)).log"
    log "[bg] greedy seed $((i+1)) dev=$dev run=$run_dir -> $logf"
    python train_greedy_schedule.py \
        +experiment=kolmo_exploration \
        "training.exploration_run_dir=$run_dir" \
        training.tau="$TAU" \
        training.device="$dev" \
        >"$logf" 2>&1 &
    greedy_pids[$i]=$!
done

greedy_ok=0
for ((i=0; i<N_SEEDS; i++)); do
    if wait "${greedy_pids[$i]}"; then
        log "[done] greedy seed $((i+1))"
    else
        log "[FAIL] greedy seed $((i+1)) — see $LOGDIR/greedy_seed$((i+1)).log"
        greedy_ok=1
    fi
done
[ "$greedy_ok" -eq 0 ] || { log "Greedy training failed — aborting."; exit 1; }

log "All done. Checkpoints in $BASE"
