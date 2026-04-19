#!/bin/bash
# WeatherBench — baselines + adaptive schedule training.
#
# Step 1: All 4 baselines × N_SEEDS seeds launch in parallel.
# Step 2: Once baselines finish, adaptive exploration launches (N_SEEDS parallel).
# Step 3: Once exploration finishes, adaptive greedy training launches (N_SEEDS parallel).
#
# Baselines:
#   unet     — plain U-Net (no diffusion)
#   linear   — DiffusionModel with linear schedule
#   sigmoid  — DiffusionModel with sigmoid schedule
#   refiner  — PDE-Refiner
#
# Usage:
#   bash run_wb.sh
#   GPUS="0 1 2" bash run_wb.sh
#
# Environment overrides:
#   N_SEEDS=3          number of independent seeds
#   TAU=1.05           exploration threshold (adaptive only)
#   GPUS="0 1 2"       pin seeds to specific GPUs (round-robin); empty = DEVICE
#   DEVICE=cuda        fallback device when GPUS is unset

set -euo pipefail

N_SEEDS="${N_SEEDS:-3}"
TAU="${TAU:-1.05}"
GPUS="${GPUS:-}"
DEVICE="${DEVICE:-cuda}"
BASE_CKPT="./checkpoints/WeatherBench"
LOGDIR="./logs/wb"

mkdir -p "$LOGDIR"

log() { echo "[$(date '+%H:%M:%S')] [WB] $*"; }

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

# ── Step 1: baselines — all types × all seeds in parallel ─────────────────────
log "--- Baselines: launching all types × $N_SEEDS seeds in parallel ---"

declare -A baseline_args=(
    [unet]="model=unet_2d"
    [linear]="model=diffusion model.diffSchedule=linear"
    [sigmoid]="model=diffusion model.diffSchedule=sigmoid"
    [refiner]="model=refiner"
)

declare -A bl_pids
for btype in "${!baseline_args[@]}"; do
    extra="${baseline_args[$btype]}"
    for ((i=0; i<N_SEEDS; i++)); do
        dev="$(device_for $i)"
        ckpt="$BASE_CKPT/baselines/${btype}/seed$((i+1))"
        logf="$LOGDIR/baseline_${btype}_seed$((i+1)).log"
        mkdir -p "$ckpt"
        log "[bg] baseline $btype seed $((i+1)) dev=$dev -> $logf"
        # Baselines train for 2× the epochs of each adaptive training round (1001 → 2002)
        # shellcheck disable=SC2086
        python train.py \
            +experiment=weatherbench \
            $extra \
            checkpoint_dir="$ckpt" \
            training.device="$dev" \
            pretraining.num_epochs=201 \
            pretraining.T_max=201 \
            >"$logf" 2>&1 &
        bl_pids[${btype}_${i}]=$!
    done
done

bl_ok=0
for key in "${!bl_pids[@]}"; do
    if wait "${bl_pids[$key]}"; then
        log "[done] $key"
    else
        log "[FAIL] $key — see $LOGDIR/baseline_${key}.log"
        bl_ok=1
    fi
done
[ "$bl_ok" -eq 0 ] || { log "One or more baselines failed — aborting."; exit 1; }
log "All baselines complete."

# ── Step 2: adaptive exploration — N_SEEDS in parallel ────────────────────────
EXP_BASE="$BASE_CKPT/exploration"

declare -A run_dirs
for ((i=0; i<N_SEEDS; i++)); do
    run_dirs[$i]="$(next_run "$EXP_BASE")"
    log "adaptive seed $((i+1)) -> ${run_dirs[$i]}"
done

log "--- Adaptive: launching $N_SEEDS explorations in parallel ---"
declare -A exp_pids
for ((i=0; i<N_SEEDS; i++)); do
    dev="$(device_for $i)"
    logf="$LOGDIR/exploration_seed$((i+1)).log"
    log "[bg] exploration seed $((i+1)) dev=$dev -> $logf"
    python train_exploration.py \
        +experiment=wb_exploration \
        training.tau="$TAU" \
        training.device="$dev" \
        checkpoint_dir="$EXP_BASE" \
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

# ── Step 3: adaptive greedy — N_SEEDS in parallel ─────────────────────────────
log "--- Adaptive: launching $N_SEEDS greedy trainings in parallel ---"
declare -A greedy_pids
for ((i=0; i<N_SEEDS; i++)); do
    dev="$(device_for $i)"
    run_dir="${run_dirs[$i]}"
    logf="$LOGDIR/greedy_seed$((i+1)).log"
    log "[bg] greedy seed $((i+1)) dev=$dev run=$run_dir -> $logf"
    python train_greedy_schedule.py \
        +experiment=wb_exploration \
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

log "All done. Checkpoints in $BASE_CKPT"
