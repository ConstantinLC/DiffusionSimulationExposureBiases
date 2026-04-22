#!/bin/bash
# Kolmogorov — full baseline + adaptive schedule training.
#
# Methods trained (3 seeds each, run sequentially one method at a time):
#   unet     — plain 2-D U-Net
#   linear   — DiffusionModel with linear schedule
#   sigmoid  — DiffusionModel with sigmoid schedule
#   refiner  — PDE-Refiner
#   adaptive — exploration (Phase 1) + greedy schedule (Phase 2+3)
#
# Methods and seeds both run sequentially.  Existing run_N
# dirs (identified by the presence of config.json) are skipped.
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

N_SEEDS="${N_SEEDS:-1}"
TAU="${TAU:-1.05}"
GPUS="${GPUS:-}"
DEVICE="${DEVICE:-cuda}"
BASE_CKPT="./checkpoints/KolmogorovFlow"
LOGDIR="./logs/kolmo"

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

log "N_SEEDS=$N_SEEDS  TAU=$TAU  GPUS=${GPUS:-<all use $DEVICE>}"

# ── run_method_seeds: run N_SEEDS seeds for one baseline method sequentially,
#    exit 1 on any failure.
# Args: <method_name> <checkpoint_dir> <extra_hydra_args...>
run_method_seeds() {
    local method=$1 mdir=$2
    shift 2
    local extra=("$@")

    log "=== $method: starting ==="

    for ((i=0; i<N_SEEDS; i++)); do
        local run_dir="$mdir/run_$i"

        if [ -d "$run_dir" ] && [ -f "$run_dir/config.json" ]; then
            log "[skip] $method run_$i already complete"
            continue
        fi

        mkdir -p "$run_dir"
        local dev logf
        dev="$(device_for "$i")"
        logf="$LOGDIR/${method}_run${i}.log"
        log "[run] $method run_$i dev=$dev -> $logf"

        python train.py \
            +experiment=kolmo \
            checkpoint_dir="$run_dir" \
            training.device="$dev" \
            "${extra[@]}" \
            >"$logf" 2>&1 || { log "[FAIL] $method run_$i — see $logf"; exit 1; }

        log "[done] $method run_$i"
    done

    log "=== $method: all seeds done ==="
}

# ── run_adaptive_seeds: run N_SEEDS adaptive (exploration+greedy) seeds
#    sequentially.
run_adaptive_seeds() {
    local exp_base="$BASE_CKPT/exploration"
    log "=== adaptive: starting ==="

    for ((i=0; i<N_SEEDS; i++)); do
        local run_dir="$exp_base/run_$i"

        if [ -d "$run_dir" ] && [ -d "$run_dir/greedy_trained" ]; then
            log "[skip] adaptive run_$i already complete"
            continue
        fi

        mkdir -p "$run_dir"
        local dev logf
        dev="$(device_for "$i")"

        if [ -f "$run_dir/exploration_state.json" ]; then
            logf="$LOGDIR/adaptive_greedy_run${i}.log"
            log "[run] adaptive run_$i greedy-only dev=$dev -> $logf"
            python train_greedy_schedule.py \
                +experiment=kolmo_exploration \
                "training.exploration_run_dir=$run_dir" \
                training.tau="$TAU" \
                training.device="$dev" \
                >"$logf" 2>&1 || { log "[FAIL] adaptive run_$i — see $logf"; exit 1; }
        else
            logf="$LOGDIR/adaptive_run${i}.log"
            log "[run] adaptive run_$i full dev=$dev -> $logf"
            python train_exploration.py \
                +experiment=kolmo_exploration \
                checkpoint_dir="$run_dir" \
                training.tau="$TAU" \
                training.device="$dev" \
                >>"$logf" 2>&1 || { log "[FAIL] adaptive run_$i exploration — see $logf"; exit 1; }
            python train_greedy_schedule.py \
                +experiment=kolmo_exploration \
                "training.exploration_run_dir=$run_dir" \
                training.tau="$TAU" \
                training.device="$dev" \
                >>"$logf" 2>&1 || { log "[FAIL] adaptive run_$i greedy — see $logf"; exit 1; }
        fi

        log "[done] adaptive run_$i"
    done

    log "=== adaptive: all seeds done ==="
}

# ── Sequential method order ───────────────────────────────────────────────────
run_method_seeds edm "$BASE_CKPT/DiffusionModel_edm" model=diffusion model.diffSchedule=edm
run_method_seeds linear  "$BASE_CKPT/DiffusionModel_linear" model=diffusion model.diffSchedule=linear
run_method_seeds unet    "$BASE_CKPT/Unet2D"                model=unet_2d
run_method_seeds cosine "$BASE_CKPT/DiffusionModel_cosine" model=diffusion model.diffSchedule=cosine
run_method_seeds refiner "$BASE_CKPT/PDERefiner"            model=refiner
#run_adaptive_seeds

log "All done. Checkpoints in $BASE_CKPT"
