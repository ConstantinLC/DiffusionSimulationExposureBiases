#!/bin/bash
# Kuramoto–Sivashinsky — full baseline + adaptive schedule training.
#
# Methods trained (3 seeds each, run sequentially one method at a time):
#   unet     — plain 1-D U-Net
#   linear   — DiffusionModel with linear schedule
#   cosine   — DiffusionModel with cosine schedule
#   edm      — DiffusionModel with EDM schedule
#   refiner  — PDE-Refiner
#   adaptive — exploration (Phase 1) + greedy schedule (Phase 2+3)
#
# For each method the N_SEEDS seeds are launched in parallel; the script waits
# for all seeds to finish before moving to the next method.  Existing run_N
# dirs (identified by the presence of config.json) are skipped.
#
# Usage:
#   bash run_ks.sh
#   GPUS="0 1 2" bash run_ks.sh
#
# Environment overrides:
#   N_SEEDS=3          number of independent seeds
#   TAU=1.05           exploration threshold (adaptive only)
#   GPUS="0 1 2"       pin seeds to specific GPUs (round-robin); empty = DEVICE
#   DEVICE=cuda        fallback device when GPUS is unset

set -euo pipefail

N_SEEDS="${N_SEEDS:-3}"
TAU="${TAU:-1.1}"
GPUS="${GPUS:-}"
DEVICE="${DEVICE:-cuda}"
BASE_CKPT="./checkpoints/KuramotoSivashinsky"
LOGDIR="./logs/ks"

mkdir -p "$LOGDIR"

log() { echo "[$(date '+%H:%M:%S')] [KS] $*"; }

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

# ── run_method_seeds: launch N_SEEDS seeds for one baseline method in parallel,
#    wait for all, exit 1 on any failure.
# Args: <method_name> <checkpoint_dir> <extra_hydra_args...>
run_method_seeds() {
    local method=$1 mdir=$2
    shift 2
    local extra=("$@")

    log "=== $method: starting ==="
    declare -A pids logs

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
        log "[bg] $method run_$i dev=$dev -> $logf"

        python train.py \
            +experiment=ks \
            checkpoint_dir="$run_dir" \
            training.device="$dev" \
            "${extra[@]}" \
            >"$logf" 2>&1 &

        pids[$i]=$!
        logs[$i]="$logf"
    done

    local ok=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            log "[done] $method run_$i"
        else
            log "[FAIL] $method run_$i — see ${logs[$i]}"
            ok=1
        fi
    done
    [ "$ok" -eq 0 ] || { log "$method failed — aborting."; exit 1; }
    log "=== $method: all seeds done ==="
}

# ── run_adaptive_seeds: launch N_SEEDS adaptive (exploration+greedy) seeds in
#    parallel, wait for all.
run_adaptive_seeds() {
    local exp_base="$BASE_CKPT/exploration"
    log "=== adaptive: starting ==="
    declare -A pids logs

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
            log "[bg] adaptive run_$i greedy-only dev=$dev -> $logf"
            python train_greedy_schedule.py \
                +experiment=ks_exploration \
                "training.exploration_run_dir=$run_dir" \
                training.tau="$TAU" \
                training.device="$dev" \
                >"$logf" 2>&1 &
        else
            logf="$LOGDIR/adaptive_run${i}.log"
            log "[bg] adaptive run_$i full dev=$dev -> $logf"
            (
                python train_exploration.py \
                    +experiment=ks_exploration \
                    checkpoint_dir="$run_dir" \
                    training.tau="$TAU" \
                    training.device="$dev" \
                    >>"$logf" 2>&1 \
                && python train_greedy_schedule.py \
                    +experiment=ks_exploration \
                    "training.exploration_run_dir=$run_dir" \
                    training.tau="$TAU" \
                    training.device="$dev" \
                    >>"$logf" 2>&1
            ) &
        fi

        pids[$i]=$!
        logs[$i]="$logf"
    done

    local ok=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            log "[done] adaptive run_$i"
        else
            log "[FAIL] adaptive run_$i — see ${logs[$i]}"
            ok=1
        fi
    done
    [ "$ok" -eq 0 ] || { log "adaptive failed — aborting."; exit 1; }
    log "=== adaptive: all seeds done ==="
}

# ── Sequential method order ───────────────────────────────────────────────────
run_method_seeds refiner "$BASE_CKPT/PDERefiner"            model=refiner model.log_sigma_min=-2.5
run_method_seeds unet    "$BASE_CKPT/Unet1D"                model=unet_1d
run_method_seeds linear  "$BASE_CKPT/DiffusionModel_linear" model=diffusion model.diffSchedule=linear
run_method_seeds cosine  "$BASE_CKPT/DiffusionModel_cosine" model=diffusion model.diffSchedule=cosine
run_method_seeds edm     "$BASE_CKPT/DiffusionModel_edm"    model=diffusion model.diffSchedule=edm
#run_adaptive_seeds

log "All done. Checkpoints in $BASE_CKPT"
