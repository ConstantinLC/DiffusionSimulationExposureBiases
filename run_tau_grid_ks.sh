#!/bin/bash
# Run exploration + greedy schedule training for each tau value across N seeds.
# After all training, evaluates with eval_tau_runs.py and averages per-seed results.
#
# Usage:
#   bash run_tau_seeds.sh
#
# Environment overrides:
#   TAU_VALUES="1.01 1.03 1.05 1.10"   space-separated list of tau values
#   N_SEEDS=3                      number of independent seeds per tau
#   OUTPUT_DIR=results/tau_seeds   directory for eval outputs
#   DEVICE=cuda                    torch device

set -euo pipefail

TAU_VALUES="${TAU_VALUES:-1.01 1.03 1.05 1.10 1.20}"
N_SEEDS="${N_SEEDS:-3}"
TAU_GRID_BASE="./checkpoints/KuramotoSivashinsky/tau_grid"
OUTPUT_DIR="${OUTPUT_DIR:-results/tau_seeds}"
DEVICE="${DEVICE:-cuda}"

# ── helpers ────────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── training loop ──────────────────────────────────────────────────────────────
for tau in $TAU_VALUES; do
    tau_dir="$TAU_GRID_BASE/tau_${tau}"
    declare -a PIDS=()

    for seed in $(seq 1 "$N_SEEDS"); do
        run_dir="$tau_dir/run_$((seed - 1))"
        greedy_trained="$run_dir/greedy_trained"
        exploration_state="$run_dir/exploration_state.json"

        if [ -d "$greedy_trained" ]; then
            log "Skipping tau=$tau seed=$seed (already complete)"
            continue
        fi

        log "Launching tau=$tau seed=$seed (bg)"

        (
            if [ ! -f "$exploration_state" ]; then
                log "--- tau=$tau seed=$seed: Phase 1 Exploration ---"
                python train_exploration.py \
                    +experiment=ks_exploration \
                    checkpoint_dir="$run_dir" \
                    training.tau="$tau" \
                    training.seed="$seed"

                if [ ! -d "$run_dir" ]; then
                    log "ERROR: exploration run dir '$run_dir' not created (tau=$tau seed=$seed)"
                    exit 1
                fi
            else
                log "--- tau=$tau seed=$seed: Exploration already done, resuming from Phase 2+3 ---"
            fi

            log "--- tau=$tau seed=$seed: Phase 2+3 Greedy schedule + final training ---"
            python train_greedy_schedule.py \
                +experiment=ks_exploration \
                "training.exploration_run_dir=$run_dir" \
                training.tau="$tau" \
                training.seed="$seed"

            if [ ! -d "$greedy_trained" ]; then
                log "ERROR: greedy_trained dir '$greedy_trained' not found (tau=$tau seed=$seed)"
                exit 1
            fi

            log "=== Completed tau=$tau, seed=$seed ==="
        ) &
        PIDS+=($!)
    done

    # Wait for all seeds for this tau, collecting failures
    FAILED=0
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            log "ERROR: seed $((i+1)) failed for tau=$tau"
            FAILED=1
        fi
    done
    [ "$FAILED" -eq 1 ] && exit 1

    unset PIDS
    log "All seeds done for tau=$tau"
done

echo ""
log "=========================================="
log "  All training complete."
log "=========================================="

# ── evaluation ─────────────────────────────────────────────────────────────────
log "--- Running eval_tau_runs.py ---"
python eval_tau_runs.py \
    --runs_dir "$TAU_GRID_BASE" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

log "Done. Results saved to $OUTPUT_DIR"
