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

TAU_VALUES="${TAU_VALUES:-1.03 1.05 1.10}"
N_SEEDS="${N_SEEDS:-2}"
TAU_GRID_BASE="./checkpoints/KuramotoSivashinsky/tau_grid"
MANIFEST="./tau_seeds_manifest.json"
OUTPUT_DIR="${OUTPUT_DIR:-results/tau_seeds}"
DEVICE="${DEVICE:-cuda}"

# ── helpers ────────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }

append_manifest() {
    local tau="$1"
    local ckpt_dir="$2"
    python3 - <<EOF
import json, fcntl
path = '$MANIFEST'
with open(path, 'r+') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    m = json.load(f)
    key = '$tau'
    if key not in m:
        m[key] = []
    m[key].append('$ckpt_dir')
    f.seek(0); f.truncate()
    json.dump(m, f, indent=2)
print(f'  manifest: tau=$tau has {len(m["$tau"])} run(s)')
EOF
}

# ── initialise manifest ────────────────────────────────────────────────────────
python3 -c "import json; json.dump({}, open('$MANIFEST', 'w'))"
log "Manifest initialised: $MANIFEST"

# ── training loop ──────────────────────────────────────────────────────────────
for tau in $TAU_VALUES; do
    tau_dir="$TAU_GRID_BASE/tau_${tau}"
    declare -a PIDS=()

    for seed in $(seq 1 "$N_SEEDS"); do
        # Each seed gets its own subdir so the auto-increment in train_exploration.py
        # doesn't race; run_dir is always run_0 within the per-seed subdir.
        seed_base="$tau_dir/seed_${seed}"
        run_dir="$seed_base/run_0"

        log "Launching tau=$tau seed=$seed (bg)"

        (
            log "--- tau=$tau seed=$seed: Phase 1 Exploration ---"
            python train_exploration.py \
                +experiment=ks_exploration \
                checkpoint_dir="$seed_base" \
                training.tau="$tau" \
                training.seed="$seed"

            if [ ! -d "$run_dir" ]; then
                log "ERROR: exploration run dir '$run_dir' not created (tau=$tau seed=$seed)"
                exit 1
            fi

            log "--- tau=$tau seed=$seed: Phase 2+3 Greedy schedule + final training ---"
            python train_greedy_schedule.py \
                +experiment=ks_exploration \
                "training.exploration_run_dir=$run_dir" \
                training.tau="$tau" \
                training.seed="$seed"

            greedy_trained="$run_dir/greedy_trained"
            if [ ! -d "$greedy_trained" ]; then
                log "ERROR: greedy_trained dir '$greedy_trained' not found (tau=$tau seed=$seed)"
                exit 1
            fi

            append_manifest "$tau" "$greedy_trained"
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
log "  Manifest: $MANIFEST"
log "=========================================="

# ── evaluation ─────────────────────────────────────────────────────────────────
log "--- Running eval_tau_runs.py with manifest ---"
python eval_tau_runs.py \
    --manifest "$MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

log "Done. Results saved to $OUTPUT_DIR"
