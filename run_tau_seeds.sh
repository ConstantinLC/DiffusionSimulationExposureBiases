#!/bin/bash
# Run exploration + greedy schedule training for each tau value across N seeds.
# After all training, evaluates with eval_tau_runs.py and averages per-seed results.
#
# Usage:
#   bash run_tau_seeds.sh
#
# Environment overrides:
#   TAU_VALUES="1.03 1.05 1.10"   space-separated list of tau values
#   N_SEEDS=3                      number of independent seeds per tau
#   OUTPUT_DIR=results/tau_seeds   directory for eval outputs
#   DEVICE=cuda                    torch device

set -euo pipefail

TAU_VALUES="${TAU_VALUES:-1.01 1.03 1.05 1.10}"
N_SEEDS="${N_SEEDS:-3}"
CHECKPOINT_BASE="./checkpoints/KuramotoSivashinsky/exploration"
MANIFEST="./tau_seeds_manifest.json"
OUTPUT_DIR="${OUTPUT_DIR:-results/tau_seeds}"
DEVICE="${DEVICE:-cuda}"

# ── helpers ────────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Return the next run_N index that does not yet exist under CHECKPOINT_BASE.
next_run_dir() {
    python3 - <<EOF
import os
base = '$CHECKPOINT_BASE'
os.makedirs(base, exist_ok=True)
i = 0
while os.path.exists(os.path.join(base, f'run_{i}')):
    i += 1
print(os.path.join(base, f'run_{i}'))
EOF
}

append_manifest() {
    local tau="$1"
    local ckpt_dir="$2"
    python3 - <<EOF
import json
path = '$MANIFEST'
with open(path) as f:
    m = json.load(f)
key = '$tau'
if key not in m:
    m[key] = []
m[key].append('$ckpt_dir')
with open(path, 'w') as f:
    json.dump(m, f, indent=2)
print(f'  manifest: tau={key} has {len(m[key])} run(s)')
EOF
}

# ── initialise manifest ────────────────────────────────────────────────────────
python3 -c "import json; json.dump({}, open('$MANIFEST', 'w'))"
log "Manifest initialised: $MANIFEST"

# ── training loop ──────────────────────────────────────────────────────────────
for tau in $TAU_VALUES; do
    for seed in $(seq 1 "$N_SEEDS"); do
        echo ""
        log "=========================================="
        log "  tau=$tau  |  seed=$seed / $N_SEEDS"
        log "=========================================="

        # Predict the run directory that exploration will create
        run_dir="$(next_run_dir)"
        log "Expected exploration run dir: $run_dir"

        # ── Phase 1: Exploration ───────────────────────────────────────────────
        log "--- Phase 1: Exploration ---"
        python train_exploration.py \
            +experiment=ks_exploration \
            training.tau="$tau"

        if [ ! -d "$run_dir" ]; then
            log "ERROR: exploration run dir '$run_dir' not created — aborting."
            exit 1
        fi

        # ── Phase 2+3: Greedy schedule construction + final training ──────────
        log "--- Phase 2+3: Greedy schedule + final training ---"
        python train_greedy_schedule.py \
            +experiment=ks_exploration \
            "training.exploration_run_dir=$run_dir" \
            training.tau="$tau"

        greedy_trained="$run_dir/greedy_trained"
        if [ ! -d "$greedy_trained" ]; then
            log "ERROR: greedy_trained dir '$greedy_trained' not found — aborting."
            exit 1
        fi

        append_manifest "$tau" "$greedy_trained"
        log "=== Completed tau=$tau, seed=$seed ==="
    done
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
