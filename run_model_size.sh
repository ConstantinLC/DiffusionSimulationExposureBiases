#!/bin/bash
# Run exploration + greedy schedule training for tau=1.05 across different model
# sizes (dataSize = base U-Net channel dim), repeated for N_SEEDS seeds.
# Within each seed, all explorations launch in parallel; once they finish, all
# greedy-schedule trainings launch in parallel. Seeds run sequentially.
#
# Usage:
#   bash run_model_size.sh
#
# Environment overrides:
#   TAU=1.05                        exploration threshold
#   DATASIZES="16 32 64 128"        U-Net base channel dims to sweep
#   N_SEEDS=3                       number of independent seeds per size
#   OUTPUT_DIR=results/model_size   directory for eval outputs
#   DEVICE=cuda                     torch device

set -euo pipefail

TAU="${TAU:-1.05}"
read -ra SIZES <<< "${DATASIZES:-16 32}" # 64 128
N_SEEDS="${N_SEEDS:-1}"
BASE="./checkpoints/KuramotoSivashinsky/model_size"
LOGDIR="./logs/model_size"
MANIFEST="./model_size_manifest.json"
OUTPUT_DIR="${OUTPUT_DIR:-results/model_size}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$LOGDIR"
python3 -c "import json; json.dump({}, open('$MANIFEST', 'w'))"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

next_run() {
    local base="$1"
    python3 - "$base" <<'EOF'
import os, sys
base = sys.argv[1]
os.makedirs(base, exist_ok=True)
i = 0
while os.path.exists(os.path.join(base, f'run_{i}')):
    i += 1
print(os.path.join(base, f'run_{i}'))
EOF
}

append_manifest() {
    local key="$1" ckpt_dir="$2"
    python3 - "$key" "$ckpt_dir" "$MANIFEST" <<'EOF'
import json, sys
key, ckpt_dir, path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    m = json.load(f)
m.setdefault(key, []).append(ckpt_dir)
with open(path, 'w') as f:
    json.dump(m, f, indent=2)
print(f'  manifest: dataSize={key} has {len(m[key])} run(s)')
EOF
}

# ══════════════════════════════════════════════════════════════════════════════
log "Sweeping dataSize over: ${SIZES[*]}  (tau=$TAU, seeds=$N_SEEDS)"

for seed in $(seq 1 "$N_SEEDS"); do
    log "=========================================="
    log "  Seed $seed / $N_SEEDS"
    log "=========================================="

    # ── Pre-allocate run dirs (no race: each size has its own subdir) ─────────
    declare -A run_dirs
    for size in "${SIZES[@]}"; do
        size_base="$BASE/size_$size"
        run_dirs[$size]="$(next_run "$size_base")"
        log "  dataSize=$size  ->  ${run_dirs[$size]}"
    done

    # ── Phase 1: launch all explorations in parallel ──────────────────────────
    log "--- Phase 1: launching ${#SIZES[@]} explorations in parallel ---"
    declare -A exp_pids
    for size in "${SIZES[@]}"; do
        size_base="$BASE/size_$size"
        logfile="$LOGDIR/exploration_size${size}_seed${seed}.log"
        log "  [bg] exploration dataSize=$size  ->  $logfile"
        python train_exploration.py \
            +experiment=ks_exploration \
            training.tau="$TAU" \
            "model.dataSize=[$size]" \
            checkpoint_dir="$size_base" \
            >"$logfile" 2>&1 &
        exp_pids[$size]=$!
    done

    exp_ok=0
    for size in "${SIZES[@]}"; do
        if wait "${exp_pids[$size]}"; then
            log "  [done] exploration dataSize=$size"
        else
            log "  [FAIL] exploration dataSize=$size — see $LOGDIR/exploration_size${size}_seed${seed}.log"
            exp_ok=1
        fi
    done
    [ "$exp_ok" -eq 0 ] || { log "One or more explorations failed — aborting."; exit 1; }

    for size in "${SIZES[@]}"; do
        [ -d "${run_dirs[$size]}" ] || {
            log "ERROR: expected dir '${run_dirs[$size]}' not found after exploration."
            exit 1
        }
    done

    # ── Phase 2+3: launch all greedy schedules in parallel ───────────────────
    log "--- Phase 2+3: launching ${#SIZES[@]} greedy-schedule trainings in parallel ---"
    declare -A greedy_pids
    for size in "${SIZES[@]}"; do
        run_dir="${run_dirs[$size]}"
        logfile="$LOGDIR/greedy_size${size}_seed${seed}.log"
        log "  [bg] greedy dataSize=$size  run_dir=$run_dir  ->  $logfile"
        python train_greedy_schedule.py \
            +experiment=ks_exploration \
            "training.exploration_run_dir=$run_dir" \
            training.tau="$TAU" \
            "model.dataSize=[$size]" \
            >"$logfile" 2>&1 &
        greedy_pids[$size]=$!
    done

    greedy_ok=0
    for size in "${SIZES[@]}"; do
        if wait "${greedy_pids[$size]}"; then
            log "  [done] greedy dataSize=$size"
            greedy_trained="${run_dirs[$size]}/greedy_trained"
            [ -d "$greedy_trained" ] || {
                log "ERROR: '$greedy_trained' missing after greedy training."
                exit 1
            }
            append_manifest "$size" "$greedy_trained"
        else
            log "  [FAIL] greedy dataSize=$size — see $LOGDIR/greedy_size${size}_seed${seed}.log"
            greedy_ok=1
        fi
    done
    [ "$greedy_ok" -eq 0 ] || { log "One or more greedy trainings failed — aborting."; exit 1; }

    unset exp_pids greedy_pids run_dirs
    declare -A exp_pids greedy_pids run_dirs   # reset for next seed

    log "=== Seed $seed complete ==="
done

# ── Evaluation ────────────────────────────────────────────────────────────────
log "=========================================="
log "  All training complete. Running evaluation."
log "=========================================="
python eval_model_size_runs.py \
    --manifest "$MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

log "Done. Results in $OUTPUT_DIR"
