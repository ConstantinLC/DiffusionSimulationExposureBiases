"""
Automated evaluation for all KolmogorovFlow experiments.

Discovers checkpoint directories produced by run_kolmo.sh, groups them by
experiment type (currently: adaptive seeds), and calls experiments/eval_kolmo.py
for each group so all seeds are evaluated together and compared in one plot.

Usage:
    python eval_kolmo_experiments.py [--device cuda] [--output_dir results/eval/kolmo]
                                     [--rollout_steps 63] [--dry_run]
"""

import os, sys, json, argparse, subprocess, glob
from pathlib import Path

ROOT     = Path(__file__).parent
EVAL_SCR = ROOT / "experiments" / "eval_kolmo.py"
BASE     = ROOT / "checkpoints" / "KolmogorovFlow"


# ── checkpoint helpers ────────────────────────────────────────────────────────

def is_valid_ckpt(path: Path) -> bool:
    """Directory contains config.json and at least one weight file."""
    if not (path / "config.json").exists():
        return False
    has_weights = (path / "best_model.pth").exists() or bool(
        list(path.glob("epoch_*.pth"))
    )
    return has_weights


def find_ckpts(pattern: str) -> list[Path]:
    """Glob for directories matching pattern that are valid checkpoints."""
    return sorted(p for p in map(Path, glob.glob(pattern)) if p.is_dir() and is_valid_ckpt(p))


# ── discovery ─────────────────────────────────────────────────────────────────

def discover_groups(base: Path) -> dict[str, list[Path]]:
    """
    Returns {group_name: [ckpt_dir, ...]} for every discovered experiment group.
    """
    groups: dict[str, list[Path]] = {}

    # Adaptive: greedy_trained dirs inside exploration runs
    adaptive = find_ckpts(str(base / "exploration" / "run_*" / "greedy_trained"))
    if adaptive:
        groups["adaptive"] = adaptive

    return groups


# ── runner ────────────────────────────────────────────────────────────────────

def run_eval(group: str, ckpt_dirs: list[Path], output_dir: Path,
             rollout_steps: int, device: str, dry_run: bool) -> dict | None:
    out = output_dir / group
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(EVAL_SCR),
        "--checkpoint_dirs", *[str(p) for p in ckpt_dirs],
        "--output_dir", str(out),
        "--rollout_steps", str(rollout_steps),
        "--device", device,
    ]
    print(f"\n{'='*60}")
    print(f"  Group: {group}  ({len(ckpt_dirs)} checkpoint(s))")
    for p in ckpt_dirs:
        print(f"    {p}")
    print(f"  Output: {out}")
    if dry_run:
        print(f"  [dry-run] would run: {' '.join(cmd)}")
        return None

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [WARN] eval returned code {result.returncode} for group '{group}'")
        return None

    metrics_path = out / "metrics_summary.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--output_dir",    default="results/eval/kolmo")
    parser.add_argument("--rollout_steps", type=int, default=63)
    parser.add_argument("--dry_run",       action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    groups = discover_groups(BASE)

    if not groups:
        print(f"No checkpoints found under {BASE}. Have the training scripts been run?")
        sys.exit(0)

    print(f"Discovered {len(groups)} group(s) under {BASE}:")
    for name, paths in groups.items():
        print(f"  {name}: {len(paths)} checkpoint(s)")

    all_metrics: dict[str, dict] = {}
    for group, ckpt_dirs in groups.items():
        metrics = run_eval(group, ckpt_dirs, output_dir,
                           args.rollout_steps, args.device, args.dry_run)
        if metrics:
            all_metrics[group] = metrics

    if all_metrics and not args.dry_run:
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        print(f"\n{'='*60}")
        print("  Summary")
        print(f"{'='*60}")
        for group, metrics in all_metrics.items():
            print(f"\n  [{group}]")
            for model_name, m in metrics.items():
                ttf = m.get("time_to_failure_avg", float("nan"))
                s1  = m.get("step1_mse", float("nan"))
                print(f"    {model_name:40s}  ttf={ttf:.1f}  step1_mse={s1:.4e}")
        print(f"\n  Full summary → {summary_path}")


if __name__ == "__main__":
    main()
