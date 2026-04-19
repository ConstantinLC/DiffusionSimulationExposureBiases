"""
Automated evaluation for all WeatherBench experiments.

Discovers checkpoint directories produced by run_wb.sh, groups them by
experiment type (unet, linear, sigmoid, refiner, adaptive), and calls
experiments/eval_kolmo.py for each group so all seeds are evaluated together.

Usage:
    python eval_wb_experiments.py [--device cuda] [--output_dir results/eval/wb]
                                  [--rollout_steps 55] [--dry_run]
"""

import os, sys, json, argparse, subprocess, glob
from pathlib import Path

ROOT     = Path(__file__).parent
EVAL_SCR = ROOT / "experiments" / "eval_kolmo.py"
BASE     = ROOT / "checkpoints" / "WeatherBench"

BASELINE_TYPES = ["unet", "linear", "sigmoid", "refiner"]


# ── checkpoint helpers ────────────────────────────────────────────────────────

def is_valid_ckpt(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    return (path / "best_model.pth").exists() or bool(list(path.glob("epoch_*.pth")))


def find_ckpts(pattern: str) -> list[Path]:
    return sorted(p for p in map(Path, glob.glob(pattern)) if p.is_dir() and is_valid_ckpt(p))


def find_model_subdirs(seed_dir: Path) -> list[Path]:
    """
    train.py saves into <seed_dir>/<ModelName>/ via get_run_dir_name.
    Return all valid checkpoint subdirs inside seed_dir.
    """
    return [p for p in seed_dir.iterdir() if p.is_dir() and is_valid_ckpt(p)]


# ── discovery ─────────────────────────────────────────────────────────────────

def discover_groups(base: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}

    # Baselines: checkpoints/WeatherBench/baselines/<type>/seed*/
    baselines_root = base / "baselines"
    if baselines_root.exists():
        for btype in BASELINE_TYPES:
            type_root = baselines_root / btype
            if not type_root.exists():
                continue
            ckpts: list[Path] = []
            for seed_dir in sorted(type_root.glob("seed*")):
                if not seed_dir.is_dir():
                    continue
                subdirs = find_model_subdirs(seed_dir)
                if subdirs:
                    ckpts.extend(subdirs)
                elif is_valid_ckpt(seed_dir):
                    ckpts.append(seed_dir)
            if ckpts:
                groups[f"baseline_{btype}"] = ckpts

    # Adaptive: checkpoints/WeatherBench/exploration/run_*/greedy_trained/
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
    parser.add_argument("--output_dir",    default="results/eval/wb")
    parser.add_argument("--rollout_steps", type=int, default=55)  # traj_length=56
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
