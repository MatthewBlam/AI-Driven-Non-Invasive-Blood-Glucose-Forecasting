"""
Hyperparameter sweep over the pooled split.

Serial:
    python src/sweep.py --data-dir data

Parallel (one command per terminal, then merge):
    python src/sweep.py --data-dir data --num-shards 6 --shard 0
    python src/sweep.py --data-dir data --num-shards 6 --shard 1
    ...
    python src/sweep.py --data-dir data --num-shards 6 --shard 5
    python src/sweep.py --merge
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from preprocess import load_segments
from train import run_pooled


SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]

SWEEPS = {
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2],
    "lr": [3e-4, 1e-3, 3e-3],
    "history": [6, 12, 24],
}

DEFAULTS = {
    "hidden_size": 64,
    "num_layers": 1,
    "lr": 1e-3,
    "history": 12,
    "ph": 30,
    "multi_horizon": False,
    "bidirectional": False,
    "batch_size": 256,
}

HEADER = [
    "sweep_param", "sweep_value", "seed",
    "test_rmse", "persistence_rmse", "skill",
    "val_rmse", "epochs", "params",
]


def build_jobs() -> list[tuple]:
    """Flat (sweep_param, value, seed) list so shards can split it evenly.

    sweep_param=None marks the shared default config: it appears as one of the
    values in every sweep list, so it trains once per seed and its row is
    written under all four params (deterministic training makes retraining it
    per param produce identical results)."""
    jobs: list[tuple] = [(None, None, seed) for seed in SEEDS]
    for sweep_param, sweep_values in SWEEPS.items():
        for value in sweep_values:
            if value == DEFAULTS[sweep_param]:
                continue  # covered by the shared default runs
            for seed in SEEDS:
                jobs.append((sweep_param, value, seed))
    return jobs


def analyze_results() -> None:
    """Per-param val RMSE stats plus paired per-seed deltas vs default.

    delta_vs_default pairs on seed (same seed = same split), which cancels
    split-difficulty variance. Select the frozen config on the val columns,
    never on test_mean: selecting on test is leakage."""
    import pandas as pd

    path = Path("results/sweep.csv")
    if not path.exists():
        print(f"{path} not found -- run the sweep (and --merge if sharded) first.")
        return

    df = pd.read_csv(path)
    print(f"{len(df)} rows, seeds: {sorted(df['seed'].unique())}")

    def fmt(v):
        return int(v) if isinstance(v, float) and v.is_integer() else v

    best = {}

    for param in SWEEPS:
        default_value = DEFAULTS[param]
        sub = df[df["sweep_param"] == param]

        # rows -> matrix: one row per seed, one column per swept value
        pivot = sub.pivot(index="seed", columns="sweep_value", values="val_rmse")
        delta = pivot.sub(pivot[default_value], axis=0)   # paired: value - default, per seed

        summary = pd.DataFrame({
            "val_mean": pivot.mean(),
            "val_std": pivot.std(),
            "delta_vs_default": delta.mean(),
            "delta_std": delta.std(),
            "test_mean": sub.pivot(index="seed", columns="sweep_value", values="test_rmse").mean(),
            "avg_epochs": sub.groupby("sweep_value")["epochs"].mean(),
            "params": sub.groupby("sweep_value")["params"].first(),
        }).round(3)

        print(f"\n=== {param} (default {default_value}) ===")
        print(summary.to_string())

        # Best setting via paired t-statistic: how many standard errors is the
        # delta away from zero? Threshold 3 sigma, not 2: the sweep tests 7
        # alternatives, and at 2 sigma the family-wide false-positive rate is
        # ~30% (winner's curse).
        n_seeds = len(pivot)
        stats = []
        for value in delta.columns:
            if value == default_value:
                continue
            d_mean = float(delta[value].mean())
            sem = float(delta[value].std()) / n_seeds ** 0.5
            stats.append((d_mean / sem if sem > 0 else 0.0, d_mean, value))

        clear = [s for s in stats if s[0] <= -3]
        borderline = [s for s in stats if -3 < s[0] <= -2]
        if clear:
            t, d_mean, value = min(clear, key=lambda s: s[1])
            best[param] = value
            print(f"--> best: {fmt(value)}  (delta {d_mean:+.3f}, t={t:.1f}: beats default beyond noise)")
        else:
            best[param] = default_value
            print(f"--> best: {fmt(default_value)}  (default -- no alternative beats it beyond noise)")
            if borderline:
                t, d_mean, value = min(borderline, key=lambda s: s[1])
                print(f"    borderline: {fmt(value)} (delta {d_mean:+.3f}, t={t:.1f}) -- "
                      f"adopt only if simpler/cheaper than the default")

    print("\nRecommended frozen config (validation-selected):")
    for param, value in best.items():
        print(f"  {param}: {fmt(value)}")




def merge_shards() -> None:
    """Concatenate results/sweep_shard*.csv into results/sweep.csv."""
    shard_paths = sorted(Path("results").glob("sweep_shard*.csv"))
    if not shard_paths:
        print("No results/sweep_shard*.csv files found.")
        return

    rows = []
    for path in shard_paths:
        with open(path, newline="") as f:
            rows.extend(list(csv.reader(f))[1:])

    # Guard against stale shard files (e.g. leftovers from a run with a
    # different --num-shards) and merging before all shards finished.
    keys = [(r[0], r[1], r[2]) for r in rows]
    expected = len(SEEDS) * sum(len(v) for v in SWEEPS.values())
    if len(keys) != len(set(keys)):
        print(f"ERROR: duplicate rows across {[p.name for p in shard_paths]} -- "
              f"stale shard files? Delete results/sweep_shard*.csv and rerun the sweep.")
        return
    if len(rows) != expected:
        print(f"WARNING: {len(rows)} rows, expected {expected}. "
              f"A shard may still be running or have crashed -- merging anyway.")

    # Order as a serial run would have: by param, then value, then seed
    param_order = {param: i for i, param in enumerate(SWEEPS)}
    rows.sort(key=lambda r: (param_order[r[0]], float(r[1]), int(r[2])))

    out_path = Path("results/sweep.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
    print(f"Merged {len(shard_paths)} shards ({len(rows)} rows) -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0, help="0-based shard index")
    parser.add_argument("--merge", action="store_true",
                        help="Merge sweep_shard*.csv into sweep.csv and exit (no training)")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze results/sweep.csv and exit (no training)")
    args = parser.parse_args()

    if args.analyze:
        analyze_results()
        return

    if args.merge:
        merge_shards()
        return

    if not (0 <= args.shard < args.num_shards):
        parser.error(f"--shard must be in [0, {args.num_shards - 1}] "
                     f"for --num-shards {args.num_shards}, got {args.shard}")

    torch.set_float32_matmul_precision("high")
    segments = load_segments(args.data_dir / "segments_split.npz")

    # Round-robin slice keeps the slow configs (history=24, num_layers=2)
    # spread across shards instead of piled onto one.
    jobs = build_jobs()[args.shard::args.num_shards]

    if args.num_shards > 1:
        out_path = Path(f"results/sweep_shard{args.shard}.csv")
        experiment_name = f"sweep_shard{args.shard}"  # own log dir: no version races
    else:
        out_path = Path("results/sweep.csv")
        experiment_name = "sweep"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        for i, (sweep_param, value, seed) in enumerate(jobs):
            label = "defaults" if sweep_param is None else f"{sweep_param}={value}"
            print(f"\n--- [{i + 1}/{len(jobs)}] {label}, seed={seed} ---")

            overrides = {} if sweep_param is None else {sweep_param: value}
            run_config = argparse.Namespace(**{**DEFAULTS, "seed": seed, **overrides})
            result = run_pooled(segments, run_config, experiment_name=experiment_name)

            test_rmse = result.get("test_rmse", 0)
            persistence = result["persistence_rmse"]
            skill = (persistence - test_rmse) / persistence
            metrics = [
                f"{test_rmse:.4f}", f"{persistence:.4f}", f"{skill:.4f}",
                f"{result.get('val_rmse', 0):.4f}",
                result["epochs"], result["params"],
            ]

            if sweep_param is None:
                for param in SWEEPS:
                    writer.writerow([param, DEFAULTS[param], seed, *metrics])
            else:
                writer.writerow([sweep_param, value, seed, *metrics])
            f.flush()

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
