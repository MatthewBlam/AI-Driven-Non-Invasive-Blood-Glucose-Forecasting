"""Aggregate pooled single-horizon runs into a mean +/- std table across seeds.

Scans results/ for the descriptively-named folders train.py writes -- `ph{PH}_seed{N}`
(e.g. ph60_seed43) -- recomputes test RMSE / persistence / skill from each
checkpoint, and prints the multi-seed single-horizon comparison table.
Multi-horizon runs (`ph{PH}_multi_seed{N}`) are intentionally skipped: their pooled
RMSE averages six horizons and isn't comparable here (use per_horizon_eval instead).

Writes results/horizon_multiseed.csv (one row per run) so the numbers are saved.

Usage:
    python src/aggregate_horizons.py
    python src/aggregate_horizons.py --detail     # also print every seed
    python src/aggregate_horizons.py --infix _mealstatic_TMYIS   # B2 horizon table
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from evaluate import iter_pooled_seeds


def discover_runs(results_dir: Path, infix: str) -> dict[int, list[int]]:
    """Scan results/ for ph{PH}{infix}_seed{N} folders -> {horizon: sorted seeds}.

    Discovery, not a hardcoded grid: a run trained at an unexpected horizon or seed
    must appear in the table, not silently vanish. (A refactor once replaced this
    scan with a fixed [30, 60, 120] x 42-52 grid -- an off-grid run then produced
    no row and no warning.) The anchored pattern keeps _multi/_bidir/_attn and
    other-infix folders out, exactly as the original regex did.
    """
    run_pattern = re.compile(rf"^ph(\d+){re.escape(infix)}_seed(\d+)$")
    found: dict[int, list[int]] = {}
    for path in sorted(results_dir.iterdir()):
        match = run_pattern.match(path.name)
        if path.is_dir() and match:
            found.setdefault(int(match.group(1)), []).append(int(match.group(2)))
    return {ph: sorted(seeds) for ph, seeds in found.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ph{PH}_seed{N} runs into a horizon table")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--detail", action="store_true", help="print every seed, not just the summary")
    parser.add_argument("--infix", type=str, default="",
                        help="folder infix for B2 runs, e.g. _mealstatic_TMYIS")
    args = parser.parse_args()

    from preprocess import load_segments
    segments = load_segments(args.data_dir / "segments_split.npz")

    discovered = discover_runs(args.results_dir, args.infix)

    all_rows = []
    by_horizon: dict[int, list[dict]] = {}
    for ph, seeds in sorted(discovered.items()):
        runs = []
        for seed, ckpt, y, preds, last, subjects in iter_pooled_seeds(
                segments, seeds=seeds, ph=ph, infix=args.infix,
                results_dir=args.results_dir):
            rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
            persistence = float(np.sqrt(np.mean((last - y) ** 2)))
            skill = (persistence - rmse) / persistence
            runs.append({"ph": ph, "seed": seed, "rmse": rmse,
                         "persistence": persistence, "skill": skill})
        if runs:
            by_horizon[ph] = sorted(runs, key=lambda d: d["seed"])
            all_rows.extend(by_horizon[ph])

    if not by_horizon:
        print(f"No runs found with infix={args.infix!r} in {args.results_dir}/")
        return

    def ms(a: np.ndarray, pct: bool = False) -> str:
        fmt = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.2f}")
        if len(a) > 1:
            return f"{fmt(a.mean())} +/- {fmt(a.std(ddof=1))}"
        return fmt(a.mean())

    label = "B2 LSTM" if args.infix else "LSTM"
    print(f"\n{'Horizon':>8} {'n':>3} {label + ' RMSE':>18} {'Persistence':>18} {'Skill':>16}")
    print("-" * 66)
    for ph in sorted(by_horizon):
        runs = by_horizon[ph]
        rmses = np.array([x["rmse"] for x in runs])
        persists = np.array([x["persistence"] for x in runs])
        skills = np.array([x["skill"] for x in runs])
        print(f"{'+' + str(ph):>8} {len(runs):>3} {ms(rmses):>18} {ms(persists):>18} {ms(skills, True):>16}")
        if args.detail:
            for x in runs:
                print(f"           seed {x['seed']}:  rmse {x['rmse']:6.2f}   "
                      f"persist {x['persistence']:6.2f}   skill {x['skill']:5.1%}")

    suffix = args.infix.strip("_") + "_" if args.infix else ""
    out = args.results_dir / f"horizon_{suffix}multiseed.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ph", "seed", "rmse", "persistence", "skill"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nsaved {out}  ({len(all_rows)} runs)")


if __name__ == "__main__":
    main()
