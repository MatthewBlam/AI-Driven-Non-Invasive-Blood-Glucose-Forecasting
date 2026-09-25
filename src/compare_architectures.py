"""
Paired architecture comparison: LSTM vs BiLSTM vs Attention-LSTM.

Recomputes test RMSE from every ph{PH}[_bidir|_attn]_seed{N} checkpoint and
compares each variant to the plain LSTM *paired per seed* -- subtract within a
seed, then average. Split difficulty dominates the raw spread (+/- 1.5 mg/dL at
+30), so an unpaired mean-vs-mean comparison cannot see a real effect of the size
an architecture change produces. The paired delta cancels it.

Only seeds present for ALL architectures are used, so every comparison is over
the same splits; anything dropped is printed, never silently ignored.

Writes results/architecture_comparison.csv (one row per run).

Usage:
    python src/compare_architectures.py
    python src/compare_architectures.py --ph 30 --detail    # also print every seed
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

# (display name, folder infix). train.py builds names as ph{PH}[_bidir][_attn]_seed{N},
# so the plain LSTM has no infix and is the baseline every variant is paired against.
ARCHITECTURES = [
    ("LSTM", ""),
    ("BiLSTM", "bidir_"),
    ("Attention-LSTM", "attn_"),
]
BASELINE = "LSTM"


def find_runs(results_dir: Path, ph: int, infix: str) -> dict[int, Path]:
    """Map seed -> run folder for one architecture."""
    pattern = re.compile(rf"^ph{ph}_{infix}seed(\d+)$")
    runs = {}
    for folder in sorted(results_dir.iterdir()):
        match = pattern.match(folder.name)
        if folder.is_dir() and match:
            runs[int(match.group(1))] = folder
    return runs


def evaluate_run(folder: Path, seed: int, ph: int, segments) -> dict | None:
    """Recompute test RMSE + persistence for one run, on that seed's own split."""
    import lightning as L
    import torch
    from dataset import CGMacrosDataModule
    from model import count_params
    from train import LitGlucoseLSTM, compute_persistence_rmse

    ckpts = sorted(folder.glob("**/checkpoints/*.ckpt"))
    if not ckpts:
        return None

    L.seed_everything(seed, workers=True)
    model = LitGlucoseLSTM.load_from_checkpoint(ckpts[0], map_location="cpu")
    model.eval()

    # seed= must be the run's own seed: it determines the split, so a mismatch
    # would score the model on windows it trained on.
    dm = CGMacrosDataModule(segments, history=12, horizon=ph // 5, batch_size=256,
                            split_mode="pooled_random", multi_horizon=False, seed=seed)
    dm.setup()
    with torch.no_grad():
        preds = model(dm.test_dataset.inputs).numpy().squeeze()
    y = dm.test_dataset.targets.numpy().squeeze()

    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    persistence, _ = compute_persistence_rmse(dm)
    return {"seed": seed, "rmse": rmse, "persistence": persistence,
            "skill": (persistence - rmse) / persistence,
            "params": count_params(model.network)}


def collect(results_dir: Path, ph: int, segments) -> pd.DataFrame:
    """Evaluate every discovered run. Returns one row per (architecture, seed)."""
    rows = []
    for name, infix in ARCHITECTURES:
        runs = find_runs(results_dir, ph, infix)
        if not runs:
            print(f"  {name:>15}: no ph{ph}_{infix}seed* folders found")
            continue
        for seed, folder in sorted(runs.items()):
            result = evaluate_run(folder, seed, ph, segments)
            if result is None:
                print(f"  ({folder.name}: no checkpoint, skipping)")
                continue
            rows.append({"architecture": name, **result})
        print(f"  {name:>15}: {sum(r['architecture'] == name for r in rows)} runs")
    return pd.DataFrame(rows)


def check_persistence_consistency(df: pd.DataFrame) -> None:
    """Persistence depends only on the split, so at a given seed it must be
    identical across architectures. If it isn't, a split is not reproducing and
    every paired number below is meaningless -- so fail loudly rather than
    report a comparison built on mismatched test sets."""
    spread = df.groupby("seed")["persistence"].agg(lambda s: s.max() - s.min())
    bad = spread[spread > 1e-6]
    if len(bad):
        raise AssertionError(
            "persistence differs across architectures at the same seed "
            f"(max spread {bad.max():.4f} at seed(s) {list(bad.index)}) -- "
            "the splits are not reproducing; do not trust these results."
        )


def paired_report(df: pd.DataFrame, detail: bool = False) -> None:
    """Summary table, then each variant paired against the baseline LSTM."""
    base = df[df["architecture"] == BASELINE].set_index("seed")
    if base.empty:
        print(f"\nno {BASELINE} runs found -- nothing to pair against.")
        return

    print("\n" + "=" * 72, f"\nSummary (unpaired means -- read the paired section below)\n" + "=" * 72)
    print(f"{'Architecture':>15} {'n':>3} {'RMSE':>16} {'Persistence':>13} {'Skill':>8} {'Params':>8}")
    for name, _ in ARCHITECTURES:
        arch_rows = df[df["architecture"] == name]
        if arch_rows.empty:
            continue
        rmse_values = arch_rows["rmse"].to_numpy()
        spread_str = f" +/- {rmse_values.std(ddof=1):.2f}" if len(rmse_values) > 1 else ""
        print(f"{name:>15} {len(arch_rows):>3} {f'{rmse_values.mean():.2f}{spread_str}':>16} "
              f"{arch_rows['persistence'].mean():>13.2f} {arch_rows['skill'].mean():>8.1%} "
              f"{int(arch_rows['params'].iloc[0]):>8,}")

    for name, _ in ARCHITECTURES:
        if name == BASELINE:
            continue
        variant = df[df["architecture"] == name].set_index("seed")
        if variant.empty:
            continue

        # Pair on the intersection only -- comparing different seed sets would
        # reintroduce exactly the split difficulty the pairing exists to remove.
        shared = sorted(set(base.index) & set(variant.index))
        dropped = sorted((set(base.index) | set(variant.index)) - set(shared))
        if not shared:
            print(f"\n{name}: no seeds shared with {BASELINE} -- cannot pair.")
            continue

        baseline_rmse = base.loc[shared, "rmse"].to_numpy()
        variant_rmse = variant.loc[shared, "rmse"].to_numpy()
        deltas = baseline_rmse - variant_rmse   # positive = variant better (lower RMSE)

        print("\n" + "=" * 72, f"\n{name} vs {BASELINE}  (paired over {len(shared)} seeds)\n" + "=" * 72)
        if dropped:
            print(f"  NOTE: seeds {dropped} missing from one side, excluded from the pairing")

        standard_error = deltas.std(ddof=1) / np.sqrt(len(deltas))
        print(f"  {BASELINE:>15}: {baseline_rmse.mean():.2f} +/- {baseline_rmse.std(ddof=1):.2f}")
        print(f"  {name:>15}: {variant_rmse.mean():.2f} +/- {variant_rmse.std(ddof=1):.2f}")
        print(f"  paired delta ({BASELINE} - {name}), positive = {name} better:")
        print(f"      {deltas.mean():+.3f} +/- {deltas.std(ddof=1):.3f} mg/dL   "
              f"(wins {int((deltas > 0).sum())}/{len(deltas)} seeds)")
        print(f"      95% CI [{deltas.mean() - 1.96 * standard_error:+.3f}, {deltas.mean() + 1.96 * standard_error:+.3f}]")

        # Two-sided: we have no directional hypothesis, we're asking "is it different?"
        if np.allclose(deltas, 0):
            print("      (identical on every seed -- no test run)")
        else:
            print(f"      Wilcoxon p = {wilcoxon(baseline_rmse, variant_rmse).pvalue:.4f}   "
                  f"paired t-test p = {ttest_rel(baseline_rmse, variant_rmse).pvalue:.4f}")

        if detail:
            print(f"\n  {'seed':>6} {BASELINE:>10} {name:>16} {'delta':>9}")
            for seed_id, base_val, var_val, delta_val in zip(shared, baseline_rmse, variant_rmse, deltas):
                print(f"  {seed_id:>6} {base_val:>10.2f} {var_val:>16.2f} {delta_val:>+9.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired architecture comparison")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--ph", type=int, default=30)
    parser.add_argument("--detail", action="store_true", help="print every seed's delta")
    parser.add_argument("--out", type=Path, default=None,
                        help="CSV path (default: <results-dir>/architecture_comparison.csv)")
    args = parser.parse_args()

    from preprocess import load_segments
    segments = load_segments(args.data_dir / "segments_split.npz")

    print(f"Evaluating ph{args.ph} runs:")
    df = collect(args.results_dir, args.ph, segments)
    if df.empty:
        print(f"\nNo runs found in {args.results_dir}/ for ph{args.ph}.")
        return

    check_persistence_consistency(df)
    paired_report(df, detail=args.detail)

    out = args.out or args.results_dir / "architecture_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved {out}  ({len(df)} runs)")


if __name__ == "__main__":
    main()
