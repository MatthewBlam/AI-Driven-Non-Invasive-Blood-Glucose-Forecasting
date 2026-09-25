"""
Post-hoc analysis of a finished LOSO run.

Reproduces the glycemic-class stratification, the Wilcoxon significance test, and
the LOSO-vs-pooled macro/micro comparison. Works for both B1 (glucose-only) and B2
(meal-augmented) runs -- the run infix is derived from the --csv filename, so the
B2 folds are always compared against their own pooled checkpoints.

Usage:
    python src/analyze_loso.py                      # B1: all three steps
    python src/analyze_loso.py --no-pooled          # skips the heavy pooled-checkpoint pass
    python src/analyze_loso.py \
        --csv results/loso_mealstatic_TMYIS_results.csv    # B2: all three steps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

A1C_NORMAL, A1C_DIABETIC = 5.7, 6.5
CLASSES = ["normal", "pre-diabetic", "diabetic"]

_pad = lambda s: s.astype(int).map(lambda n: f"{n:03d}")

def stratify_by_class(results_csv: Path, bio_csv: Path, out_csv: Path) -> pd.DataFrame:
    """Join per-subject RMSE to glycemic class, print per-class stats,
    and save the merged table (adds `glycemic_class` and `skill` columns)."""
    results = pd.read_csv(results_csv)
    bio = pd.read_csv(bio_csv)

    results["subject"] = _pad(results["subject"])
    bio["subject"] = _pad(bio["subject"])

    bio["glycemic_class"] = pd.cut(
        bio["A1c PDL (Lab)"].astype(float),
        bins=[0, A1C_NORMAL, A1C_DIABETIC, 20],
        labels=CLASSES,
        right=False,
    )

    merged = results.merge(bio[["subject", "glycemic_class"]], on="subject")
    assert len(merged) == len(results), f"merge dropped subjects: {len(merged)} of {len(results)}"
    merged["skill"] = (merged["persistence_rmse"] - merged["test_rmse"]) / merged["persistence_rmse"]

    missing = int(merged["glycemic_class"].isna().sum())
    print("=" * 60, f"\nGlycemic-class stratification  (n={len(merged)}, {missing} missing A1c)\n" + "=" * 60)
    for cls in CLASSES:
        sub = merged[merged["glycemic_class"] == cls]
        if len(sub) == 0:
            continue
        print(f"{cls:>14} (n={len(sub):2d}):  "
              f"model {sub['test_rmse'].mean():5.2f} +/- {sub['test_rmse'].std(ddof=1):4.2f}  "
              f"persist {sub['persistence_rmse'].mean():5.2f}  "
              f"skill {sub['skill'].mean():5.1%}")

    merged.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")
    return merged


def wilcoxon_test(results_csv: Path) -> None:
    """Paired Wilcoxon signed-rank on all folds. Uses loso_results.csv
    directly (not the bio.csv join), so the headline test never depends on a
    subject being dropped by the merge."""
    results = pd.read_csv(results_csv)
    stat, p = wilcoxon(results["persistence_rmse"], results["test_rmse"], alternative="greater")
    n_wins = int((results["test_rmse"] < results["persistence_rmse"]).sum())
    print("\n" + "=" * 60, "\nWilcoxon signed-rank (H1: persistence > model)\n" + "=" * 60)
    print(f"n={len(results)}   statistic={stat:.1f}   p={p:.3e}   "
          f"(model wins {n_wins}/{len(results)} subjects)")


def loso_vs_pooled(results_csv: Path, results_dir: Path, segments_npz: Path,
                   infix: str = "") -> None:
    """LOSO vs pooled, each computed macro (per-subject mean) and micro
    (window-weighted). Compare macro-to-macro or micro-to-micro, never across.

    Uses evaluate.iter_pooled_seeds so B2 meal features are handled automatically."""
    from preprocess import load_segments
    from evaluate import iter_pooled_seeds

    results = pd.read_csv(results_csv)
    loso_macro = results["test_rmse"].mean()
    window_counts = results["test_windows"]
    loso_micro = float(np.sqrt((window_counts * results["test_rmse"] ** 2).sum() / window_counts.sum()))

    segments = load_segments(segments_npz)
    per_subject_rmses, seed_micros = [], []
    n_seeds = 0
    for seed, ckpt, y, preds, last_input, subjects in iter_pooled_seeds(
            segments, infix=infix, results_dir=results_dir):
        n_seeds += 1
        seed_micros.append(np.sqrt(np.mean((preds - y) ** 2)))
        for s in np.unique(subjects):
            m = subjects == s
            per_subject_rmses.append(np.sqrt(np.mean((preds[m] - y[m]) ** 2)))

    if not seed_micros:
        print("\n" + "=" * 60, "\nLOSO vs pooled\n" + "=" * 60)
        print(f"  no pooled runs with infix={infix!r} in {results_dir}/ -- run the pooled seeds first.")
        return

    pooled_macro = float(np.mean(per_subject_rmses))
    pooled_micro = float(np.mean(seed_micros))

    print("\n" + "=" * 60, "\nLOSO vs pooled (macro & micro)\n" + "=" * 60)
    print(f"LOSO    macro {loso_macro:5.2f}   micro {loso_micro:5.2f}   ({len(results)} subjects)")
    print(f"pooled  macro {pooled_macro:5.2f}   micro {pooled_micro:5.2f}   "
          f"({n_seeds} seeds, {len(per_subject_rmses)} subject-runs)")
    print("gaps (LOSO - pooled):")
    print(f"  macro vs macro:  {loso_macro - pooled_macro:+.2f} mg/dL")
    print(f"  micro vs micro:  {loso_micro - pooled_micro:+.2f} mg/dL")
    print(f"  naive (macro LOSO - micro pooled): {loso_macro - pooled_micro:+.2f}  <- mixes averaging")


def main() -> None:
    parser = argparse.ArgumentParser(description="LOSO post-hoc analysis")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="dir holding segments_split.npz")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--bio", type=Path, default=Path("dataset/CGMacros/bio.csv"))
    parser.add_argument("--csv", type=Path, default=None,
                        help="LOSO results CSV (default: results/loso_results.csv)")
    parser.add_argument("--infix", type=str, default=None,
                        help="pooled run infix for LOSO-vs-pooled, e.g. _mealstatic_TMYIS "
                             "(default: derived from the --csv filename)")
    parser.add_argument("--no-pooled", action="store_true",
                        help="skip the pooled-checkpoint pass")
    args = parser.parse_args()

    results_csv = args.csv or (args.results_dir / "loso_results.csv")
    if not results_csv.exists():
        print(f"{results_csv} not found -- run LOSO first.")
        return

    # train.py names the CSV loso{infix}_results.csv, so the stem both yields the
    # output name and encodes which pooled runs this LOSO belongs to. Refusing a
    # non-conforming stem guards two silent failures at once: an output name equal
    # to the input (the _results replace would be a no-op and overwrite a 60-90 min
    # run's results in place), and a forgotten --infix comparing B2 LOSO folds
    # against B1 pooled checkpoints.
    stem = results_csv.stem
    if not (stem.startswith("loso") and stem.endswith("_results")):
        raise SystemExit(
            f"--csv stem {stem!r} does not match loso{{infix}}_results -- rename the "
            f"file to what train.py writes (e.g. loso_mealstatic_TMYIS_results.csv)")
    derived_infix = stem[len("loso"):-len("_results")]

    infix = args.infix if args.infix is not None else derived_infix
    if infix != derived_infix:
        raise SystemExit(
            f"--infix {infix!r} contradicts the CSV name (which implies "
            f"{derived_infix!r}) -- comparing one run's LOSO folds against another "
            f"run's pooled checkpoints would look plausible and be wrong")

    out_csv = results_csv.parent / f"loso{infix}_with_class.csv"
    assert out_csv != results_csv

    stratify_by_class(results_csv, args.bio, out_csv)
    wilcoxon_test(results_csv)
    if not args.no_pooled:
        loso_vs_pooled(results_csv, args.results_dir, args.data_dir / "segments_split.npz",
                       infix=infix)


if __name__ == "__main__":
    main()
