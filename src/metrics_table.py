"""Every number in the results table: NRMSE, error metrics, and Parkes zones.

NRMSE, the results table, and Parkes zone percentages are all columns of ONE table,
so they share one loop over the 11 pooled checkpoints instead of loading them three
times. The Parkes *figure* and remaining plots live in figures.py.

Everything here is inference over checkpoints that already exist -- no training.

Two invariants checked at runtime:
  - Every model number sits next to persistence from the same split.
  - Persistence at a given seed is identical across architectures (asserted).
    If it differs, the splits aren't reproducing and every paired number is invalid.

Outputs:
    results/metrics_table.csv   one tidy row per (architecture, seed) -- every raw number
    results/metrics_loso.csv    one row per (LOSO arm, fold)   (only with --loso)
    results/metrics_table.md    the paper-ready aggregated table, mean +/- std over seeds

Usage (from the project root):
    python src/metrics_table.py                 # 5 arms x 11 seeds + Parkes (~7 min)
    python src/metrics_table.py --no-parkes     # error metrics only            (~30s, see below)
    python src/metrics_table.py --loso          # add the LOSO rows: B1 + B2, 44 folds each
    python src/metrics_table.py --seeds 42      # quick single-seed check

No training -- inference only. ~93% of runtime is methcomp.parkeszones (pure-Python zone
polygon tests, ~2s per 22k-window call). --no-parkes removes nearly all of it.

Note on the `Time` column: wall-clock seconds are printed by run_pooled but never
persisted, so they cannot be recovered from a checkpoint. `best_epoch` (which can,
from the checkpoint filename) is reported instead; copy Time from the run log if
the paper needs it.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import (DEFAULT_SEEDS, RESULTS_DIR, checkpoint_params, find_checkpoint,
                      iter_pooled_seeds, load_loso_subject, load_test_predictions,
                      pooled_config, run_tag)
from preprocess import load_segments

# Fixed denominator -- NOT y.max()-y.min(). The per-seed range swings 205-360 mg/dL
# because it is decided by two extreme readings pinned at the Dexcom reporting
# limits, which INVERTS the seed ranking (seed 46 has the best RMSE and the worst
# per-seed NRMSE). The fixed sensor range avoids that instability.
SENSOR_RANGE = 400 - 40

# (label, folder infix) -- train.py names runs ph{PH}{infix}_seed{N}
# B3-LSTM is the null result: the row documents that the experiment ran.
ARCHITECTURES = [("LSTM", ""), ("BiLSTM", "_bidir"), ("Attn-LSTM", "_attn"),
                 ("B2-LSTM", "_mealstatic_TMYIS"), ("B3-LSTM", "_b3static_HRFAE")]

# The LOSO arms that were RUN (44 folds each, train.py --mode loso): B1 and the
# headline B2. load_loso_subject derives each fold's feature config from the
# infix, so a B2 fold can never be scored with B1 features. An arm whose results
# CSV is absent is skipped with a notice, never silently averaged around.
LOSO_ARMS = [("LSTM", ""), ("B2-LSTM", "_mealstatic_TMYIS")]

# 2 = primary: CGMacros is a normal / pre-diabetic / type-2 cohort with NO type 1
# participants. 1 = secondary conservative reference only (tighter zones); it is a
# sensitivity check, never a claim about type 1 performance.
DIABETES_TYPES = (2, 1)

try:
    from methcomp import parkeszones
except ImportError:                                     # pragma: no cover
    parkeszones = None


# ------------------------------------------------------------------ error metrics

def error_metrics(y, p, last) -> dict:
    """RMSE / MAE / NRMSE / skill for one split, model and persistence side by side."""
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    persist_rmse = float(np.sqrt(np.mean((last - y) ** 2)))
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(p - y))),
        "nrmse": rmse / SENSOR_RANGE,
        "persistence_rmse": persist_rmse,
        "persistence_mae": float(np.mean(np.abs(last - y))),
        "persistence_nrmse": persist_rmse / SENSOR_RANGE,
        "skill": (persist_rmse - rmse) / persist_rmse,
        # Dispersion: the model's spread vs the truth's. The ratio is the under-dispersion
        # finding as one number -- consistently ~0.87 pooled, i.e. the forecast varies ~13%
        # less than reality, which is why it cannot follow large excursions.
        "pred_std": float(np.std(p)),
        "actual_std": float(np.std(y)),
        "dispersion_ratio": float(np.std(p) / np.std(y)),
        # Recorded so the fixed denominator above is auditable rather than asserted.
        "split_range": float(y.max() - y.min()),
        "n_test": int(len(y)),
    }


def row_metrics(checkpoint_path, segments, config) -> dict:
    """One table row's error metrics straight from a checkpoint.

    The main loop below does NOT call this -- it needs y/preds/last for the Parkes
    and lag columns too, and going through here would load every checkpoint twice.
    Kept as the single-checkpoint entry point for interactive use."""
    y, p, last, _ = load_test_predictions(checkpoint_path, segments, config)
    assert y.ndim == 1, "table rows are single-horizon; use per_horizon_eval.py for multi-horizon"
    return error_metrics(y, p, last)


def effective_lag_minutes(y, p, subjects, step_min=5, max_steps=6) -> float:
    """Median over subjects of the shift (minutes) that best aligns prediction with
    actual. ~0 = in phase (real forecasting); ~30 = collapsed to persistence.
    Grouped by subject because a lag is only meaningful on a contiguous run."""
    lags = []
    for s in np.unique(subjects):
        idx = np.where(subjects == s)[0]
        ys, ps = y[idx], p[idx]
        if len(ys) < max_steps + 10:
            continue
        errs = [np.mean((ps - ys) ** 2)] + [
            np.mean((ps[k:] - ys[:-k]) ** 2) for k in range(1, max_steps + 1)
        ]
        lags.append(int(np.argmin(errs)) * step_min)
    return float(np.median(lags)) if lags else float("nan")


# ------------------------------------------------------------------ Parkes zones

def zone_pct(reference, test, dtype: int) -> dict:
    """Percent of points in each Parkes zone.

    reference (the TRUE value) first, test (the prediction) second. Swapping them
    does not raise -- it silently moves zone A by ~2pp while A+B barely budges, so
    the mistake is invisible if you only look at A+B."""
    zone_labels = parkeszones(dtype, reference, test, units="mgdl")
    counts, total = Counter(zone_labels), len(zone_labels)
    return {z: 100 * counts.get(z, 0) / total for z in "ABCDE"}


def parkes_columns(y, pred, dtype: int) -> dict:
    """The three reported Parkes numbers for one prediction array."""
    z = zone_pct(y, pred, dtype)
    return {f"p{dtype}_A": z["A"],
            f"p{dtype}_AB": z["A"] + z["B"],
            f"p{dtype}_DE": z["D"] + z["E"]}


# ------------------------------------------------------------------ bookkeeping

def best_epoch(ckpt) -> int:
    """Epoch of the best checkpoint, read out of Lightning's filename."""
    match = re.search(r"epoch=(\d+)", Path(ckpt).name)
    return int(match.group(1)) if match else -1


def fmt(values, digits=2, pct=False) -> str:
    """mean +/- std over seeds, or the bare value when there is only one."""
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return "--"
    scale, suffix = (100, "%") if pct else (1, "")
    if len(a) == 1:
        return f"{a[0] * scale:.{digits}f}{suffix}"
    return f"{a.mean() * scale:.{digits}f} +/- {a.std(ddof=1) * scale:.{digits}f}{suffix}"


def paired(model_vals, baseline_vals, label: str, unit: str = "") -> str:
    """Paired delta (model - baseline) per seed, with a win count."""
    deltas = np.asarray(model_vals, float) - np.asarray(baseline_vals, float)
    spread = f" +/- {deltas.std(ddof=1):.3f}" if len(deltas) > 1 else ""   # ddof=1 is nan at n=1
    return (f"{label}: {deltas.mean():+.3f}{spread}{unit}, "
            f"model ahead on {int((deltas > 0).sum())}/{len(deltas)} seeds")


# ------------------------------------------------------------------ the table

def pooled_rows(segments, seeds, ph: int, do_parkes: bool, results_dir: Path) -> list[dict]:
    """One row per (architecture, seed), every column computed from one load."""
    rows: list[dict] = []
    persistence_seen: dict[int, float] = {}      # seed -> persistence RMSE (split invariant)

    for label, infix in ARCHITECTURES:
        print(f"\n{label}:")
        for seed, ckpt, y, p, last, subjects in iter_pooled_seeds(
                segments, seeds=seeds, ph=ph, infix=infix, results_dir=results_dir):
            row = {"architecture": label, "split": "pooled", "ph": ph, "seed": seed}
            row.update(error_metrics(y, p, last))
            row["lag_min"] = effective_lag_minutes(y, p, subjects)
            row["persistence_lag_min"] = effective_lag_minutes(y, last, subjects)
            row["params"] = checkpoint_params(ckpt)
            row["best_epoch"] = best_epoch(ckpt)

            if do_parkes:
                for dtype in DIABETES_TYPES:
                    row.update(parkes_columns(y, p, dtype))
                    row.update({f"persistence_{k}": v
                                for k, v in parkes_columns(y, last, dtype).items()})

            # Persistence is split-determined: identical across architectures at a
            # given seed. If this trips, the splits are not reproducing and every
            # paired comparison in this file is meaningless.
            if seed in persistence_seen:
                assert np.isclose(row["persistence_rmse"], persistence_seen[seed], atol=1e-6), (
                    f"persistence RMSE differs at seed {seed}: "
                    f"{row['persistence_rmse']:.6f} vs {persistence_seen[seed]:.6f} "
                    f"-- the {label} split does not match the LSTM split")
            else:
                persistence_seen[seed] = row["persistence_rmse"]

            rows.append(row)
            print(f"  seed {seed}: RMSE {row['rmse']:.2f}  persist {row['persistence_rmse']:.2f}  "
                  f"skill {row['skill']:.1%}  NRMSE {row['nrmse']:.4f}"
                  + (f"  zoneA(t2) {row['p2_A']:.2f}%" if do_parkes else ""))
    return rows


def loso_rows(segments, do_parkes: bool, results_dir: Path,
              label: str = "LSTM", infix: str = "") -> list[dict]:
    """One row per LOSO fold of one arm, recomputed from that fold's own checkpoint.

    Also re-verifies each fold against the RMSE train.py logged in the arm's
    loso{infix}_results.csv -- a recomputed number that disagrees means the
    fold's split is not rebuilding."""
    logged = pd.read_csv(results_dir / f"loso{infix}_results.csv")
    logged["sid"] = logged["subject"].astype(int).map(lambda n: f"{n:03d}")

    rows, worst_delta = [], 0.0
    for sid in logged["sid"]:
        y, p, last, subjects = load_loso_subject(sid, segments,
                                                 results_dir=results_dir, infix=infix)
        row = {"architecture": label, "split": "loso", "ph": 30, "subject": sid}
        row.update(error_metrics(y, p, last))
        row["lag_min"] = effective_lag_minutes(y, p, subjects)
        row["persistence_lag_min"] = effective_lag_minutes(y, last, subjects)
        if do_parkes:
            for dtype in DIABETES_TYPES:
                row.update(parkes_columns(y, p, dtype))
                row.update({f"persistence_{k}": v
                            for k, v in parkes_columns(y, last, dtype).items()})

        expected = float(logged.loc[logged["sid"] == sid, "test_rmse"].iloc[0])
        delta = abs(row["rmse"] - expected)
        worst_delta = max(worst_delta, delta)
        rows.append(row)
        print(f"  {sid}: RMSE {row['rmse']:.2f} (logged {expected:.2f}, delta {delta:.4f})  "
              f"persist {row['persistence_rmse']:.2f}  skill {row['skill']:+.1%}")

    print(f"\n  max |recomputed - logged| RMSE across {len(rows)} folds: {worst_delta:.4f} mg/dL")
    assert worst_delta < 0.01, "a LOSO fold did not reproduce -- check the val_subjects rebuild"
    return rows


def markdown_table(pooled: pd.DataFrame, loso: pd.DataFrame | None, do_parkes: bool) -> str:
    """The paper-ready table: one row per model, mean +/- std across seeds."""
    header = ("| Model | Split | PH | RMSE (mg/dL) | MAE | NRMSE | Skill | "
              "Parkes A (t2) | A+B (t2) | D+E (t2) | Parkes A (t1) | Lag | Params | n |")
    sep = "|" + "---|" * 14
    lines = [header, sep]

    def add(label, split, sub, ph, rmse_col, mae_col, nrmse_col, skill, params, prefix=""):
        cols = [label, split, f"+{ph}", fmt(sub[rmse_col]), fmt(sub[mae_col]),
                fmt(sub[nrmse_col], digits=4), skill]
        for key in (f"{prefix}p2_A", f"{prefix}p2_AB", f"{prefix}p2_DE", f"{prefix}p1_A"):
            cols.append(fmt(sub[key]) if do_parkes and key in sub else "--")
        lag_col = "persistence_lag_min" if prefix else "lag_min"
        cols += [f"{np.nanmedian(sub[lag_col]):.0f}", params, str(len(sub))]
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")

    # Persistence first: every model row below is only meaningful next to it.
    lstm = pooled[pooled["architecture"] == "LSTM"]
    add("Persistence", "pooled", lstm, 30, "persistence_rmse", "persistence_mae",
        "persistence_nrmse", "--", "0", prefix="persistence_")

    params_by_arch = {}
    for label, _ in ARCHITECTURES:
        sub = pooled[pooled["architecture"] == label]
        if sub.empty:
            continue
        params_by_arch[label] = f"{int(sub['params'].iloc[0]):,}"
        add(label, "pooled", sub, 30, "rmse", "mae", "nrmse",
            fmt(sub["skill"], digits=1, pct=True), params_by_arch[label])

    if loso is not None and not loso.empty:
        # Persistence is fold-determined, identical across arms (asserted in
        # main), so ONE persistence row from a single arm's folds -- summing over
        # arms would double-count the 44 subjects.
        base = loso[loso["architecture"] == loso["architecture"].iloc[0]]
        add("Persistence", "LOSO", base, 30, "persistence_rmse", "persistence_mae",
            "persistence_nrmse", "--", "0", prefix="persistence_")
        for label, _ in LOSO_ARMS:
            sub = loso[loso["architecture"] == label]
            if sub.empty:
                continue
            add(label, "LOSO", sub, 30, "rmse", "mae", "nrmse",
                fmt(sub["skill"], digits=1, pct=True),
                params_by_arch.get(label, "--"))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Results table: NRMSE, error metrics, and Parkes zones")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--ph", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--no-parkes", action="store_true",
                        help="skip the Parkes columns (~4 min faster)")
    parser.add_argument("--loso", action="store_true",
                        help="also recompute all 44 LOSO folds for the LOSO row")
    args = parser.parse_args()

    # 44 folds x "Seed set to 42" is pure noise; keep the numbers readable.
    logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.WARNING)

    do_parkes = not args.no_parkes
    if do_parkes and parkeszones is None:
        print("methcomp not installed -- skipping Parkes columns (pip install methcomp)")
        do_parkes = False

    segments = load_segments(args.data_dir / "segments_split.npz")

    print(f"pooled runs: ph{args.ph}, seeds {args.seeds[0]}-{args.seeds[-1]}, "
          f"{len(ARCHITECTURES)} architectures, parkes={do_parkes}")
    pooled = pd.DataFrame(pooled_rows(segments, args.seeds, args.ph, do_parkes, args.results_dir))
    if pooled.empty:
        print("no pooled checkpoints found -- nothing to do")
        return

    csv_path = args.results_dir / "metrics_table.csv"
    pooled.to_csv(csv_path, index=False)

    # LOSO: recompute on request, otherwise reuse a previous run's cached rows.
    loso_path = args.results_dir / "metrics_loso.csv"
    loso = None
    if args.loso:
        frames = []
        for label, infix in LOSO_ARMS:
            arm_csv = args.results_dir / f"loso{infix}_results.csv"
            if not arm_csv.exists():
                print(f"\n  [skip] {arm_csv} not found -- the {label} LOSO arm has not been run")
                continue
            print(f"\nLOSO folds, {label} (recomputed from each fold's own checkpoint):")
            frames.append(pd.DataFrame(
                loso_rows(segments, do_parkes, args.results_dir, label, infix)))
        if frames:
            loso = pd.concat(frames, ignore_index=True)
            loso.to_csv(loso_path, index=False)
    elif loso_path.exists():
        loso = pd.read_csv(loso_path)
        arms = ", ".join(sorted(loso["architecture"].unique()))
        print(f"\nreusing LOSO rows from {loso_path} ({arms}; pass --loso to recompute)")

    # A fold's test set is fully determined by the held-out subject, so
    # persistence must be identical across arms fold by fold. If it drifts, the
    # arms were scored on different windows and no cross-arm LOSO comparison
    # (including the shared persistence row in the table) is valid.
    if loso is not None and loso["architecture"].nunique() > 1:
        by_subject = loso.pivot_table(index="subject", columns="architecture",
                                      values="persistence_rmse")
        worst = float((by_subject.max(axis=1) - by_subject.min(axis=1)).max())
        assert worst < 1e-6, (
            f"persistence RMSE differs across LOSO arms (worst spread {worst:.6f} "
            f"mg/dL) -- the arms are not scoring the same held-out windows")
        print(f"  cross-arm check: persistence identical fold-by-fold across "
              f"{loso['architecture'].nunique()} LOSO arms (worst spread {worst:.2e})")

    # ------------------------------------------------------------------ report
    lstm = pooled[pooled["architecture"] == "LSTM"]
    print("\n" + "=" * 78)
    print(f"  RESULTS TABLE  (pooled, +{args.ph}, n={len(lstm)} seeds)")
    print("=" * 78)
    table = markdown_table(pooled, loso, do_parkes)
    print(table)

    print(f"\nNRMSE, normalized by the fixed {SENSOR_RANGE} mg/dL sensor range "
          f"(40-400 Dexcom):")
    print(f"  LSTM        {fmt(lstm['nrmse'], digits=4)}   (= RMSE / {SENSOR_RANGE}; a rescaling, "
          f"not new information)")
    print(f"  persistence {fmt(lstm['persistence_nrmse'], digits=4)}")
    print(f"  this cohort's per-split range ran {lstm['split_range'].min():.0f}-"
          f"{lstm['split_range'].max():.0f} mg/dL "
          f"({lstm['split_range'].max() / lstm['split_range'].min():.2f}x) "
          f"-- why the denominator is fixed, not per-seed")

    print(f"\nPaired gaps (same split, so split difficulty cancels):")
    print("  " + paired(lstm["persistence_rmse"], lstm["rmse"], "RMSE gap (persist - model)",
                        " mg/dL"))
    for label, _ in ARCHITECTURES[1:]:
        sub = pooled[pooled["architecture"] == label]
        common = sorted(set(lstm["seed"]) & set(sub["seed"]))
        if not common:
            continue
        a = lstm.set_index("seed").loc[common, "rmse"]
        b = sub.set_index("seed").loc[common, "rmse"]
        print("  " + paired(a, b, f"LSTM - {label} (positive = {label} better)", " mg/dL"))

    if do_parkes:
        print(f"\nParkes error grid, paired per seed:")
        for dtype in DIABETES_TYPES:
            tag = "primary, matches the cohort" if dtype == 2 else "secondary reference only"
            print(f"  type {dtype} ({tag}):")
            print(f"    zone A: model {fmt(lstm[f'p{dtype}_A'])}  "
                  f"persistence {fmt(lstm[f'persistence_p{dtype}_A'])}")
            print("    " + paired(lstm[f"p{dtype}_A"], lstm[f"persistence_p{dtype}_A"],
                                  "zone A delta", " pp"))
            print("    " + paired(lstm[f"p{dtype}_AB"], lstm[f"persistence_p{dtype}_AB"],
                                  "A+B delta  ", " pp"))
            print(f"    D+E: model {fmt(lstm[f'p{dtype}_DE'])}  "
                  f"persistence {fmt(lstm[f'persistence_p{dtype}_DE'])}")
        print("  ^ A+B is saturated (both ~99.9%, delta ~0, a coin flip across seeds); zone A is\n"
              "    the discriminating metric. Type 1 is a conservative reference -- NO subject in\n"
              "    CGMacros has type 1 and the model never saw type 1 data.")

    md_path = args.results_dir / "metrics_table.md"
    note = ("\n\n_NRMSE denominator: fixed 360 mg/dL (40-400 Dexcom sensor range). "
            "Parkes type 2 is primary (the cohort has no type 1 participants); type 1 is a "
            "conservative reference only. A+B is saturated at +30 and does not discriminate "
            "between methods -- zone A is the signal. Lag is the median over subjects of the "
            "shift that best aligns prediction with truth: ~0 = forecasting, ~30 = echoing._\n")
    md_path.write_text(f"# Results table\n\n{table}{note}", encoding="utf-8")

    print(f"\nwrote {csv_path}  ({len(pooled)} rows)")
    if loso is not None and args.loso:
        print(f"wrote {loso_path}  ({len(loso)} rows)")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
