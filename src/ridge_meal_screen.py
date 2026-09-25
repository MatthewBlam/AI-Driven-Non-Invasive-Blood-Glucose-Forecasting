"""Ridge-regression screen comparing meal-feature encodings against a CGM-only baseline.

Uses the production feature pipeline (meals.py / meal_features.py)
with the same 11 seeds and subject-disjoint splits as Baseline 1.
Runs on CPU in a few minutes.

Usage:
    python src/ridge_meal_screen.py
    python src/ridge_meal_screen.py --detail --horizon 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baselines import ridge_fit, ridge_predict, rmse, HISTORY
from dataset import split_subjects
from meal_features import build_all_features, block_names, DEFAULT_BLOCKS
from meals import (load_meals, fit_macro_cleaning, apply_macro_cleaning,
                   scale_macros)
from preprocess import load_segments, make_windows, fit_scaler, apply_scaler

SEEDS = tuple(range(42, 53))
FASTING_CUTOFF_MIN = 300

BLOCK_SETS = [
    ("CGM-only",          ()),
    ("timing (T)",        ("T",)),
    ("carbs (T,C)",       ("T", "C")),
    ("macros (T,M)",      ("T", "M")),
    ("type (T,M,Y)",      ("T", "M", "Y")),
    ("all (T,M,Y,I,S)",   ("T", "M", "Y", "I", "S")),
    ("impulse only (I)",  ("I",)),
]

STRATA = [
    ("meal <=30 min",     lambda dt: dt <= 30),
    ("meal 30-60 min",    lambda dt: (dt > 30) & (dt <= 60)),
    ("meal 60-120 min",   lambda dt: (dt > 60) & (dt <= 120)),
    ("fasting >=5 h",     lambda dt: dt >= FASTING_CUTOFF_MIN),
]


def build_ridge_data(segments, feature_matrices, mean, std, horizon_steps):
    """Build aligned arrays from segments + their feature matrices."""
    all_cgm, all_meal, all_y = [], [], []

    for seg, feat in zip(segments, feature_matrices):
        X, targets = make_windows(seg, HISTORY, horizon_steps)
        if len(X) == 0:
            continue
        num_windows = len(X)
        scaled_cgm = apply_scaler(X, mean, std)
        # Grab each window's meal features at its final (prediction-time) index
        window_features = feat[np.arange(num_windows) + HISTORY - 1]

        all_cgm.append(scaled_cgm)
        all_meal.append(window_features)
        all_y.append(targets)

    if not all_cgm:
        num_feature_cols = feature_matrices[0].shape[1] if feature_matrices else 0
        return np.empty((0, HISTORY)), np.empty((0, num_feature_cols)), np.empty((0,))

    cgm = np.concatenate(all_cgm)
    meal = np.concatenate(all_meal)
    y = np.concatenate(all_y)
    return cgm, meal, y


def select_block_cols(blocks, all_blocks=DEFAULT_BLOCKS):
    """Which columns of the full feature matrix belong to the requested blocks."""
    all_names = block_names(all_blocks)
    wanted = set(block_names(blocks))
    return [i for i, name in enumerate(all_names) if name in wanted]


def run_one_seed(segments, meals, seed, horizon_steps, all_blocks):
    subjects = sorted({s.subject for s in segments})
    train_subs, _, test_subs = split_subjects(subjects, seed=seed)

    segs_by_subj = {}
    for seg in segments:
        segs_by_subj.setdefault(seg.subject, []).append(seg)

    train_segs = [s for subj in train_subs for s in segs_by_subj.get(subj, [])]
    test_segs = [s for subj in test_subs for s in segs_by_subj.get(subj, [])]

    mean, std = fit_scaler(train_segs)
    cleaning = fit_macro_cleaning(meals, train_subs)
    ready = scale_macros(apply_macro_cleaning(meals, cleaning))

    train_feats = build_all_features(train_segs, ready, all_blocks)
    test_feats = build_all_features(test_segs, ready, all_blocks)

    train_cgm, train_meal, train_targets = build_ridge_data(
        train_segs, train_feats, mean, std, horizon_steps)
    test_cgm, test_meal, test_targets = build_ridge_data(
        test_segs, test_feats, mean, std, horizon_steps)

    last_input = test_cgm[:, -1] * std + mean
    persistence = rmse(test_targets - last_input)

    # Recover minutes-since-last-meal at each prediction time
    feature_names = block_names(all_blocks)
    dt_col = feature_names.index("dt_norm")
    minutes_since_meal = test_meal[:, dt_col] * FASTING_CUTOFF_MIN

    results = {"_persistence": persistence, "_n_test": len(test_targets)}

    # Fit and predict for each block set; store predictions for stratification
    preds_store = {}
    for label, blocks in BLOCK_SETS:
        if not blocks:
            X_train, X_test = train_cgm, test_cgm
        else:
            cols = select_block_cols(blocks, all_blocks)
            X_train = np.hstack([train_cgm, train_meal[:, cols]])
            X_test = np.hstack([test_cgm, test_meal[:, cols]])

        weights = ridge_fit(X_train, train_targets, lam=1.0)
        preds = ridge_predict(X_test, weights)
        preds_store[label] = preds
        results[label] = {"rmse": rmse(preds - test_targets)}

    # Stratified results: CGM-only vs all blocks
    preds_cgm = preds_store["CGM-only"]
    preds_all = preds_store["all (T,M,Y,I,S)"]
    strata = {}
    for name, mask_fn in STRATA:
        mask = mask_fn(minutes_since_meal)
        count = int(mask.sum())
        if count == 0:
            continue
        strata[name] = {
            "n": count,
            "persist_rmse": rmse(test_targets[mask] - last_input[mask]),
            "cgm_rmse": rmse(preds_cgm[mask] - test_targets[mask]),
            "all_rmse": rmse(preds_all[mask] - test_targets[mask]),
        }
    results["_strata"] = strata
    results["_dt_test"] = minutes_since_meal

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/CGMacros"))
    parser.add_argument("--horizon", type=int, default=30,
                        help="prediction horizon in minutes")
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--seeds", type=str, default=None, help="e.g. '42,43,44'")
    args = parser.parse_args()

    segments = load_segments(args.data_dir / "segments_split.npz")
    meals = load_meals(args.raw_dir)
    horizon_steps = args.horizon // 5
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(SEEDS)
    all_blocks = DEFAULT_BLOCKS

    print(f"Ridge meal screen: {len(seeds)} seeds, +{args.horizon} min horizon")
    print(f"  segments: {len(segments)}, meals: {len(meals)}")

    all_results = {}
    for seed in seeds:
        print(f"  seed {seed}...", end="", flush=True)
        all_results[seed] = run_one_seed(segments, meals, seed, horizon_steps, all_blocks)
        print(f" done (CGM-only {all_results[seed]['CGM-only']['rmse']:.2f})")

    # ---------------------------------------------------------------- headline table
    print(f"\n{'='*76}")
    print(f"Ridge oracle, +{args.horizon} min, {len(seeds)} seeds, "
          f"paired vs CGM-only ridge")
    print(f"{'='*76}")

    cgm_rmses = np.array([all_results[s]["CGM-only"]["rmse"] for s in seeds])
    std_str = f" +/- {cgm_rmses.std(ddof=1):.2f}" if len(seeds) > 1 else ""
    print(f"\n  CGM-only ridge baseline: {cgm_rmses.mean():.2f}{std_str}")
    persist = np.array([all_results[s]["_persistence"] for s in seeds])
    std_str = f" +/- {persist.std(ddof=1):.2f}" if len(seeds) > 1 else ""
    print(f"  Persistence:             {persist.mean():.2f}{std_str}")

    print(f"\n  {'Encoding':<25} {'RMSE':>8} {'Paired D':>10} {'Wins':>7}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*7}")

    for label, blocks in BLOCK_SETS:
        if not blocks:
            continue
        rmses = np.array([all_results[s][label]["rmse"] for s in seeds])
        deltas = cgm_rmses - rmses
        wins = int((deltas > 0).sum())
        delta_str = f"+{deltas.mean():.3f}" if deltas.mean() >= 0 else f"{deltas.mean():.3f}"
        print(f"  {label:<25} {rmses.mean():>8.2f} {delta_str:>10} "
              f"{wins:>3}/{len(seeds)}")

    # ---------------------------------------------------------------- per-seed detail
    if args.detail:
        print(f"\n  Per-seed detail:")
        headers = [label[:12] for label, _ in BLOCK_SETS]
        print(f"  {'seed':>6}" + "".join(f"  {header:>12}" for header in headers))
        for seed in seeds:
            print(f"  {seed:>6}", end="")
            for label, _ in BLOCK_SETS:
                print(f"  {all_results[seed][label]['rmse']:>12.2f}", end="")
            print()

    # ---------------------------------------------------------------- stratified
    print(f"\n  Stratified, CGM-only vs CGM+all, mean across {len(seeds)} seeds:")
    print(f"  {'Stratum':<25} {'n':>8} {'persist':>9} {'CGM':>9} "
          f"{'CGM+meal':>9} {'skill':>7}")
    print(f"  {'-'*25} {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*7}")
    strata_keys = list(all_results[seeds[0]]["_strata"].keys())
    for key in strata_keys:
        counts = [all_results[s]["_strata"][key]["n"] for s in seeds]
        persist_vals = [all_results[s]["_strata"][key]["persist_rmse"] for s in seeds]
        cgm_vals = [all_results[s]["_strata"][key]["cgm_rmse"] for s in seeds]
        all_vals = [all_results[s]["_strata"][key]["all_rmse"] for s in seeds]
        cgm_mean, all_mean = np.mean(cgm_vals), np.mean(all_vals)
        skill = (cgm_mean - all_mean) / cgm_mean * 100 if cgm_mean > 0 else 0
        print(f"  {key:<25} {np.mean(counts):>8.0f} {np.mean(persist_vals):>9.2f} "
              f"{cgm_mean:>9.2f} {all_mean:>9.2f} {skill:>+6.1f}%")

    # ---------------------------------------------------------------- prediction check
    print(f"\n  PREDICTION CHECK:")
    print(f"  Prototype baseline: timing +0.648 vs macros +0.549, gap +0.10")
    print(f"  Expected with missing-macro fix: macro rows improve, gap NARROWS.")

    timing_deltas = cgm_rmses - np.array([all_results[s]["timing (T)"]["rmse"] for s in seeds])
    macro_deltas = cgm_rmses - np.array([all_results[s]["macros (T,M)"]["rmse"] for s in seeds])
    gap = timing_deltas.mean() - macro_deltas.mean()
    print(f"  Measured: timing +{timing_deltas.mean():.3f}, macros +{macro_deltas.mean():.3f}, "
          f"gap {gap:+.3f}")
    if gap < 0:
        print(f"  -> Gap REVERSED (macros now better). The prediction was right:")
        print(f"     the missing-macro fix improved macro performance enough to overtake timing.")
    elif gap < 0.10:
        print(f"  -> Gap NARROWED from +0.10 to {gap:+.3f}. Prediction confirmed.")
    else:
        print(f"  -> Gap HELD or WIDENED. Prediction was wrong.")


if __name__ == "__main__":
    main()
