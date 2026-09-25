"""Ridge screen for B3: do activity features add linear signal beyond CGM + meals?

The screen that decided B3's shape before any GPU run, re-homed from
scratch/b3_ridge_screen.py onto the project modules (activity.py / activity_features.py).
Same 11 seeds and subject-disjoint splits as B1/B2; paired deltas are measured against
the CGM+meal ridge (the B2-equivalent baseline), not CGM-only.

This re-run SUPERSEDES guide 12 section 0.4: the scratch screen aggregated raw minutes by
row position, which reads minutes older than t-4 near the 285 skipped raw rows (the
subject-003 bug caught by activity.py's dual-path verification), and it divided the
missing fraction by however many rows the window happened to hold instead of the 5 it
covers. This version inherits the timestamp-correct aggregation, so rows may differ from
the recorded table in the third decimal; anything larger means the restructure changed
the arithmetic and must be explained before any conclusion is drawn from it.

    python src/ridge_activity_screen.py                 # +30 min, 11 seeds
    python src/ridge_activity_screen.py --horizon 120
    python src/ridge_activity_screen.py --seeds 42,43
    python src/ridge_activity_screen.py --figure        # all three horizons ->
                                                        # activity_screen_horizons.png

Every run persists its per-seed numbers (the B2 horizon-CSV convention):
results/activity_screen_ph{H}.csv (table rows) and
results/activity_screen_strata_ph{H}.csv (strata); --figure writes all six.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from activity import aggregate_segments, fit_activity_stats, load_activity
from activity_features import ACTIVITY_BLOCKS, DT_ACTIVE_CLIP_MIN
from activity_features import build_features as build_activity_features
from baselines import HISTORY, ridge_fit, ridge_predict, rmse
from dataset import split_subjects
from meal_features import DEFAULT_BLOCKS as MEAL_BLOCKS
from meal_features import block_names as meal_block_names
from meal_features import build_all_features as build_all_meal_features
from meals import apply_macro_cleaning, fit_macro_cleaning, load_meals, scale_macros
from preprocess import apply_scaler, fit_scaler, load_segments, make_windows

SEEDS = tuple(range(42, 53))

# The full feature matrix every row slices from: all five activity blocks plus the K
# control, in ACTIVITY_BLOCKS order (13 columns).
SCREEN_BLOCKS = ("H", "R", "F", "A", "E", "K")

# Rows of the screen table: label -> activity blocks added on top of CGM+meal.
# "X" is not a block: it triggers the meal-x-activity interaction products at fit time.
SCREEN_ROWS = [
    ("clock only (K)",       ("K",)),
    ("heart rate (H)",       ("H",)),
    ("subject-rel HR (H,R)", ("H", "R")),
    ("HR + missing (H,F)",   ("H", "F")),
    ("+ timing (H,F,A)",     ("H", "F", "A")),
    ("+ energy (H,F,A,E)",   ("H", "F", "A", "E")),
    ("interactions (X)",     ("X",)),
    ("act + inter (HFAE,X)", ("H", "F", "A", "E", "X")),
    ("all + clock",          ("H", "R", "F", "A", "E", "K", "X")),
]

# Meal-x-activity products: (meal feature, activity feature). The physiological
# hypotheses -- "activity right after a meal blunts the spike" and friends -- live
# here, not in the main-effect rows.
INTERACTIONS = [
    ("decay", "act_decay"),
    ("decay", "hr_mean30_z"),
    ("carbs_x_decay", "hr_mean30_z"),
    ("impulse_60", "act_decay"),
    ("impulse_60", "cal30_z"),
    ("postprandial", "dt_active_norm"),
]

SCREEN_FEATURE_NAMES = [name for block in SCREEN_BLOCKS for name in ACTIVITY_BLOCKS[block]]


def activity_columns(blocks: tuple[str, ...]) -> list[int]:
    """Column indices into the full 13-column matrix for the given blocks ("X" ignored)."""
    wanted = {name for block in blocks if block != "X" for name in ACTIVITY_BLOCKS[block]}
    return [k for k, name in enumerate(SCREEN_FEATURE_NAMES) if name in wanted]


def interaction_matrix(meal_features: np.ndarray, activity_features: np.ndarray) -> np.ndarray:
    """One column per INTERACTIONS pair: meal feature x activity feature, row-wise."""
    meal_names = meal_block_names(MEAL_BLOCKS)
    columns = []
    for meal_name, activity_name in INTERACTIONS:
        columns.append(meal_features[:, meal_names.index(meal_name)]
                       * activity_features[:, SCREEN_FEATURE_NAMES.index(activity_name)])
    return np.column_stack(columns)


def build_ridge_data(segments, meal_matrices, activity_matrices, dt_actives,
                     mean: float, std: float, horizon_steps: int):
    """Stack per-segment windows into flat design matrices.

    Each window's feature row is the one at its LAST input reading (row
    window_start + HISTORY - 1) -- prediction-time state, the same alignment
    WindowDataset uses for statics."""
    all_cgm, all_meal, all_activity, all_targets, all_dt_active = [], [], [], [], []
    for segment, meal_matrix, activity_matrix, dt_active in zip(
            segments, meal_matrices, activity_matrices, dt_actives):
        windows, targets = make_windows(segment, HISTORY, horizon_steps)
        if len(windows) == 0:
            continue
        rows = np.arange(len(windows)) + HISTORY - 1
        all_cgm.append(apply_scaler(windows, mean, std))
        all_meal.append(meal_matrix[rows])
        all_activity.append(activity_matrix[rows])
        all_dt_active.append(dt_active[rows])
        all_targets.append(targets)
    return (np.concatenate(all_cgm), np.concatenate(all_meal),
            np.concatenate(all_activity), np.concatenate(all_targets),
            np.concatenate(all_dt_active))


def run_one_seed(segments, meals, aggregate_by_id, seed: int, horizon_steps: int) -> dict:
    """All screen rows plus the stratified comparison, for one seed's split."""
    subjects = sorted({segment.subject for segment in segments})
    train_subjects, _, test_subjects = split_subjects(subjects, seed=seed)

    segments_by_subject = {}
    for segment in segments:
        segments_by_subject.setdefault(segment.subject, []).append(segment)
    train_segments = [s for subj in train_subjects for s in segments_by_subject.get(subj, [])]
    test_segments = [s for subj in test_subjects for s in segments_by_subject.get(subj, [])]

    # The three fit objects, all train-only: glucose scaler, meal cleaning, activity stats.
    mean, std = fit_scaler(train_segments)
    cleaning = fit_macro_cleaning(meals, train_subjects)
    ready_meals = scale_macros(apply_macro_cleaning(meals, cleaning))
    stats = fit_activity_stats([aggregate_by_id[id(s)] for s in train_segments])

    def prepare(split_segments):
        meal_matrices = build_all_meal_features(split_segments, ready_meals, MEAL_BLOCKS)
        activity_matrices, dt_actives = [], []
        for segment in split_segments:
            features, names = build_activity_features(
                aggregate_by_id[id(segment)], stats, SCREEN_BLOCKS)
            activity_matrices.append(features)
            # dt_active in minutes, recovered from its normalized column -- the strata
            # below cut on it directly.
            dt_actives.append(features[:, names.index("dt_active_norm")] * DT_ACTIVE_CLIP_MIN)
        return build_ridge_data(split_segments, meal_matrices, activity_matrices,
                                dt_actives, mean, std, horizon_steps)

    train_cgm, train_meal, train_activity, train_y, _ = prepare(train_segments)
    test_cgm, test_meal, test_activity, test_y, test_dt_active = prepare(test_segments)

    last_input = test_cgm[:, -1] * std + mean
    dt_meal = test_meal[:, meal_block_names(MEAL_BLOCKS).index("dt_norm")] * 300.0

    results = {"_persistence": rmse(test_y - last_input), "_n": len(test_y)}
    predictions = {}

    def fit_score(label, x_train, x_test):
        weights = ridge_fit(x_train, train_y, lam=1.0)
        preds = ridge_predict(x_test, weights)
        predictions[label] = preds
        results[label] = rmse(preds - test_y)

    fit_score("CGM-only", train_cgm, test_cgm)
    fit_score("CGM+meal", np.hstack([train_cgm, train_meal]),
              np.hstack([test_cgm, test_meal]))
    columns = activity_columns(("H", "R", "F", "A", "E"))
    fit_score("activity, no meal", np.hstack([train_cgm, train_activity[:, columns]]),
              np.hstack([test_cgm, test_activity[:, columns]]))

    train_inter = interaction_matrix(train_meal, train_activity)
    test_inter = interaction_matrix(test_meal, test_activity)
    for label, blocks in SCREEN_ROWS:
        columns = activity_columns(blocks)
        train_parts = [train_cgm, train_meal, train_activity[:, columns]]
        test_parts = [test_cgm, test_meal, test_activity[:, columns]]
        if "X" in blocks:
            train_parts.append(train_inter)
            test_parts.append(test_inter)
        fit_score(label, np.hstack(train_parts), np.hstack(test_parts))

    # Stratified: CGM+meal vs meal + all activity blocks (no clock).
    base = predictions["CGM+meal"]
    full = predictions["+ energy (H,F,A,E)"]
    strata = {}
    for name, mask in [
        ("active <=30 min", test_dt_active <= 30),
        ("no activity >=2 h", test_dt_active >= 120),
        ("meal <=30 min", dt_meal <= 30),
        ("meal & active <=30", (dt_meal <= 30) & (test_dt_active <= 30)),
        ("fasting >=5 h", dt_meal >= 300),
    ]:
        if mask.sum() == 0:
            continue
        strata[name] = {"n": int(mask.sum()),
                        "base": rmse(base[mask] - test_y[mask]),
                        "full": rmse(full[mask] - test_y[mask])}
    results["_strata"] = strata
    return results


def run_screen(segments, meals, aggregate_by_id, seeds, horizon_steps: int) -> dict:
    """run_one_seed over a seed list -> {seed: results}, with progress printing."""
    all_results = {}
    for seed in seeds:
        print(f"  seed {seed}...", end="", flush=True)
        all_results[seed] = run_one_seed(segments, meals, aggregate_by_id, seed, horizon_steps)
        print(f" done (CGM+meal {all_results[seed]['CGM+meal']:.2f})")
    return all_results


def paired_deltas(all_results: dict, seeds, label: str) -> np.ndarray:
    """Per-seed paired delta of one screen row vs the CGM+meal baseline
    (positive = the row is better)."""
    baseline = np.array([all_results[seed]["CGM+meal"] for seed in seeds])
    row = np.array([all_results[seed][label] for seed in seeds])
    return baseline - row


FIGURE_HORIZONS = (30, 60, 120)
CLOCK_LABEL = "clock only (K)"


def save_screen_csvs(all_results: dict, seeds, horizon: int, results_dir: Path) -> None:
    """Persist the screen's per-seed numbers -- the B2 horizon-CSV convention: every
    printed mean, win count, and p-value stays recomputable from disk without a
    re-run. Two tidy files per horizon, because the table rows and the strata have
    different shapes."""
    table_path = results_dir / f"activity_screen_ph{horizon}.csv"
    row_labels = (["persistence", "CGM-only", "CGM+meal", "activity, no meal"]
                  + [label for label, _ in SCREEN_ROWS])
    with open(table_path, "w", newline="") as table_file:
        writer = csv.DictWriter(table_file,
                                fieldnames=["ph", "seed", "row", "rmse", "delta_vs_cgm_meal"])
        writer.writeheader()
        for seed in seeds:
            results = all_results[seed]
            baseline_rmse = results["CGM+meal"]
            for label in row_labels:
                if label == "persistence":
                    row_rmse = results["_persistence"]
                else:
                    row_rmse = results[label]
                writer.writerow({"ph": horizon, "seed": seed, "row": label,
                                 "rmse": row_rmse,
                                 "delta_vs_cgm_meal": baseline_rmse - row_rmse})
    print(f"  saved {table_path}")

    strata_path = results_dir / f"activity_screen_strata_ph{horizon}.csv"
    with open(strata_path, "w", newline="") as strata_file:
        writer = csv.DictWriter(strata_file,
                                fieldnames=["ph", "seed", "stratum", "n",
                                            "cgm_meal_rmse", "meal_activity_rmse", "delta"])
        writer.writeheader()
        for seed in seeds:
            for stratum_name, stratum in all_results[seed]["_strata"].items():
                writer.writerow({"ph": horizon, "seed": seed, "stratum": stratum_name,
                                 "n": stratum["n"],
                                 "cgm_meal_rmse": stratum["base"],
                                 "meal_activity_rmse": stratum["full"],
                                 "delta": stratum["base"] - stratum["full"]})
    print(f"  saved {strata_path}")


def figure_screen_horizons(segments, meals, aggregate_by_id, seeds,
                           results_dir: Path = Path("results")) -> None:
    """The control rises, activity does not: screen deltas across three horizons.

    Grey lines are the activity rows of SCREEN_ROWS (mean paired delta over the
    seeds; rows containing K are excluded because they carry the control). Orange
    is the clock-only control, drawn with its per-seed dots. This figure carries
    the method argument for the null: the same screen registers the small circadian
    effect, so an activity effect of that size would have been visible.

    Everything is computed live -- three horizons x all seeds, ~6 min -- because a
    figure rebuilt from hardcoded numbers stops being evidence the moment anything
    upstream changes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    activity_labels = [label for label, blocks in SCREEN_ROWS if "K" not in blocks]

    activity_means = {label: [] for label in activity_labels}
    clock_deltas_per_horizon = []
    for horizon in FIGURE_HORIZONS:
        print(f"\n+{horizon} min:")
        all_results = run_screen(segments, meals, aggregate_by_id, seeds, horizon // 5)
        save_screen_csvs(all_results, seeds, horizon, results_dir)
        for label in activity_labels:
            activity_means[label].append(paired_deltas(all_results, seeds, label).mean())
        clock_deltas_per_horizon.append(paired_deltas(all_results, seeds, CLOCK_LABEL))

    fig, ax = plt.subplots(figsize=(9, 6))
    for label in activity_labels:
        ax.plot(FIGURE_HORIZONS, activity_means[label], "-", color="grey",
                alpha=0.55, lw=1.2, zorder=2)

    clock_means = [float(deltas.mean()) for deltas in clock_deltas_per_horizon]
    for horizon, deltas in zip(FIGURE_HORIZONS, clock_deltas_per_horizon):
        jitter = np.linspace(-2.5, 2.5, len(deltas))
        ax.scatter(horizon + jitter, deltas, s=18, color="C1", alpha=0.45, zorder=3)
    ax.plot(FIGURE_HORIZONS, clock_means, "-o", color="C1", lw=2.2, zorder=4)

    # Annotate the control's endpoint with its win count -- the one number the
    # reader should leave with.
    final_deltas = clock_deltas_per_horizon[-1]
    wins = int((final_deltas > 0).sum())
    ax.annotate(f"+{clock_means[-1]:.3f} ({wins}/{len(seeds)} seeds)",
                (FIGURE_HORIZONS[-1], clock_means[-1]), xytext=(-8, 2),
                textcoords="offset points", ha="right", fontsize=9, color="C1")

    ax.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax.set_xticks(FIGURE_HORIZONS)
    ax.set_xticklabels([f"+{horizon}" for horizon in FIGURE_HORIZONS])
    ax.set_xlabel("prediction horizon (min)")
    ax.set_ylabel("paired Δ RMSE vs CGM+meal ridge (mg/dL, positive = better)")
    ax.set_title("Ridge screen by horizon: every activity feature set vs the clock control")
    activity_proxy = Line2D([0], [0], color="grey", alpha=0.55, lw=1.2)
    clock_proxy = Line2D([0], [0], color="C1", marker="o", lw=2.2)
    ax.legend([clock_proxy, activity_proxy],
              ["clock control (sin/cos hour)",
               f"activity feature sets ({len(activity_labels)} rows, mean over "
               f"{len(seeds)} seeds)"],
              loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = results_dir / "activity_screen_horizons.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ridge screen: activity features on top of CGM+meal, paired across seeds")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/CGMacros"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--horizon", type=int, default=30, choices=[30, 60, 120])
    parser.add_argument("--seeds", type=str, default=None,
                        help="comma-separated (default: 42..52)")
    parser.add_argument("--figure", action="store_true",
                        help="run all three horizons and save activity_screen_horizons.png "
                             "(ignores --horizon)")
    args = parser.parse_args()

    segments = load_segments(args.data_dir / "segments_split.npz")
    meals = load_meals(args.raw_dir)
    print("loading raw activity columns...", flush=True)
    frames = load_activity(args.raw_dir)
    print("aggregating onto the 5-min reading grid...", flush=True)
    # Aggregates are seed-independent (raw data + grid, no fitted constants), so build
    # them once and let every seed's split index into them by segment identity.
    aggregates = aggregate_segments(segments, frames)
    aggregate_by_id = {id(segment): aggregate
                      for segment, aggregate in zip(segments, aggregates)}

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(SEEDS)

    if args.figure:
        figure_screen_horizons(segments, meals, aggregate_by_id, seeds, args.results_dir)
        return

    all_results = run_screen(segments, meals, aggregate_by_id, seeds, args.horizon // 5)

    labels = ["CGM-only", "CGM+meal", "activity, no meal"] + [l for l, _ in SCREEN_ROWS]
    base_rmses = np.array([all_results[s]["CGM+meal"] for s in seeds])
    cgm_rmses = np.array([all_results[s]["CGM-only"] for s in seeds])

    print(f"\nRidge screen, +{args.horizon} min, {len(seeds)} seeds, "
          f"paired vs CGM+meal ridge ({base_rmses.mean():.2f} +/- {base_rmses.std(ddof=1):.2f})")
    print(f"  CGM-only ridge: {cgm_rmses.mean():.2f}  "
          f"(meal delta +{(cgm_rmses - base_rmses).mean():.3f})")
    print(f"\n  {'Row':<28} {'RMSE':>7} {'Paired D':>10} {'Wins':>7}")
    for label in labels[2:]:
        rmses = np.array([all_results[s][label] for s in seeds])
        deltas = base_rmses - rmses
        wins = int((deltas > 0).sum())
        print(f"  {label:<28} {rmses.mean():>7.2f} {deltas.mean():>+10.3f} "
              f"{wins:>3}/{len(seeds)}")

    print(f"\n  Stratified, CGM+meal vs meal+activity(H,F,A,E), mean across seeds:")
    print(f"  {'Stratum':<22} {'n':>7} {'meal':>8} {'meal+act':>9} {'delta':>8}")
    for key in all_results[seeds[0]]["_strata"]:
        rows = [all_results[s]["_strata"].get(key) for s in seeds]
        rows = [r for r in rows if r]
        mean_n = np.mean([r["n"] for r in rows])
        mean_base = np.mean([r["base"] for r in rows])
        mean_full = np.mean([r["full"] for r in rows])
        print(f"  {key:<22} {mean_n:>7.0f} {mean_base:>8.2f} {mean_full:>9.2f} "
              f"{mean_base - mean_full:>+8.3f}")

    print()
    save_screen_csvs(all_results, seeds, args.horizon, args.results_dir)


if __name__ == "__main__":
    main()
