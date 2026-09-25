"""Paired B3 activity ladder: every b3 arm against B2 static TMYIS.

Modelled on ablation_meals.py, one level up the baseline chain: B2's ladder paired
against B1, this one pairs against B2's winner (meal static TMYIS) -- the stored B2
checkpoints are LOADED, never retrained. Window-level pairing is what makes the deltas
mean anything: at the same seed every arm sees the byte-identical test set, so per-seed
differences are model differences, not split differences. That claim is asserted, not
assumed: y must be array-equal and persistence equal across every arm at every seed
(B1 included) -- the invariant that catches a split that silently stopped reproducing.

Arms are DISCOVERED from results/ folder names (ph30_b3{ENC}_{BLOCKS}_seed{N}), so the
gated extras (K-only control, feature-ablation rungs) join the table automatically the
moment their runs exist.

The stratified evaluation reuses the same checkpoint predictions: per-stratum paired
deltas on identical windows, cut by activity recency, meal-x-activity (the professor's
question as a table row), trend, and the fasting-AND-inactive negative control. Every
stratum is defined on information available at prediction time -- the B1 error_bins
lesson -- and the extraction is glucose-anchored against each checkpoint's own inputs.

Every run writes activity_ablation.csv (per-seed data) and activity_ablation.md
(the readable tables -- the metrics_table convention).

Usage:
    python src/ablation_activity.py                 # arms table + CSV + MD
    python src/ablation_activity.py --detail        # per-seed breakdown
    python src/ablation_activity.py --strata        # stratified table (B2 vs one b3 arm)
    python src/ablation_activity.py --dispersion    # under-dispersion ratio vs B2's 0.893
    python src/ablation_activity.py --figures       # activity_ablation.png +
                                                    # activity_strata.png (implies --strata)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from evaluate import (
    DEFAULT_SEEDS, RESULTS_DIR, checkpoint_params, find_checkpoint,
    load_test_predictions, meal_infix, parse_feature_tag, pooled_config, run_tag,
)
from preprocess import load_segments

B2_LABEL = "B2 static TMYIS"
B1_LABEL = "B1 (context)"

# Display order for encodings; unknown ones sort after, alphabetically.
ENCODING_ORDER = {"static": 0, "channels": 1, "hybrid": 2}

# Stratum cutoffs, all in minutes. The meal cutoffs mirror ablation_meals.py (dt_norm's
# scale IS the 300-min fasting cutoff); the activity cutoffs mirror the ridge screen's
# strata so the LSTM table reads against the same rows the screen reported.
ACTIVE_RECENT_MIN = 30
ACTIVE_STALE_MIN = 120
FASTING_CUTOFF_MIN = 300

# B2's recorded under-dispersion ratio (pred SD / actual SD), the reference for
# "did activity features make the predictions braver?" NOTE the recorded 0.893 (and
# the 7.1% rising skill) were computed against B2's BEST rung, static TMYI --
# ablation_meals auto-picks the top ladder rung -- while this table pairs against
# the headline arm TMYIS that b3 is built on, so small offsets (0.898, 7.3%) are
# a rung difference, not a discrepancy.
B2_DISPERSION_REFERENCE = 0.893

# Context lines for the ladder figure, both recorded B2 numbers on this exact axis:
# +0.841 is the meal effect (B2 static TMYIS vs B1, paired over the same 11 seeds) --
# what a real signal looked like -- and 0.59 is that comparison's paired seed-to-seed
# std, the noise band an arm must clear before its mean delta means anything.
MEAL_EFFECT_MGDL = 0.841
NOISE_BAND_MGDL = 0.59


def find_b3_runs(results_dir: Path, ph: int) -> dict[tuple[str, str], dict[int, str]]:
    """Discover every b3 arm: {(encoding, blocks): {seed: run tag}}.

    Discovery, not a hardcoded arm list, for the same reason aggregate_horizons
    scans folders: a K-only control or ablation rung trained later must appear in
    this table, not silently vanish."""
    pattern = re.compile(rf"^ph{ph}_b3(static|channels|hybrid)_([A-Z]+)_seed(\d+)$")
    arms: dict[tuple[str, str], dict[int, str]] = {}
    for folder in sorted(results_dir.iterdir()):
        match = pattern.match(folder.name)
        if folder.is_dir() and match:
            key = (match.group(1), match.group(2))
            arms.setdefault(key, {})[int(match.group(3))] = folder.name
    return arms


def evaluate_tag(tag: str, seed: int, ph: int, segments, results_dir: Path):
    """Predictions + summary numbers for one run, on its own rebuilt test split.

    The config comes from the tag (parse_feature_tag), so a b3 checkpoint is always
    scored with meal static TMYIS underneath and its own activity settings -- the
    fourth instance of that discipline."""
    ckpt = find_checkpoint(tag, results_dir)
    if ckpt is None:
        return None

    meal_encoding, meal_blocks, activity_encoding, activity_blocks = parse_feature_tag(tag)
    config = pooled_config(seed, ph=ph,
                           meal_blocks=meal_blocks, meal_encoding=meal_encoding or "static",
                           activity_blocks=activity_blocks,
                           activity_encoding=activity_encoding or "static")
    try:
        y, preds, last_input, subjects = load_test_predictions(ckpt, segments, config)
        params = checkpoint_params(ckpt)
    except FileNotFoundError:
        # A run that is TRAINING right now: save_top_k=1 replaces the checkpoint file
        # between our glob and our open. Announce and move on -- the arm's seed count
        # in the table shows it is incomplete.
        print(f"  ({tag}: checkpoint vanished mid-read -- still training?)")
        return None

    persistence = float(np.sqrt(np.mean((last_input - y) ** 2)))
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    return {
        "seed": seed, "rmse": rmse, "persistence": persistence,
        "skill": (persistence - rmse) / persistence,
        "params": params,
        "y": y, "preds": preds, "last_input": last_input,
    }


def collect_arms(segments, results_dir: Path, ph: int, seeds=DEFAULT_SEEDS):
    """Evaluate B2 (baseline), B1 (context), and every discovered b3 arm.

    Returns {label: {seed: result}} with the baseline first."""
    results: dict[str, dict[int, dict]] = {}

    specs = [(B2_LABEL, meal_infix("static", ("T", "M", "Y", "I", "S"))),
             (B1_LABEL, "")]
    for label, infix in specs:
        results[label] = {}
        for seed in seeds:
            result = evaluate_tag(run_tag(seed, ph, infix), seed, ph, segments, results_dir)
            if result is None:
                print(f"  [skip] {label}: no run at seed {seed}")
                continue
            results[label][seed] = result
        print(f"  {label:>22}: {len(results[label])} seeds")

    arms = find_b3_runs(results_dir, ph)
    arm_keys = sorted(arms, key=lambda key: (ENCODING_ORDER.get(key[0], 9), key[1]))
    for encoding, blocks in arm_keys:
        label = f"b3 {encoding} {blocks}"
        results[label] = {}
        for seed, tag in sorted(arms[(encoding, blocks)].items()):
            result = evaluate_tag(tag, seed, ph, segments, results_dir)
            if result is None:
                print(f"  ({tag}: no checkpoint)")
                continue
            results[label][seed] = result
        print(f"  {label:>22}: {len(results[label])} seeds")

    return results


def check_pairing(results, baseline_label: str = B2_LABEL) -> None:
    """Every arm must see the byte-identical test set the baseline does, per seed.

    y array-equal and persistence equal (not just close) are both properties of the
    split alone; a difference means a split stopped reproducing and every paired
    delta below would be measuring splits, not models."""
    base = results.get(baseline_label, {})
    for label, seeds_dict in results.items():
        if label == baseline_label:
            continue
        for seed, result in seeds_dict.items():
            if seed not in base:
                continue
            baseline_result = base[seed]
            assert np.array_equal(result["y"], baseline_result["y"]), \
                f"{label} seed {seed}: y_actual differs from {baseline_label} -- PAIRING BROKEN"
            assert abs(result["persistence"] - baseline_result["persistence"]) < 1e-6, \
                (f"{label} seed {seed}: persistence differs "
                 f"({result['persistence']:.6f} vs {baseline_result['persistence']:.6f})")


def summarize_arms(results, baseline_label: str = B2_LABEL):
    """One summary row per arm (means, paired delta, wins, Wilcoxon p) plus the
    per-seed frame. Computed ONCE and consumed by the console table, the CSV, and
    the markdown file, so the three can never disagree."""
    base = results.get(baseline_label, {})
    summaries = []
    per_seed_rows = []
    for label, seeds_dict in results.items():
        shared = sorted(set(seeds_dict) & set(base))
        if not shared:
            continue

        rmses = np.array([seeds_dict[s]["rmse"] for s in shared])
        persists = np.array([seeds_dict[s]["persistence"] for s in shared])
        skills = np.array([seeds_dict[s]["skill"] for s in shared])
        summary = {
            "arm": label,
            "n": len(shared),
            "rmse_mean": float(rmses.mean()),
            "rmse_std": float(rmses.std(ddof=1)) if len(shared) > 1 else None,
            "persist_mean": float(persists.mean()),
            "skill_mean": float(skills.mean()),
            "delta_mean": None,        # stays None for the baseline row
            "wins": None,
            "pvalue": None,
            "params": seeds_dict[shared[0]]["params"],
        }

        if label != baseline_label:
            base_rmses = np.array([base[s]["rmse"] for s in shared])
            deltas = base_rmses - rmses        # positive = arm better than B2
            summary["delta_mean"] = float(deltas.mean())
            summary["wins"] = int((deltas > 0).sum())
            if len(shared) >= 3 and not np.allclose(deltas, 0):
                summary["pvalue"] = float(wilcoxon(base_rmses, rmses).pvalue)
            for s in shared:
                per_seed_rows.append({"arm": label, "seed": s,
                                      "rmse": seeds_dict[s]["rmse"],
                                      "persistence": seeds_dict[s]["persistence"],
                                      "skill": seeds_dict[s]["skill"],
                                      "b2_rmse": base[s]["rmse"],
                                      "delta_vs_b2": base[s]["rmse"] - seeds_dict[s]["rmse"],
                                      "params": summary["params"]})

        summaries.append(summary)

    return summaries, pd.DataFrame(per_seed_rows)


def paired_report(summaries, baseline_label: str = B2_LABEL) -> None:
    """Print the arms table from the shared summaries."""
    print(f"\n{'=' * 88}")
    print(f"Activity ladder (paired vs {baseline_label})")
    print(f"{'=' * 88}")
    print(f"\n  {'Arm':<22} {'n':>3} {'RMSE':>16} {'Persist':>8} {'Skill':>7} "
          f"{'vs B2':>10} {'Wins':>6} {'p':>8} {'Params':>8}")
    print(f"  {'-' * 22} {'-' * 3} {'-' * 16} {'-' * 8} {'-' * 7} "
          f"{'-' * 10} {'-' * 6} {'-' * 8} {'-' * 8}")

    for summary in summaries:
        rmse_cell = f"{summary['rmse_mean']:.2f}"
        if summary["rmse_std"] is not None:
            rmse_cell += f" +/- {summary['rmse_std']:.2f}"

        if summary["delta_mean"] is None:
            delta_cell, wins_cell, pvalue_cell = "--", "--", "--"
        else:
            delta_cell = f"{summary['delta_mean']:+.3f}"
            wins_cell = f"{summary['wins']}/{summary['n']}"
            pvalue_cell = (f"{summary['pvalue']:.4f}"
                           if summary["pvalue"] is not None else "--")

        print(f"  {summary['arm']:<22} {summary['n']:>3} {rmse_cell:>16} "
              f"{summary['persist_mean']:>8.2f} {summary['skill_mean']:>6.1%} "
              f"{delta_cell:>10} {wins_cell:>6} {pvalue_cell:>8} {summary['params']:>8,}")


def detail_report(results, baseline_label: str = B2_LABEL) -> None:
    """Per-seed deltas for every arm -- the show-every-point rule, in text."""
    base = results.get(baseline_label, {})
    for label, seeds_dict in results.items():
        if label == baseline_label:
            continue
        shared = sorted(set(seeds_dict) & set(base))
        if not shared:
            continue
        print(f"\n  {label}:")
        for s in shared:
            delta = base[s]["rmse"] - seeds_dict[s]["rmse"]
            print(f"    seed {s}:  rmse {seeds_dict[s]['rmse']:6.2f}   "
                  f"b2 {base[s]['rmse']:6.2f}   delta {delta:+.3f}")


def load_window_state(seed: int, segments, meals, aggregate_by_id, ph: int):
    """Prediction-time state for one seed's test windows: (dt_meal, dt_active, trend,
    last_reading), each one value per window in dataset order.

    Everything is read at the window's LAST input reading (row window_start +
    HISTORY - 1) -- the B1 error_bins lesson: never stratify on anything the model
    could not know at prediction time. The meal cleaning and ActivityStats are
    refit from this seed's training subjects, exactly as the checkpoints saw them.

    last_reading exists purely so the caller can assert alignment against the
    checkpoint's own last_input -- a stratification built one row off would still
    produce a plausible table."""
    from baselines import HISTORY
    from activity import fit_activity_stats
    from activity_features import DT_ACTIVE_CLIP_MIN
    from activity_features import build_features as build_activity_features
    from dataset import split_subjects
    from meal_features import block_names as meal_block_names
    from meal_features import build_all_features as build_all_meal_features
    from meals import apply_macro_cleaning, fit_macro_cleaning, scale_macros
    from preprocess import make_windows

    subjects = sorted({segment.subject for segment in segments})
    train_subjects, _, test_subjects = split_subjects(subjects, seed=seed)

    segments_by_subject: dict[str, list] = {}
    for segment in segments:
        segments_by_subject.setdefault(segment.subject, []).append(segment)
    train_segments = [s for subj in train_subjects for s in segments_by_subject.get(subj, [])]
    test_segments = [s for subj in test_subjects for s in segments_by_subject.get(subj, [])]

    cleaning = fit_macro_cleaning(meals, train_subjects)
    ready_meals = scale_macros(apply_macro_cleaning(meals, cleaning))
    stats = fit_activity_stats([aggregate_by_id[id(s)] for s in train_segments])

    # Only the timing blocks are needed: meal "T" carries dt_norm, activity "A"
    # carries dt_active_norm. Both are normalized by their clip, so minutes come
    # back by multiplying the clip in again.
    meal_matrices = build_all_meal_features(test_segments, ready_meals, ("T",))
    dt_meal_column = meal_block_names(("T",)).index("dt_norm")

    all_dt_meal, all_dt_active, all_trend, all_last = [], [], [], []
    for segment, meal_matrix in zip(test_segments, meal_matrices):
        windows, _ = make_windows(segment, HISTORY, ph // 5)      # raw mg/dL
        if len(windows) == 0:
            continue
        rows = np.arange(len(windows)) + HISTORY - 1

        activity_matrix, activity_names = build_activity_features(
            aggregate_by_id[id(segment)], stats, ("A",))
        dt_active_column = activity_names.index("dt_active_norm")

        all_dt_meal.append(meal_matrix[rows, dt_meal_column] * FASTING_CUTOFF_MIN)
        all_dt_active.append(activity_matrix[rows, dt_active_column] * DT_ACTIVE_CLIP_MIN)
        # mg/dL per minute over the last 15 min -- the trend-strata definition
        # shared with ablation_meals.py.
        all_trend.append((windows[:, -1] - windows[:, -4]) / 15.0)
        all_last.append(windows[:, -1])

    return (np.concatenate(all_dt_meal), np.concatenate(all_dt_active),
            np.concatenate(all_trend), np.concatenate(all_last))


def stratum_masks(dt_meal, dt_active, trend) -> list[tuple[str, np.ndarray]]:
    """Named boolean masks over one seed's test windows, in table order.

    The activity-recency rows partition on dt_active (which clips at 300, so
    "no activity >=2 h" includes never-elevated); "meal & active <=30" is the
    professor's question as a row; "fasting+inactive" is the negative control --
    those windows contain no meal or activity information for B3 to use, so any
    delta there is a bug or a confound, full stop."""
    recently_active = dt_active <= ACTIVE_RECENT_MIN
    recent_meal = dt_meal <= 30
    return [
        ("active <=30 min",     recently_active),
        ("active 30-120 min",   (dt_active > ACTIVE_RECENT_MIN) & (dt_active < ACTIVE_STALE_MIN)),
        ("no activity >=2 h",   dt_active >= ACTIVE_STALE_MIN),
        ("meal <=30 min",       recent_meal),
        ("meal & active <=30",  recent_meal & recently_active),
        ("meal & NOT active",   recent_meal & ~recently_active),
        ("rising > +1",         trend > 1.0),
        ("flat -1..+1",         (trend >= -1.0) & (trend <= 1.0)),
        ("falling < -1",        trend < -1.0),
        ("fasting+inactive",    (dt_meal >= FASTING_CUTOFF_MIN) & (dt_active >= ACTIVE_STALE_MIN)),
    ]


def stratified_report(results, segments, meals, aggregate_by_id, ph: int,
                      arm_label: str):
    """Per-stratum table: B2 vs one b3 arm on identical windows, plus the negative
    control and the joint-stratum power note the Part 8 writeup needs."""
    base = results.get(B2_LABEL, {})
    arm = results.get(arm_label, {})
    shared = sorted(set(arm) & set(base))
    if not shared:
        print(f"No shared seeds between {B2_LABEL} and {arm_label!r} -- "
              f"known arms: {[label for label in results]}")
        return None

    print(f"\n{'=' * 88}")
    print(f"Stratified evaluation -- {arm_label} vs {B2_LABEL} ({len(shared)} seeds)")
    print(f"{'=' * 88}")

    # Filled lazily from the first seed's masks; dict insertion order IS table order.
    strata: dict[str, dict[str, list]] = {}

    for seed in shared:
        base_result, arm_result = base[seed], arm[seed]
        y = base_result["y"]
        dt_meal, dt_active, trend, last_reading = load_window_state(
            seed, segments, meals, aggregate_by_id, ph)

        # Alignment must be proven, not assumed: same window count, and the raw
        # last reading must equal the checkpoint's own un-scaled last input
        # (float32 round trip allows ~1e-3).
        assert len(dt_meal) == len(y), \
            f"seed {seed}: {len(dt_meal)} state rows for {len(y)} windows"
        assert np.allclose(last_reading, base_result["last_input"], atol=1e-3), \
            f"seed {seed}: window state is not glucose-anchored -- rows are misaligned"

        for name, mask in stratum_masks(dt_meal, dt_active, trend):
            count = int(mask.sum())
            if count == 0:
                continue
            stratum = strata.setdefault(
                name, {"n": [], "persist": [], "b2": [], "b3": [], "delta": []})
            persist_rmse = float(np.sqrt(np.mean(
                (base_result["last_input"][mask] - y[mask]) ** 2)))
            b2_rmse = float(np.sqrt(np.mean((base_result["preds"][mask] - y[mask]) ** 2)))
            b3_rmse = float(np.sqrt(np.mean((arm_result["preds"][mask] - y[mask]) ** 2)))
            stratum["n"].append(count)
            stratum["persist"].append(persist_rmse)
            stratum["b2"].append(b2_rmse)
            stratum["b3"].append(b3_rmse)
            stratum["delta"].append(b2_rmse - b3_rmse)          # positive = B3 better

    print(f"\n  {'Stratum':<20} {'n/seed':>7} {'Persist':>8} {'B2':>8} {'B3':>8} "
          f"{'B2 skill':>9} {'B3 skill':>9} {'B3-B2':>9}")
    print(f"  {'-' * 20} {'-' * 7} {'-' * 8} {'-' * 8} {'-' * 8} "
          f"{'-' * 9} {'-' * 9} {'-' * 9}")
    for name, stratum in strata.items():
        mean_persist = np.mean(stratum["persist"])
        mean_b2 = np.mean(stratum["b2"])
        mean_b3 = np.mean(stratum["b3"])
        mean_delta = np.mean(stratum["delta"])
        delta_str = f"+{mean_delta:.3f}" if mean_delta >= 0 else f"{mean_delta:.3f}"
        print(f"  {name:<20} {np.mean(stratum['n']):>7.0f} {mean_persist:>8.2f} "
              f"{mean_b2:>8.2f} {mean_b3:>8.2f} "
              f"{(mean_persist - mean_b2) / mean_persist:>+8.1%} "
              f"{(mean_persist - mean_b3) / mean_persist:>+8.1%} {delta_str:>9}")

    # The professor's question, spelled out: post-meal windows, active vs not.
    active_row = strata.get("meal & active <=30")
    inactive_row = strata.get("meal & NOT active")
    if active_row and inactive_row:
        print(f"\n  THE PROFESSOR'S QUESTION (meal <=30 min, split by activity <=30 min):")
        print(f"    active:   n/seed {np.mean(active_row['n']):>5.0f}   "
              f"B3-B2 {np.mean(active_row['delta']):+.3f} mg/dL")
        print(f"    inactive: n/seed {np.mean(inactive_row['n']):>5.0f}   "
              f"B3-B2 {np.mean(inactive_row['delta']):+.3f} mg/dL")
        # Small stratum -> state what the 11-seed comparison could have seen, not
        # just that it saw nothing (Part 8 must claim "no effect >= X", not "no effect").
        deltas = np.array(active_row["delta"])
        standard_error = deltas.std(ddof=1) / np.sqrt(len(deltas))
        print(f"    joint-stratum power: per-seed delta {deltas.mean():+.3f} "
              f"+/- {deltas.std(ddof=1):.3f}; detectable effect ~ "
              f"{2 * standard_error:.2f} mg/dL (2 SE over {len(deltas)} seeds)")

    control = strata.get("fasting+inactive")
    if control:
        control_delta = float(np.mean(control["delta"]))
        print(f"\n  NEGATIVE CONTROL (fasting >=5 h AND no activity >=2 h):")
        print(f"    B3-B2 delta: {control_delta:+.3f} mg/dL "
              f"(n/seed {np.mean(control['n']):.0f})")
        if abs(control_delta) < 0.2:
            print(f"    -> Near zero, as it must be: these windows hold no meal or "
                  f"activity information.")
        else:
            print(f"    -> WARNING: {abs(control_delta):.2f} mg/dL on windows with no "
                  f"usable information -- investigate for leakage or confound.")

    return strata


def dispersion_report(results, arm_label: str) -> None:
    """Under-dispersion ratio (pred SD / actual SD), B2 vs the b3 arm, per seed."""
    base = results.get(B2_LABEL, {})
    arm = results.get(arm_label, {})
    shared = sorted(set(arm) & set(base))
    if not shared:
        print(f"No shared seeds between {B2_LABEL} and {arm_label!r}.")
        return

    print(f"\n{'=' * 88}")
    print(f"Under-dispersion -- {arm_label} vs {B2_LABEL}")
    print(f"{'=' * 88}")
    print(f"\n  {'Seed':>6} {'B2 pred/act SD':>16} {'B3 pred/act SD':>16}")

    b2_ratios, b3_ratios = [], []
    for seed in shared:
        y = base[seed]["y"]
        b2_ratio = float(np.std(base[seed]["preds"]) / np.std(y))
        b3_ratio = float(np.std(arm[seed]["preds"]) / np.std(y))
        b2_ratios.append(b2_ratio)
        b3_ratios.append(b3_ratio)
        print(f"  {seed:>6} {b2_ratio:>16.3f} {b3_ratio:>16.3f}")

    print(f"\n  B2 mean ratio: {np.mean(b2_ratios):.3f} "
          f"(recorded reference {B2_DISPERSION_REFERENCE} was for B2's best rung, "
          f"static TMYI)")
    print(f"  B3 mean ratio: {np.mean(b3_ratios):.3f}")
    print(f"  Change: {np.mean(b3_ratios) - np.mean(b2_ratios):+.3f} "
          f"-> predictions {'got braver' if np.mean(b3_ratios) > np.mean(b2_ratios) else 'did not get braver'}")


def figure_activity_strata(strata, arm_label: str,
                           out_path: Path = Path("results/activity_strata.png")) -> None:
    """Paired delta by stratum, one dot per seed -- the show-every-point rule.

    Modelled on meal_strata.png's role but drawn as deltas: for a null, bars near
    zero with seed dots straddling it ARE the result, and hiding the spread behind
    an error bar would oversell the precision."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [name for name in strata if strata[name]["n"]]
    positions = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(12, 6))
    for position, name in zip(positions, names):
        deltas = np.array(strata[name]["delta"])
        jitter = np.linspace(-0.15, 0.15, len(deltas))
        ax.scatter(position + jitter, deltas, s=30, alpha=0.7, zorder=3, color="C0")
        ax.plot([position - 0.28, position + 0.28], [deltas.mean()] * 2,
                "k-", lw=2, zorder=4)

    # n labels go in AFTER all data: ylim is only final now, so every label sits at
    # the same height just above the bottom axis.
    for position, name in zip(positions, names):
        ax.annotate(f"n={np.mean(strata[name]['n']):.0f}",
                    (position, ax.get_ylim()[0]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=7, color="grey")

    ax.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("paired Δ RMSE vs B2 (mg/dL, positive = B3 better)")
    ax.set_title(f"Where activity would help, if it helped -- {arm_label} vs B2, "
                 f"one dot per seed (+30 min)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  saved {out_path}")


def figure_activity_ablation(results,
                             out_path: Path = Path("results/activity_ablation.png")) -> None:
    """The headline null figure: one dot per seed per arm, paired delta vs B2.

    The two context lines ARE the argument. A real effect on this exact axis --
    the meal effect, measured the same paired way over the same seeds -- sat at
    +0.841; the noise band is +/-0.59. Every activity arm's dots straddle zero or
    sit below it, an order of magnitude short of where signal lives."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = results.get(B2_LABEL, {})
    arm_labels = [label for label in results if label not in (B2_LABEL, B1_LABEL)]
    if not base or not arm_labels:
        print("  activity_ablation figure: need the B2 baseline and at least one b3 arm")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for position, label in enumerate(arm_labels):
        seeds_dict = results[label]
        shared = sorted(set(seeds_dict) & set(base))
        if not shared:
            continue
        deltas = [base[s]["rmse"] - seeds_dict[s]["rmse"] for s in shared]
        jitter = np.linspace(-0.12, 0.12, len(deltas))
        ax.scatter(position + jitter, deltas, s=32, alpha=0.75, zorder=3, color="C0")
        ax.plot([position - 0.22, position + 0.22], [np.mean(deltas)] * 2,
                "k-", lw=2, zorder=4)

    ax.axhspan(-NOISE_BAND_MGDL, NOISE_BAND_MGDL, color="grey", alpha=0.10,
               label=f"±{NOISE_BAND_MGDL} noise band (B2-vs-B1 paired std)")
    ax.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(MEAL_EFFECT_MGDL, color="C2", ls=":", lw=1.6,
               label=f"the meal effect (+{MEAL_EFFECT_MGDL}, B2 vs B1) — a real signal")
    ax.set_xticks(range(len(arm_labels)))
    ax.set_xticklabels(arm_labels)
    ax.set_xlim(-0.6, len(arm_labels) - 0.4)
    ax.set_ylabel("paired Δ RMSE vs B2 (mg/dL, positive = better)")
    ax.set_title("Activity ladder: each dot is one seed (+30 min)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  saved {out_path}")


def write_markdown(summaries, strata, arm_label: str,
                   out_path: Path = Path("results/activity_ablation.md")) -> None:
    """The readable version of the tables this script prints -- the metrics_table
    convention: CSV for data, MD for people. Rebuilt on every run from the same
    summaries the console table uses; the strata section appears when the strata
    were computed this run (`--strata`), with an explicit note when they were not."""
    lines = ["# B3 activity ladder — paired vs B2 static TMYIS", ""]
    lines.append("Generated by `python src/ablation_activity.py --strata`. Per-seed data: "
                 "`activity_ablation.csv`. Positive Δ = better than B2; p is a paired "
                 "Wilcoxon over seeds.")
    lines.append("")
    lines.append("| Arm | n | RMSE (mg/dL) | Persistence | Skill | Δ vs B2 | Wins | p | Params |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for summary in summaries:
        rmse_cell = f"{summary['rmse_mean']:.2f}"
        if summary["rmse_std"] is not None:
            rmse_cell += f" ± {summary['rmse_std']:.2f}"
        if summary["delta_mean"] is None:
            delta_cell, wins_cell, pvalue_cell = "—", "—", "—"
        else:
            delta_cell = f"{summary['delta_mean']:+.3f}"
            wins_cell = f"{summary['wins']}/{summary['n']}"
            pvalue_cell = (f"{summary['pvalue']:.4f}"
                           if summary["pvalue"] is not None else "—")
        lines.append(f"| {summary['arm']} | {summary['n']} | {rmse_cell} | "
                     f"{summary['persist_mean']:.2f} | {summary['skill_mean']:.1%} | "
                     f"{delta_cell} | {wins_cell} | {pvalue_cell} | {summary['params']:,} |")

    if strata is None:
        lines.append("")
        lines.append("*Stratified section omitted — rerun with `--strata` to include it.*")
    else:
        lines.append("")
        lines.append(f"## Stratified: {arm_label} vs B2, identical windows")
        lines.append("")
        lines.append("| Stratum | n/seed | Persistence | B2 | B3 | B2 skill | B3 skill | B3−B2 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for name, stratum in strata.items():
            mean_persist = float(np.mean(stratum["persist"]))
            mean_b2 = float(np.mean(stratum["b2"]))
            mean_b3 = float(np.mean(stratum["b3"]))
            lines.append(f"| {name} | {np.mean(stratum['n']):.0f} | {mean_persist:.2f} | "
                         f"{mean_b2:.2f} | {mean_b3:.2f} | "
                         f"{(mean_persist - mean_b2) / mean_persist:+.1%} | "
                         f"{(mean_persist - mean_b3) / mean_persist:+.1%} | "
                         f"{np.mean(stratum['delta']):+.3f} |")

        joint = strata.get("meal & active <=30")
        if joint:
            deltas = np.array(joint["delta"])
            standard_error = deltas.std(ddof=1) / np.sqrt(len(deltas))
            lines.append("")
            lines.append(f"- **The professor's question** (meal ≤30 min AND active ≤30 min, "
                         f"n = {np.mean(joint['n']):.0f}/seed): Δ = {deltas.mean():+.3f} ± "
                         f"{deltas.std(ddof=1):.3f} mg/dL — no detectable effect ≥ "
                         f"{2 * standard_error:.2f} mg/dL (2 SE over {len(deltas)} seeds).")
        control = strata.get("fasting+inactive")
        if control:
            lines.append(f"- **Negative control** (fasting ≥5 h AND no activity ≥2 h, "
                         f"n = {np.mean(control['n']):.0f}/seed): Δ = "
                         f"{np.mean(control['delta']):+.3f} mg/dL — these windows hold no "
                         f"meal or activity information, so ≈0 is required, and holds.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired B3 activity ladder vs B2 static TMYIS")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--ph", type=int, default=30)
    parser.add_argument("--detail", action="store_true", help="per-seed breakdown")
    parser.add_argument("--strata", action="store_true",
                        help="stratified table: activity recency, meal x activity, "
                             "trend, negative control")
    parser.add_argument("--dispersion", action="store_true",
                        help="under-dispersion ratio comparison vs B2")
    parser.add_argument("--figures", action="store_true",
                        help="activity_ablation.png + activity_strata.png "
                             "(computes the strata if needed)")
    parser.add_argument("--arm", default="b3 static HRFAE",
                        help="which arm the strata/dispersion compare against B2 "
                             "(default: the static arm; channels is a known harm)")
    parser.add_argument("--out", type=Path, default=None,
                        help="CSV path (default: results/activity_ablation.csv)")
    args = parser.parse_args()

    segments = load_segments(args.data_dir / "segments_split.npz")

    print(f"Scanning {args.results_dir}/ for b3 runs (ph{args.ph})...")
    results = collect_arms(segments, args.results_dir, args.ph)

    check_pairing(results)
    summaries, frame = summarize_arms(results)
    if not summaries:
        print(f"No arms share a seed with {B2_LABEL} -- nothing to report.")
        return
    paired_report(summaries)

    if args.detail:
        detail_report(results)

    strata = None
    if args.strata or args.figures:
        from activity import aggregate_segments, load_activity
        from meals import load_meals

        raw_dir = Path("dataset/CGMacros")
        meals = load_meals(raw_dir)
        frames = load_activity(raw_dir)
        aggregates = aggregate_segments(segments, frames)
        aggregate_by_id = {id(segment): aggregate
                          for segment, aggregate in zip(segments, aggregates)}

        strata = stratified_report(results, segments, meals, aggregate_by_id,
                                   args.ph, args.arm)
        if args.figures:
            figure_activity_ablation(results)
            if strata is not None:
                figure_activity_strata(strata, args.arm)

    if args.dispersion:
        dispersion_report(results, args.arm)

    print()
    write_markdown(summaries, strata, args.arm,
                   out_path=args.results_dir / "activity_ablation.md")

    if len(frame):
        out = args.out or args.results_dir / "activity_ablation.csv"
        frame.to_csv(out, index=False)
        print(f"\nSaved {out} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
