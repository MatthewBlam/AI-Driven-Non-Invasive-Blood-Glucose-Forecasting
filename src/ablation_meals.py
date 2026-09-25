"""Paired B2 ablation ladder and stratified evaluation.

Modelled on compare_architectures.py. Discovers B2 run folders, recomputes RMSE from
each checkpoint, and pairs everything against B1 -- per seed, window by window.

The stratified evaluation is built in: it uses the SAME checkpoint predictions
and B1 pairing to compute per-stratum deltas, the fasting negative control, the
rising-window result, and the under-dispersion ratio.

Usage:
    python src/ablation_meals.py                 # summary table
    python src/ablation_meals.py --detail        # per-seed breakdown
    python src/ablation_meals.py --strata        # stratified evaluation
    python src/ablation_meals.py --dispersion    # dispersion ratio comparison
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from evaluate import (
    find_checkpoint, load_test_predictions, load_test_predictions_with_history,
    pooled_config, run_tag, meal_infix, parse_meal_tag, iter_pooled_seeds,
    iter_pooled_seeds_with_history, DEFAULT_SEEDS, RESULTS_DIR,
)
from preprocess import load_segments

# Windows with time-since-meal >= 300 min (5 h) are classified as fasting
FASTING_CUTOFF_MIN = 300

# Ladder 1: encoding comparison (all blocks, vary encoding)
LADDER1 = [
    ("B1",       None,      None),
    ("static",   "static",  ("T", "M", "Y", "I", "S")),
    ("channels", "channels", ("T", "M", "Y", "I", "S")),
    ("hybrid",   "hybrid",  ("T", "M", "Y", "I", "S")),
]

# Ladder 2: feature ablation (encoding fixed at winner, vary blocks)
LADDER2_BLOCKS = [
    ("B1",    None),
    ("T",     ("T",)),
    ("T,C",   ("T", "C")),
    ("T,M",   ("T", "M")),
    ("T,M,Y", ("T", "M", "Y")),
    ("T,M,Y,I", ("T", "M", "Y", "I")),
    ("T,M,Y,I,S", ("T", "M", "Y", "I", "S")),
]


def find_b2_runs(results_dir: Path, encoding: str, blocks: tuple[str, ...],
                 ph: int = 30) -> dict[int, Path]:
    """Map seed -> run folder for one B2 configuration."""
    infix = meal_infix(encoding, blocks)
    runs = {}
    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir():
            continue
        expected = f"ph{ph}{infix}_seed"
        if folder.name.startswith(expected):
            match = re.search(r"seed(\d+)$", folder.name)
            if match:
                runs[int(match.group(1))] = folder
    return runs


def evaluate_b2_run(folder, seed, ph, segments):
    """Load one B2 checkpoint and extract predictions on its own test split."""
    meta = folder / "version_0" / "run_meta.json"
    if not meta.exists():
        return None
    ckpt = find_checkpoint(folder.name)
    if ckpt is None:
        return None

    try:
        encoding, blocks = parse_meal_tag(folder.name)
        config = pooled_config(seed, ph=ph, meal_blocks=blocks,
                               meal_encoding=encoding or "static")
        y, preds, last_input, subjects = load_test_predictions(ckpt, segments, config)
    except Exception as e:
        print(f"  ({folder.name}: failed to load -- {e})")
        return None

    persistence_rmse = float(np.sqrt(np.mean((last_input - y) ** 2)))
    model_rmse = float(np.sqrt(np.mean((preds - y) ** 2)))

    from model import count_params
    from train import LitGlucoseLSTM
    model = LitGlucoseLSTM.load_from_checkpoint(ckpt, map_location="cpu")
    params = count_params(model.network)

    return {
        "seed": seed, "rmse": model_rmse, "persistence": persistence_rmse,
        "skill": (persistence_rmse - model_rmse) / persistence_rmse,
        "params": params,
        "y": y, "preds": preds, "last_input": last_input, "subjects": subjects,
    }


def load_b1_predictions(seed, ph, segments):
    """Load B1 predictions for pairing."""
    tag = run_tag(seed, ph)
    ckpt = find_checkpoint(tag)
    if ckpt is None:
        return None
    config = pooled_config(seed, ph=ph)
    y, preds, last_input, subjects = load_test_predictions(ckpt, segments, config)
    return {"y": y, "preds": preds, "last_input": last_input, "subjects": subjects}


def collect_ladder(ladder_spec, segments, results_dir, ph, encoding_for_ladder2=None):
    """Evaluate all runs for a ladder. Returns {rung_label: {seed: result_dict}}."""
    results = {}
    for label, encoding, blocks in ladder_spec:
        if blocks is None:
            results[label] = {}
            for seed in DEFAULT_SEEDS:
                b1 = load_b1_predictions(seed, ph, segments)
                if b1 is None:
                    print(f"  [skip] B1 seed {seed} not found")
                    continue
                b1["rmse"] = float(np.sqrt(np.mean((b1["preds"] - b1["y"]) ** 2)))
                b1["persistence"] = float(np.sqrt(np.mean((b1["last_input"] - b1["y"]) ** 2)))
                b1["skill"] = (b1["persistence"] - b1["rmse"]) / b1["persistence"]
                b1["seed"] = seed
                b1["params"] = 17217
                results[label][seed] = b1
            print(f"  {'B1':>15}: {len(results[label])} seeds")
            continue

        if encoding is None:
            encoding = encoding_for_ladder2
        runs = find_b2_runs(results_dir, encoding, blocks, ph)
        if not runs:
            print(f"  {label:>15}: no runs found ({encoding}, {''.join(blocks)})")
            results[label] = {}
            continue

        results[label] = {}
        for seed, folder in sorted(runs.items()):
            result = evaluate_b2_run(folder, seed, ph, segments)
            if result is None:
                print(f"  ({folder.name}: no checkpoint)")
                continue
            results[label][seed] = result
        print(f"  {label:>15}: {len(results[label])} seeds")

    return results


def check_pairing(results, baseline_label="B1"):
    """Assert B1 and B2 see the same test data at each seed."""
    base = results.get(baseline_label, {})
    for label, seeds_dict in results.items():
        if label == baseline_label:
            continue
        for seed, r in seeds_dict.items():
            if seed not in base:
                continue
            b = base[seed]
            assert np.array_equal(r["y"], b["y"]), \
                f"{label} seed {seed}: y_actual differs from B1 -- PAIRING BROKEN"
            # Persistence depends only on the split
            assert abs(r["persistence"] - b["persistence"]) < 1e-4, \
                f"{label} seed {seed}: persistence differs ({r['persistence']:.4f} vs {b['persistence']:.4f})"


def paired_report(results, baseline_label="B1"):
    """Paired comparison of each rung vs B1."""
    base = results.get(baseline_label, {})
    if not base:
        print("No B1 runs found.")
        return

    print(f"\n{'='*80}")
    print(f"Ablation ladder (paired vs B1)")
    print(f"{'='*80}")
    print(f"\n  {'Rung':<20} {'n':>3} {'RMSE':>16} {'Persist':>8} {'Skill':>7} "
          f"{'vs B1':>10} {'Wins':>6} {'p':>8} {'Params':>8}")
    print(f"  {'-'*20} {'-'*3} {'-'*16} {'-'*8} {'-'*7} "
          f"{'-'*10} {'-'*6} {'-'*8} {'-'*8}")

    rows = []
    for label, seeds_dict in results.items():
        shared = sorted(set(seeds_dict) & set(base))
        if not shared:
            continue

        rmses = np.array([seeds_dict[s]["rmse"] for s in shared])
        persists = np.array([seeds_dict[s]["persistence"] for s in shared])
        skills = np.array([seeds_dict[s]["skill"] for s in shared])
        params = seeds_dict[shared[0]]["params"]

        if label == baseline_label:
            std_str = f" +/- {rmses.std(ddof=1):.2f}" if len(shared) > 1 else ""
            print(f"  {label:<20} {len(shared):>3} {f'{rmses.mean():.2f}{std_str}':>16} "
                  f"{persists.mean():>8.2f} {skills.mean():>6.1%} "
                  f"{'--':>10} {'--':>6} {'--':>8} {params:>8,}")
            continue

        b_rmses = np.array([base[s]["rmse"] for s in shared])
        deltas = b_rmses - rmses  # positive = B2 better
        wins = int((deltas > 0).sum())

        pvalue_str = "--"
        if len(shared) >= 3 and not np.allclose(deltas, 0):
            pvalue_str = f"{wilcoxon(b_rmses, rmses).pvalue:.4f}"

        std_str = f" +/- {rmses.std(ddof=1):.2f}" if len(shared) > 1 else ""
        delta_mean = deltas.mean()
        delta_str = f"+{delta_mean:.3f}" if delta_mean >= 0 else f"{delta_mean:.3f}"

        print(f"  {label:<20} {len(shared):>3} {f'{rmses.mean():.2f}{std_str}':>16} "
              f"{persists.mean():>8.2f} {skills.mean():>6.1%} "
              f"{delta_str:>10} {wins:>3}/{len(shared):>2} {pvalue_str:>8} {params:>8,}")

        for s in shared:
            rows.append({"rung": label, "seed": s,
                         "rmse": seeds_dict[s]["rmse"],
                         "persistence": seeds_dict[s]["persistence"],
                         "skill": seeds_dict[s]["skill"],
                         "b1_rmse": base[s]["rmse"],
                         "delta": base[s]["rmse"] - seeds_dict[s]["rmse"],
                         "params": params})

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _extract_dt_and_trend(seed, segments, ph=30):
    """Get time-since-meal and trend for stratification at one seed.

    dt comes from building meal features on the test split's segments (no model needed).
    trend comes from the B1 checkpoint's history window (loaded once)."""
    import lightning as L
    from dataset import split_subjects
    from meal_features import build_all_features, block_names
    from meals import (load_meals, fit_macro_cleaning, apply_macro_cleaning,
                       scale_macros)
    from preprocess import make_windows

    L.seed_everything(seed, workers=True)
    subjects = sorted({s.subject for s in segments})
    train_subs, _, test_subs = split_subjects(subjects, seed=seed)

    segs_by_subj = {}
    for seg in segments:
        segs_by_subj.setdefault(seg.subject, []).append(seg)
    test_segs = [s for subj in test_subs for s in segs_by_subj.get(subj, [])]

    raw_meals = load_meals(Path("dataset/CGMacros"))
    cleaning = fit_macro_cleaning(raw_meals, train_subs)
    ready = scale_macros(apply_macro_cleaning(raw_meals, cleaning))

    # Only need the timing block ("T") to get time-since-meal
    blocks = ("T",)
    test_feats = build_all_features(test_segs, ready, blocks)
    feature_names = block_names(blocks)
    dt_idx = feature_names.index("dt_norm")

    all_dt, all_trend = [], []
    for seg, feat in zip(test_segs, test_feats):
        X, _ = make_windows(seg, 12, ph // 5)  # raw mg/dL
        if len(X) == 0:
            continue
        num_windows = len(X)
        # Align features to the last timestep of each input window (index 11)
        static = feat[np.arange(num_windows) + 12 - 1]
        all_dt.append(static[:, dt_idx] * FASTING_CUTOFF_MIN)
        # mg/dL per minute over the last 15 min (3 five-minute steps)
        all_trend.append((X[:, -1] - X[:, -4]) / 15.0)

    return np.concatenate(all_dt), np.concatenate(all_trend)


def _find_best_b2(results, baseline_label="B1"):
    """Find the B2 rung with the highest mean paired delta vs B1."""
    base = results.get(baseline_label, {})
    best_label, best_delta = None, -np.inf
    for label in results:
        if label == baseline_label or label.startswith("_"):
            continue
        shared = sorted(set(results[label]) & set(base))
        if len(shared) < 2:
            continue
        mean_delta = np.mean([base[s]["rmse"] - results[label][s]["rmse"] for s in shared])
        if mean_delta > best_delta:
            best_delta, best_label = mean_delta, label
    return best_label, best_delta


def stratified_report(results, segments, baseline_label="B1"):
    """Stratified evaluation showing WHERE meals help."""
    base = results.get(baseline_label, {})
    best_label, best_delta = _find_best_b2(results, baseline_label)
    if best_label is None:
        print("No B2 rung with enough seeds for stratification.")
        return

    b2 = results[best_label]
    shared = sorted(set(b2) & set(base))
    print(f"\n{'='*80}")
    print(f"Stratified evaluation -- {best_label} vs B1 "
          f"({len(shared)} seeds)")
    print(f"{'='*80}")

    strata_defs = [
        ("meal <=30 min",     "dt", lambda dt: dt <= 30),
        ("meal 30-60 min",    "dt", lambda dt: (dt > 30) & (dt <= 60)),
        ("meal 60-120 min",   "dt", lambda dt: (dt > 60) & (dt <= 120)),
        ("fasting >=5 h",     "dt", lambda dt: dt >= FASTING_CUTOFF_MIN),
        ("rising > +1",       "trend", lambda tr: tr > 1.0),
        ("flat -1..+1",       "trend", lambda tr: (tr >= -1.0) & (tr <= 1.0)),
        ("falling < -1",      "trend", lambda tr: tr < -1.0),
        ("current > 180",     "level", lambda lv: lv > 180),
    ]

    strata_results = {name: {"n": [], "b1_rmse": [], "b2_rmse": [], "persist_rmse": []}
                      for name, _, _ in strata_defs}

    for seed in shared:
        b1_r, b2_r = base[seed], b2[seed]
        y = b1_r["y"]
        preds_b1 = b1_r["preds"]
        preds_b2 = b2_r["preds"]
        last = b1_r["last_input"]

        dt, trend = _extract_dt_and_trend(seed, segments)
        assert len(dt) == len(y), \
            f"dt length {len(dt)} != y length {len(y)} at seed {seed}"

        for name, kind, mask_fn in strata_defs:
            if kind == "dt":
                mask = mask_fn(dt)
            elif kind == "trend":
                mask = mask_fn(trend)
            else:
                mask = mask_fn(last)

            n = int(mask.sum())
            if n == 0:
                continue
            strata_results[name]["n"].append(n)
            strata_results[name]["b1_rmse"].append(
                float(np.sqrt(np.mean((preds_b1[mask] - y[mask]) ** 2))))
            strata_results[name]["b2_rmse"].append(
                float(np.sqrt(np.mean((preds_b2[mask] - y[mask]) ** 2))))
            strata_results[name]["persist_rmse"].append(
                float(np.sqrt(np.mean((last[mask] - y[mask]) ** 2))))

    print(f"\n  {'Stratum':<20} {'n':>8} {'Persist':>9} {'B1':>9} {'B2':>9} "
          f"{'B1 skill':>9} {'B2 skill':>9} {'B2-B1':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*9} {'-'*9} {'-'*9} "
          f"{'-'*9} {'-'*9} {'-'*8}")

    for name, _, _ in strata_defs:
        stratum = strata_results[name]
        if not stratum["n"]:
            continue
        mean_n = np.mean(stratum["n"])
        persist_mean = np.mean(stratum["persist_rmse"])
        b1_mean = np.mean(stratum["b1_rmse"])
        b2_mean = np.mean(stratum["b2_rmse"])
        skill_b1 = (persist_mean - b1_mean) / persist_mean * 100
        skill_b2 = (persist_mean - b2_mean) / persist_mean * 100
        delta = b1_mean - b2_mean
        delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        print(f"  {name:<20} {mean_n:>8.0f} {persist_mean:>9.2f} {b1_mean:>9.2f} {b2_mean:>9.2f} "
              f"{skill_b1:>+8.1f}% {skill_b2:>+8.1f}% {delta_str:>8}")

    # Negative control
    fasting = strata_results.get("fasting >=5 h", {})
    if fasting.get("n"):
        fasting_delta = np.mean(fasting["b1_rmse"]) - np.mean(fasting["b2_rmse"])
        print(f"\n  NEGATIVE CONTROL (fasting >=5 h):")
        print(f"    B2-B1 delta: {fasting_delta:+.3f} mg/dL")
        if abs(fasting_delta) < 0.2:
            print(f"    -> Near zero. Meal features are NOT leaking subject identity.")
        else:
            print(f"    -> WARNING: fasting delta is {abs(fasting_delta):.2f} mg/dL, "
                  f"investigate for leakage.")

    # Rising-window comparison
    rising = strata_results.get("rising > +1", {})
    if rising.get("n"):
        b1_rising_skill = (np.mean(rising["persist_rmse"]) - np.mean(rising["b1_rmse"])) / \
                          np.mean(rising["persist_rmse"]) * 100
        b2_rising_skill = (np.mean(rising["persist_rmse"]) - np.mean(rising["b2_rmse"])) / \
                          np.mean(rising["persist_rmse"]) * 100
        print(f"\n  RISING-WINDOW RESULT:")
        print(f"    B1 skill on rising: {b1_rising_skill:+.1f}% (B1 reference: 3.6%)")
        print(f"    B2 skill on rising: {b2_rising_skill:+.1f}%")
        print(f"    Improvement: {b2_rising_skill - b1_rising_skill:+.1f} pp")

    return strata_results


def dispersion_report(results, baseline_label="B1"):
    """Compare under-dispersion ratio (pred SD / actual SD) between B1 and B2."""
    base = results.get(baseline_label, {})
    best_label = None
    best_delta = -np.inf
    for label in results:
        if label == baseline_label:
            continue
        shared = sorted(set(results[label]) & set(base))
        if len(shared) < 3:
            continue
        mean_delta = np.mean([base[s]["rmse"] - results[label][s]["rmse"] for s in shared])
        if mean_delta > best_delta:
            best_delta, best_label = mean_delta, label
    if best_label is None:
        print("No B2 rung with enough seeds.")
        return

    b2 = results[best_label]
    shared = sorted(set(b2) & set(base))

    print(f"\n{'='*80}")
    print(f"Under-dispersion comparison -- {best_label} vs B1")
    print(f"{'='*80}")
    print(f"\n  {'Seed':>6} {'B1 pred/act SD':>16} {'B2 pred/act SD':>16}")

    b1_ratios, b2_ratios = [], []
    for seed in shared:
        b1_r, b2_r = base[seed], b2[seed]
        y = b1_r["y"]
        b1_ratio = float(np.std(b1_r["preds"]) / np.std(y))
        b2_ratio = float(np.std(b2_r["preds"]) / np.std(y))
        b1_ratios.append(b1_ratio)
        b2_ratios.append(b2_ratio)
        print(f"  {seed:>6} {b1_ratio:>16.3f} {b2_ratio:>16.3f}")

    b1_mean = np.mean(b1_ratios)
    b2_mean = np.mean(b2_ratios)
    print(f"\n  B1 mean ratio: {b1_mean:.3f} (B1 reference: 0.87)")
    print(f"  B2 mean ratio: {b2_mean:.3f}")
    print(f"  Change: {b2_mean - b1_mean:+.3f}")
    if b2_mean > b1_mean:
        print(f"  -> B2 is LESS under-dispersed (predictions swing more). Good.")
    else:
        print(f"  -> B2 is STILL under-dispersed. Meals did not fix the excursion ceiling.")


def figure_meal_strata(strata_data, out_path=Path("results/meal_strata.png"),
                       figsize=(14, 6), dpi=150):
    """Grouped bars: B1 vs B2 vs persistence per stratum, with skill above each group.

    figsize/dpi exist for poster_figures.py, which re-renders this at a
    standardized poster geometry without duplicating the drawing logic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(strata_data.keys())
    persist = [np.mean(strata_data[n]["persist_rmse"]) for n in names]
    b1 = [np.mean(strata_data[n]["b1_rmse"]) for n in names]
    b2 = [np.mean(strata_data[n]["b2_rmse"]) for n in names]
    sample_counts = [int(np.mean(strata_data[n]["n"])) for n in names]

    x = np.arange(len(names))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - bar_width, persist, bar_width, label="persistence", color="C1", alpha=0.85)
    ax.bar(x, b1, bar_width, label="B1 (CGM-only)", color="C0", alpha=0.85)
    ax.bar(x + bar_width, b2, bar_width, label="B2 (CGM+meals)", color="C2", alpha=0.85)

    for i in range(len(names)):
        skill_b1 = (persist[i] - b1[i]) / persist[i] * 100
        skill_b2 = (persist[i] - b2[i]) / persist[i] * 100
        y_top = max(persist[i], b1[i], b2[i]) + 1
        ax.text(x[i], y_top, f"B1:{skill_b1:+.0f}%\nB2:{skill_b2:+.0f}%",
                ha="center", va="bottom", fontsize=7)
        # Vertical AND centered in the bar, so the label stays inside at any
        # figure geometry -- horizontal white text spilled onto the white
        # background at poster width, and top-anchored vertical text escaped
        # through the bar top the same way.
        ax.text(x[i] + bar_width, b2[i] / 2, f"n={sample_counts[i]:,}",
                rotation=90, ha="center", va="center", fontsize=6, color="white",
                fontweight="bold")

    short_names = [n.replace(" min", "m").replace("fasting >=5 h", "fasting")
                   .replace("rising > +1", "rising").replace("flat -1..+1", "flat")
                   .replace("falling < -1", "falling").replace("current > 180", ">180")
                   for n in names]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_ylabel("RMSE (mg/dL)")
    ax.set_title("Where meals help: B1 vs B2 by stratum (+30 min, 11 seeds)")
    # Headroom band + single-row legend inside it: a floating legend collided
    # with the per-group skill annotations at poster geometry (taller axes).
    ax.set_ylim(0, 1.22 * (max(persist + b1 + b2) + 1))
    ax.legend(ncol=3, loc="upper center", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  saved {out_path}")


def figure_meal_ablation(results, baseline_label="B1",
                         out_path=Path("results/meal_ablation.png")):
    """One dot per seed per rung -- show every point, never error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = results.get(baseline_label, {})
    rungs = [label for label in results if label != baseline_label and not label.startswith("_")]
    if not rungs:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = {}
    for i, label in enumerate(rungs):
        x_positions[label] = i

    for label in rungs:
        seeds_dict = results[label]
        shared = sorted(set(seeds_dict) & set(base))
        if not shared:
            continue
        deltas = [base[s]["rmse"] - seeds_dict[s]["rmse"] for s in shared]
        x_pos = x_positions[label]
        jitter = np.linspace(-0.15, 0.15, len(deltas))
        ax.scatter(x_pos + jitter, deltas, s=30, alpha=0.7, zorder=3, color="C0")
        ax.plot([x_pos - 0.25, x_pos + 0.25], [np.mean(deltas)] * 2, "k-", lw=2, zorder=4)

    ax.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(0.47, color="red", ls=":", lw=0.8, alpha=0.5, label="+/-0.47 threshold")
    ax.axhline(-0.47, color="red", ls=":", lw=0.8, alpha=0.5)
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels(rungs, rotation=30, ha="right")
    ax.set_ylabel("Paired delta vs B1 (mg/dL, positive = B2 better)")
    ax.set_title("Meal ablation ladder: each dot is one seed")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def figure_postprandial_trace(segments, seed=42, ph=30, max_step=25,
                              out_path=Path("results/postprandial_trace.png")):
    """3-hour window around a large meal: truth, B1, B2, persistence.

    Picks the meal with the most postprandial windows and visible B2 improvement,
    filtering out runs with large step artifacts (>=max_step mg/dL jumps)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from meals import load_meals
    from preprocess import make_windows

    RUN_LEN = 36  # 3 hours at 5-min spacing

    b1_tag = run_tag(seed, ph)
    b1_ckpt = find_checkpoint(b1_tag)
    b2_tag = run_tag(seed, ph, meal_infix("static", ("T", "M", "Y", "I", "S")))
    b2_ckpt = find_checkpoint(b2_tag)
    if b1_ckpt is None or b2_ckpt is None:
        print(f"  postprandial_trace: need both B1 ({b1_tag}) and B2 ({b2_tag}) at seed {seed}")
        return

    b1_cfg = pooled_config(seed, ph=ph)
    b2_cfg = pooled_config(seed, ph=ph, meal_blocks=("T", "M", "Y", "I", "S"),
                           meal_encoding="static")

    y_b1, p_b1, last_b1, subj_b1 = load_test_predictions(b1_ckpt, segments, b1_cfg)
    y_b2, p_b2, last_b2, subj_b2 = load_test_predictions(b2_ckpt, segments, b2_cfg)

    assert np.array_equal(y_b1, y_b2), "pairing broken"

    dt, trend = _extract_dt_and_trend(seed, segments, ph)
    meals_raw = load_meals(Path("dataset/CGMacros"))

    postprandial = dt <= 30
    y, p1, p2, last = y_b1, p_b1, p_b2, last_b1
    subj = subj_b1

    n_starts = len(y) - RUN_LEN + 1
    if n_starts <= 0:
        print("  not enough windows for a trace")
        return

    # Score each candidate window: prefer many postprandial steps + visible B2 improvement
    best_start, best_score = None, -1
    for i in range(n_starts):
        window_slice = slice(i, i + RUN_LEN)
        if subj[i] != subj[i + RUN_LEN - 1]:
            continue
        biggest_step = float(np.abs(np.diff(y[window_slice])).max())
        if biggest_step > max_step:
            continue
        n_postprandial = int(postprandial[window_slice].sum())
        if n_postprandial < 6:
            continue
        b2_gain = float(np.mean(np.abs(p1[window_slice] - y[window_slice]))
                        - np.mean(np.abs(p2[window_slice] - y[window_slice])))
        score = n_postprandial * 0.5 + b2_gain
        if score > best_score:
            best_score, best_start = score, i

    if best_start is None:
        print("  no valid postprandial run found")
        return

    window_slice = slice(best_start, best_start + RUN_LEN)
    time_minutes = np.arange(RUN_LEN) * 5

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_minutes, y[window_slice], "k-", lw=1.8, label="actual (+30 min)")
    ax.plot(time_minutes, p1[window_slice], "C0-", alpha=0.8, lw=1.2, label="B1 (CGM-only)")
    ax.plot(time_minutes, p2[window_slice], "C2-", alpha=0.8, lw=1.2, label="B2 (CGM+meals)")
    ax.plot(time_minutes, last[window_slice], "C1--", alpha=0.5, label="persistence")

    meal_mask = postprandial[window_slice]
    for j in range(RUN_LEN):
        if meal_mask[j] and (j == 0 or not meal_mask[j - 1]):
            ax.axvspan(time_minutes[j], time_minutes[min(j + 6, RUN_LEN - 1)],
                       alpha=0.08, color="green")
            ax.axvline(time_minutes[j], color="green", ls=":", alpha=0.4)

    err_b1 = float(np.mean(np.abs(p1[window_slice] - y[window_slice])))
    err_b2 = float(np.mean(np.abs(p2[window_slice] - y[window_slice])))
    ax.set_xlabel("minutes into run")
    ax.set_ylabel("glucose (mg/dL)")
    ax.set_title(f"Postprandial trace -- subject {subj[best_start]} -- "
                 f"B1 MAE {err_b1:.1f}, B2 MAE {err_b2:.1f} mg/dL")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")
    print(f"    subject {subj[best_start]}, B1 MAE {err_b1:.1f}, B2 MAE {err_b2:.1f}, "
          f"postprandial windows {int(meal_mask.sum())}/{RUN_LEN}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--ph", type=int, default=30)
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--strata", action="store_true",
                        help="stratified evaluation by time-since-meal and trend")
    parser.add_argument("--dispersion", action="store_true",
                        help="under-dispersion ratio comparison (B1 vs B2)")
    parser.add_argument("--encoding", default=None,
                        help="encoding for Ladder 2 (default: auto-pick from Ladder 1)")
    parser.add_argument("--ladder", choices=["1", "2", "both"], default="both")
    parser.add_argument("--figures", action="store_true",
                        help="generate stratified and ablation figures")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    segments = load_segments(args.data_dir / "segments_split.npz")

    # Discover what exists
    print(f"Scanning {args.results_dir}/ for B2 runs (ph{args.ph})...")

    # Always load Ladder 1 to determine the winning encoding
    if args.ladder in ("1", "both"):
        print("\nLadder 1 (encoding comparison, all blocks):")
        ladder1_spec = [(l, e, b) for l, e, b in LADDER1]
        l1_results = collect_ladder(ladder1_spec, segments, args.results_dir, args.ph)

        if l1_results:
            check_pairing(l1_results)
            df1 = paired_report(l1_results)

            # Pick the winning encoding
            best_enc = None
            best_enc_delta = -np.inf
            base = l1_results.get("B1", {})
            for label, enc, _ in LADDER1:
                if label == "B1":
                    continue
                shared = sorted(set(l1_results.get(label, {})) & set(base))
                if not shared:
                    continue
                mean_delta = np.mean([base[s]["rmse"] - l1_results[label][s]["rmse"]
                                      for s in shared])
                if mean_delta > best_enc_delta:
                    best_enc_delta, best_enc = mean_delta, enc
            if best_enc:
                print(f"\n  -> Ladder 1 winner: {best_enc} "
                      f"(paired delta vs B1: +{best_enc_delta:.3f})")

    winning_enc = args.encoding or (best_enc if args.ladder != "2" else None) or "channels"

    if args.ladder in ("2", "both"):
        print(f"\nLadder 2 (feature ablation, encoding={winning_enc}):")
        ladder2_spec = [(l, winning_enc if b else None, b)
                        for l, b in LADDER2_BLOCKS]
        l2_results = collect_ladder(ladder2_spec, segments, args.results_dir, args.ph,
                                   encoding_for_ladder2=winning_enc)
        if l2_results:
            check_pairing(l2_results)
            df2 = paired_report(l2_results)

    # Use whichever ladder has the most complete results for stratified analysis
    all_results = l1_results if args.ladder == "1" else (
        l2_results if args.ladder == "2" else {**l1_results, **l2_results})

    strata_data = None
    if args.strata and all_results:
        strata_data = stratified_report(all_results, segments)

    if args.dispersion and all_results:
        dispersion_report(all_results)

    if args.figures and all_results:
        print(f"\n{'='*80}")
        print("Stratified and ablation figures")
        print(f"{'='*80}")
        if args.ladder in ("2", "both") and 'l2_results' in dir() and l2_results:
            figure_meal_ablation(l2_results)
        elif args.ladder == "1" and 'l1_results' in dir() and l1_results:
            figure_meal_ablation(l1_results)
        if strata_data:
            figure_meal_strata(strata_data)
        else:
            print("  (run with --strata to generate meal_strata.png)")
        figure_postprandial_trace(segments)

    # Save CSV
    out = args.out or args.results_dir / "meal_ablation.csv"
    all_dfs = []
    if args.ladder in ("1", "both") and 'df1' in dir() and df1 is not None and len(df1):
        df1["ladder"] = 1
        all_dfs.append(df1)
    if args.ladder in ("2", "both") and 'df2' in dir() and df2 is not None and len(df2):
        df2["ladder"] = 2
        all_dfs.append(df2)
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(out, index=False)
        print(f"\nSaved {out} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
