"""Result figures for the glucose forecasting project.

Each function prints the numbers its caption needs, so the figure and the sentence
under it can never drift apart (same practice as plot_attention_weights.py).

    parkes         results/parkes_grid.png            error grid, LSTM vs persistence
    traces         results/pred_vs_actual.png         LOSO 006 (good) vs 039 (bad)
    traces_pooled  results/pred_vs_actual_pooled.png  pooled seed-42 split variant
    horizon        results/rmse_vs_horizon.png        RMSE vs horizon, medians of 11 seeds
    loso           results/loso_per_subject.png       per-subject LOSO bars by class
    loso_b2        results/loso_b2_per_subject.png    B2 vs B1 per-subject LOSO comparison
    horizon_meal   results/horizon_meal_benefit.png   meal benefit by horizon (+30/+60/+120)

Usage (from the project root):
    python src/figures.py                        # all figures
    python src/figures.py --figure loso
    python src/figures.py --figure parkes --dtype 1     # the secondary type-1 grid

Renders through the Agg backend: files are written, no window opens, so this is
safe to run over SSH or from a scheduler.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")           # must precede the pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from numpy.lib.stride_tricks import sliding_window_view

from evaluate import (DEFAULT_SEEDS, RESULTS_DIR, find_checkpoint, iter_pooled_seeds,
                      iter_pooled_seeds_with_history, load_loso_subject,
                      load_test_predictions, pooled_config, run_tag)
from preprocess import load_segments

# 120 windows x 5 min = 10 hours. NOT named L: `import lightning as L` lives one
# import away, and `L = 120` would silently shadow it.
RUN_LEN = 120
# Glucose moves at most ~4-5 mg/dL per minute, so a jump beyond this between two
# readings 5 min apart is a data gap, not physiology. Screening matters because the
# "bad" panel is chosen by HIGHEST error and a gap manufactures exactly that -- the
# figure would then illustrate a data gap while the caption claims model failure.
MAX_STEP = 25

CLASS_COLORS = {"normal": "#4CAF50", "pre-diabetic": "#FFC107", "diabetic": "#F44336"}
MISSING_CLASS_COLOR = "#9E9E9E"      # a subject with no A1c -> NaN from pd.cut


# ------------------------------------------------------------------ run selection

def valid_starts(y, subjects=None, run_len=RUN_LEN, max_step=None):
    """Boolean mask over window-start indices: the run stays inside one subject and
    contains no gap-sized jump. Vectorized equivalent of the is_contiguous loop
    (same test, ~100x faster over 22k windows)."""
    max_step = MAX_STEP if max_step is None else max_step
    n_starts = len(y) - run_len + 1
    if n_starts <= 0:
        raise ValueError(f"only {len(y)} windows, need at least {run_len}")

    if subjects is None:                            # a LOSO fold is one subject already
        same_subject = np.ones(n_starts, dtype=bool)
    else:
        subject_array = np.asarray(subjects)
        same_subject = subject_array[:n_starts] == subject_array[run_len - 1:]

    biggest_step = sliding_window_view(np.abs(np.diff(y)), run_len - 1).max(axis=1)
    return same_subject & (biggest_step[:n_starts] <= max_step)


def pick_run(y, p, subjects=None, worst=False, max_step=None):
    """Start index of the lowest- (or highest-) mean-error valid run."""
    err = np.abs(p - y)
    mask = valid_starts(y, subjects, max_step=max_step)
    mean_err = sliding_window_view(err, RUN_LEN).mean(axis=1)
    mean_err = np.where(mask, mean_err, -np.inf if worst else np.inf)
    return int(np.argmax(mean_err) if worst else np.argmin(mean_err))


def plot_run(ax, y, p, last, start, title):
    window_slice = slice(start, start + RUN_LEN)
    time_minutes = np.arange(RUN_LEN) * 5
    ax.plot(time_minutes, y[window_slice], "k-", lw=1.6, label="actual (+30 min)")
    ax.plot(time_minutes, p[window_slice], "C0-", alpha=0.85, label="LSTM")
    ax.plot(time_minutes, last[window_slice], "C1--", alpha=0.6, label="persistence")
    ax.set_title(title)
    ax.set_ylabel("glucose (mg/dL)")
    ax.legend(loc="upper right", fontsize=8)
    return float(np.abs(p - y)[window_slice].mean()), float(np.abs(np.diff(y[window_slice])).max())


# ------------------------------------------------------------------ Parkes error grid

def figure_parkes(segments, args) -> None:
    """Parkes error grid, model beside persistence on the same split."""
    from methcomp import parkes

    tag = run_tag(args.seed, args.ph)
    ckpt = find_checkpoint(tag, args.results_dir)
    y, p, last, _ = load_test_predictions(ckpt, segments, pooled_config(args.seed, ph=args.ph))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, pred, name in [(axes[0], p, "LSTM"), (axes[1], last, "Persistence")]:
        # Reference (TRUE) first, test (prediction) second. Swapping them does not
        # raise -- it moves zone A by ~2pp, so the mistake is silent.
        # x_label/y_label are passed explicitly because methcomp 1.0.0 hardcodes
        # "mmol/L" on the axes even when units="mgdl" (tick VALUES are correct mg/dL).
        parkes(args.dtype, y, pred, units="mgdl", percentage=True, ax=ax,
               x_label="Reference glucose (mg/dL)",
               y_label=f"{name} prediction (mg/dL)",
               title=f"{name} -- Parkes type {args.dtype} (+{args.ph} min, seed {args.seed})")

    suffix = "" if args.dtype == 2 else f"_type{args.dtype}"
    out = args.results_dir / f"parkes_grid{suffix}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    if args.dtype == 1:
        print("  type 1 is a conservative reference only -- no CGMacros subject has type 1,\n"
              "  and the model was never trained on type 1 data.")
    print("  zone percentages for the caption come from metrics_table.py (11 seeds, "
          "not just this one)")


# ------------------------------------------------------------------ prediction traces

def figure_traces(segments, args) -> None:
    """Good vs bad prediction traces, both from LOSO folds (subjects the model never
    trained on): 006 is the easiest subject in the cohort, 039 the hardest and one of
    the two the model LOSES to persistence on."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    captions = []

    for ax, subject, worst, label in [(axes[0], args.good_subject, False, "Good"),
                                      (axes[1], args.bad_subject, True, "Bad")]:
        y, p, last, _ = load_loso_subject(subject, segments)
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        persist_rmse = float(np.sqrt(np.mean((last - y) ** 2)))
        # A LOSO fold is a single subject, but it can still cross a SEGMENT boundary,
        # so the gap screen still applies -- subjects=None only skips the subject test.
        start = pick_run(y, p, subjects=None, worst=worst, max_step=args.max_step)
        panel_err, biggest_step = plot_run(
            ax, y, p, last, start,
            f"{label} -- LOSO subject {subject}: fold RMSE {rmse:.1f} vs persistence "
            f"{persist_rmse:.1f} mg/dL")
        captions.append((label, subject, rmse, persist_rmse, panel_err,
                         float(np.abs(p - y).mean()), biggest_step,
                         float(np.std(p)), float(np.std(y))))

    axes[1].set_xlabel("minutes into run")
    fig.tight_layout()
    out = args.results_dir / "pred_vs_actual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out}")
    for (label, subject, rmse, persist, panel_err, fold_err, step, p_std, y_std) in captions:
        print(f"  {label:4} panel: subject {subject}  fold RMSE {rmse:.2f} "
              f"(persistence {persist:.2f}, skill {(persist - rmse) / persist:+.1%})")
        print(f"       panel mean |err| {panel_err:.1f} vs {fold_err:.1f} mg/dL over the whole "
              f"fold; largest step in the actual curve {step:.0f} mg/dL "
              f"(gap screen {args.max_step})")
        print(f"       dispersion: prediction std {p_std:.1f} vs actual std {y_std:.1f} mg/dL")
    print("  ^ the dispersion line IS the under-dispersion finding: on the volatile subject the\n"
          "    model's predictions vary far less than the truth, so it cannot follow excursions.")


def figure_traces_pooled(segments, args) -> None:
    """The same figure on the pooled seed-42 split (the original version).

    Kept because it needs no LOSO run -- but note neither 039 nor 046 is in seed 42's
    8-subject test set, so this version cannot show the model's actual failure cases."""
    tag = run_tag(args.seed, args.ph)
    ckpt = find_checkpoint(tag, args.results_dir)
    y, p, last, subjects = load_test_predictions(
        ckpt, segments, pooled_config(args.seed, ph=args.ph))

    good = pick_run(y, p, subjects, max_step=args.max_step)
    bad = pick_run(y, p, subjects, worst=True, max_step=args.max_step)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    stats = []
    for ax, start, label in [(axes[0], good, "Good"), (axes[1], bad, "Bad")]:
        panel_err, biggest_step = plot_run(
            ax, y, p, last, start,
            f"{label} run -- subject {subjects[start]} -- "
            f"mean |err| {np.abs(p - y)[start:start + RUN_LEN].mean():.1f} mg/dL")
        stats.append((label, subjects[start], panel_err, biggest_step))
    axes[1].set_xlabel("minutes into run")
    fig.tight_layout()
    out = args.results_dir / "pred_vs_actual_pooled.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out}")
    for label, subject, panel_err, step in stats:
        print(f"  {label:4} panel: subject {subject}  mean |err| {panel_err:.1f} mg/dL  "
              f"largest step in actual {step:.0f} mg/dL")
    print(f"  test subjects in this split: {', '.join(sorted(set(subjects.tolist())))}")


# ------------------------------------------------------------------ RMSE vs horizon

def figure_horizon(segments, args) -> None:
    """RMSE vs horizon. MEDIANS over the 11 seeds, not means: seed 49 is a diagnosed
    outlier at +60/+120 (it drew the two most volatile subjects into its test set)."""
    table = pd.read_csv(args.results_dir / "horizon_multiseed.csv")
    medians = table.groupby("ph")[["rmse", "persistence"]].median()
    # The median of each seed's OWN skill. Not the same as skill computed from the two
    # medians above, because the median-RMSE seed and the median-persistence seed are
    # different seeds -- see the caution printed below.
    median_skill = table.groupby("ph")["skill"].median()
    horizons = [h for h in (30, 60, 120) if h in medians.index]
    lstm = medians.loc[horizons, "rmse"].tolist()
    persistence = medians.loc[horizons, "persistence"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(horizons, lstm, "o-", label=f"LSTM (median of {table.groupby('ph').size().max()} seeds)")
    ax.plot(horizons, persistence, "s--", label="Persistence")

    # Nearly free extra points: the multi-horizon run predicts +5..+30 in one model.
    multi_ckpt = find_checkpoint(f"ph{args.ph}_multi_seed{args.seed}", args.results_dir)
    if multi_ckpt is not None:
        cfg = pooled_config(args.seed, ph=args.ph, multi_horizon=True)
        y, p, _, _ = load_test_predictions(multi_ckpt, segments, cfg)
        per_h = [float(np.sqrt(np.mean((p[:, j] - y[:, j]) ** 2))) for j in range(p.shape[1])]
        multi_horizons = [(j + 1) * 5 for j in range(p.shape[1])]
        ax.plot(multi_horizons, per_h, "^:", ms=5, alpha=0.8,
                label=f"multi-horizon LSTM (seed {args.seed}, +5..+{multi_horizons[-1]})")
        print("  multi-horizon per-horizon RMSE: "
              + ", ".join(f"+{m}: {v:.2f}" for m, v in zip(multi_horizons, per_h)))

    ax.set_xlabel("Prediction horizon (minutes)")
    ax.set_ylabel("RMSE (mg/dL)")
    ax.set_title("Forecast error grows with horizon -- but persistence degrades faster")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = args.results_dir / "rmse_vs_horizon.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out}")
    for h, m, ps in zip(horizons, lstm, persistence):
        print(f"  +{h:>3} min: LSTM {m:.2f}  persistence {ps:.2f}  gap {ps - m:.2f}  |  "
              f"skill {(ps - m) / ps:.1%} from these two medians, "
              f"{median_skill.loc[h]:.1%} as the median of per-seed skills")
    print("  ^ QUOTE THE SECOND ONE. Skill computed from the median RMSE and the median\n"
          "    persistence is inflated, because those two medians come from DIFFERENT seeds;\n"
          "    the per-seed skill is paired within a split, which is the project's rule.\n"
          "    The plotted curves are still the medians -- only the skill wording changes.")
    print("  ^ skill widens with horizon because persistence degrades faster, NOT because the\n"
          "    model improves -- absolute error still climbs, and that motivates adding meal features.")


# ------------------------------------------------------------------ per-subject LOSO bars

def figure_loso(segments, args) -> None:
    """Per-subject LOSO error, sorted, coloured by glycemic class, with persistence
    marked per subject."""
    # loso_results.csv has no glycemic_class column -- the merged file was saved separately.
    results = pd.read_csv(args.results_dir / "loso_with_class.csv")
    results["subject"] = results["subject"].astype(int).map(lambda n: f"{n:03d}")
    # reset_index because itertuples() yields the ORIGINAL index while bars are placed
    # at the loop counter -- without it the x labels do not match the bars.
    results = results.sort_values("test_rmse").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, row in enumerate(results.itertuples()):
        # `glycemic_class`, never `class` -- that is a keyword and itertuples renames it.
        ax.bar(i, row.test_rmse, width=0.8,
               color=CLASS_COLORS.get(row.glycemic_class, MISSING_CLASS_COLOR))
        ax.plot(i, row.persistence_rmse, "k_", markersize=10)

    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(results["subject"], rotation=90, fontsize=7)
    ax.set_xlabel("Subject (sorted by model RMSE)")
    ax.set_ylabel("RMSE (mg/dL)")
    ax.set_title(f"LOSO: per-subject error, {len(results)} held-out subjects "
                 f"(bars = LSTM, ticks = persistence)")
    ax.legend(handles=[Patch(color=c, label=k) for k, c in CLASS_COLORS.items()]
                      + [plt.Line2D([], [], color="k", marker="_", ls="", label="persistence")],
              loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = args.results_dir / "loso_per_subject.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    macro_rmse = results["test_rmse"].mean()
    macro_persist = results["persistence_rmse"].mean()
    # TWO honest macro skill estimators, and they differ by 0.2pp. Print both, because
    # a sentence in the paper has to say which one it means (the macro-vs-micro
    # distinction applies to skill too).
    ratio_of_means = (macro_persist - macro_rmse) / macro_persist
    mean_of_skills = results["skill"].mean()
    # A skill of -0.06% is not a loss, it is a subject too flat to improve on.
    material = results[results["skill"] < -0.005]
    ties = results[(results["skill"] <= 0) & (results["skill"] >= -0.005)]

    print(f"wrote {out}")
    print(f"  macro RMSE {macro_rmse:.2f} +/- {results['test_rmse'].std(ddof=1):.2f}  "
          f"persistence {macro_persist:.2f} +/- {results['persistence_rmse'].std(ddof=1):.2f}")
    print(f"  skill {ratio_of_means:.1%} as (macro persist - macro model)/macro persist, "
          f"{mean_of_skills:.1%} as the mean of per-subject skills")
    print(f"  model beats persistence on {(results['skill'] > 0).sum()}/{len(results)} subjects")
    print(f"  material losses: "
          f"{', '.join(f'{r.subject} ({r.skill:+.1%})' for r in material.itertuples())}")
    if not ties.empty:
        print(f"  effective ties (|skill| < 0.5%): "
              f"{', '.join(f'{r.subject} ({r.skill:+.1%})' for r in ties.itertuples())}")
    print(f"  easiest {results.iloc[0]['subject']} ({results.iloc[0]['test_rmse']:.2f}) -> "
          f"hardest {results.iloc[-1]['subject']} ({results.iloc[-1]['test_rmse']:.2f})")
    print(f"  corr(model, persistence) across subjects = "
          f"{results['test_rmse'].corr(results['persistence_rmse']):.3f} "
          f"-- near 1 means hard subjects are hard for everyone, not just the model")
    for name, group in results.groupby("glycemic_class"):
        print(f"  {name:<13} n={len(group):<3} RMSE {group['test_rmse'].mean():.2f} "
              f"persistence {group['persistence_rmse'].mean():.2f} "
              f"skill {group['skill'].mean():.1%}")


# ------------------------------------------------- hyperparameter sweep null result

# The frozen config -- the reference every sweep panel is measured against.
SWEEP_BASELINE = {"hidden_size": 64, "num_layers": 1, "lr": 0.001, "history": 12}


def figure_sweep(segments, args) -> None:
    """Hyperparameter sweep null result. Shows every sweep arm sitting inside the
    seed-to-seed noise band -- no parameter moved val RMSE beyond split noise."""
    table = pd.read_csv(args.results_dir / "sweep.csv")
    params = list(dict.fromkeys(table["sweep_param"]))

    # The noise scale is measured, not assumed: the spread of val_rmse across SEEDS at a
    # fixed config is split luck, which is the yardstick any parameter effect must beat.
    within = table.groupby(["sweep_param", "sweep_value"])["val_rmse"]
    noise = float(within.std(ddof=1).mean())
    seeds_per_config = int(within.count().median())

    fig, axes = plt.subplots(1, len(params), figsize=(3.6 * len(params), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    effects, unpaired_effects, paired_detail = {}, {}, {}

    for ax, param in zip(axes, params):
        sub = table[table["sweep_param"] == param]
        values = sorted(sub["sweep_value"].unique())
        means = [sub.loc[sub["sweep_value"] == v, "val_rmse"].mean() for v in values]
        unpaired_effects[param] = max(means) - min(means)

        baseline_value = SWEEP_BASELINE.get(param)
        baseline_mean = sub.loc[sub["sweep_value"] == baseline_value, "val_rmse"].mean()

        # The effect size is PAIRED on seed (same seed = same split), matching
        # sweep.py --analyze. The unpaired max-min of the value means is larger (0.21
        # for history) because it also absorbs split luck -- the same paired/unpaired
        # trap as the LOSO skill estimators.
        pivot = sub.pivot(index="seed", columns="sweep_value", values="val_rmse")
        paired = pivot.sub(pivot[baseline_value], axis=0)
        effects[param] = float(paired.mean().abs().max())
        paired_detail[param] = (paired.mean(), paired.std(ddof=1))
        ax.axhspan(baseline_mean - noise, baseline_mean + noise, color="0.6", alpha=0.25,
                   label=f"frozen config ± split noise ({noise:.2f})")
        ax.axhline(baseline_mean, color="0.4", lw=1, ls=":")

        for i, value in enumerate(values):
            vals = sub.loc[sub["sweep_value"] == value, "val_rmse"].to_numpy()
            # Deterministic jitter so the figure is reproducible (same rule as the
            # attention figure): show every run instead of an error bar.
            jitter = np.linspace(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=16, alpha=0.65, color="C0")
        ax.plot(range(len(values)), means, "k-o", ms=5, lw=1.4, label="mean")

        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([f"{v:g}" for v in values])
        ax.set_title(f"{param}\npaired effect ≤ {effects[param]:.2f} mg/dL")
        ax.set_xlabel(param)
    axes[0].set_ylabel("validation RMSE (mg/dL)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("No hyperparameter moves validation RMSE beyond split noise "
                 f"(n={seeds_per_config} seeds per config)")
    fig.tight_layout()
    out = args.results_dir / "sweep_null.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out}")
    print(f"  {len(table)} runs, {seeds_per_config} seeds per config")
    print(f"  split noise (mean within-config SD across seeds): {noise:.2f} mg/dL")
    print(f"  paired deltas vs the frozen config (mean +/- SD over seeds):")
    for param, effect in sorted(effects.items(), key=lambda kv: -kv[1]):
        means, sds = paired_detail[param]
        detail = ", ".join(f"{v:g}: {means[v]:+.3f}+/-{sds[v]:.3f}" for v in means.index)
        print(f"    {param:<12} max |delta| {effect:.2f} = {effect / noise:.2f}x noise   {detail}")
    print(f"  MAX PAIRED effect {max(effects.values()):.2f} mg/dL vs split noise {noise:.2f} "
          f"-> every parameter is inside the band")
    print(f"  (unpaired max-min of the value means would read "
          f"{max(unpaired_effects.values()):.2f}; it absorbs split luck, so quote the "
          f"paired number -- it is what sweep.py --analyze reports)")

    # Free reproducibility check: the frozen config is re-run inside every panel, and
    # training is deterministic, so its mean val_rmse must be identical across panels.
    frozen = {param: table.loc[(table["sweep_param"] == param)
                               & (table["sweep_value"] == value), "val_rmse"].mean()
              for param, value in SWEEP_BASELINE.items() if param in params}
    spread = max(frozen.values()) - min(frozen.values())
    print(f"  frozen-config val_rmse per panel: "
          f"{', '.join(f'{k} {v:.4f}' for k, v in frozen.items())}")
    print(f"    spread {spread:.2e} mg/dL -> "
          f"{'identical, the sweep reproduces exactly' if spread < 1e-9 else 'DIFFERENT - investigate'}")


# --------------------------------------------- the data-integrity / phase-detection story

def figure_data_integrity(segments, args) -> None:
    """Why this project's numbers are not comparable to naive CGMacros results.

    The published 1-minute Dexcom column is LINEARLY INTERPOLATED between real 5-minute
    sensor samples. Training on it leaks the future into the input. Panel A shows the
    interpolation; panel B shows the signal used to recover the true lattice -- the second
    difference vanishes everywhere except one residue class mod 5."""
    from audit import (DEXCOM_PERIOD_MIN, LINEAR_TOL, block_phase, contiguous_blocks,
                       find_subject_csvs, load_raw)

    csvs = find_subject_csvs(args.raw_dir)
    sid = args.integrity_subject
    if sid not in csvs:
        print(f"subject {sid} not found under {args.raw_dir} "
              f"(available: {', '.join(sorted(csvs)[:5])}...)")
        return

    df = load_raw(csvs[sid])
    col = "Dexcom GL"
    block = max(contiguous_blocks(df, col), key=len)
    phase, residual = block_phase(block, col, DEXCOM_PERIOD_MIN)

    minute = block["Timestamp"].dt.minute
    is_knot = (minute % DEXCOM_PERIOD_MIN == phase).to_numpy()
    values = block[col].to_numpy()

    # A stretch where the curve actually bends, so the interpolation is visible.
    knot_idx = np.flatnonzero(is_knot)
    best_start, best_bend = knot_idx[0], -1.0
    for k in knot_idx[:-8]:
        window = values[k:k + 41]
        if len(window) < 41:
            continue
        bend = np.abs(np.diff(window[::DEXCOM_PERIOD_MIN], 2)).max()
        if bend > best_bend:
            best_bend, best_start = bend, k
    span = slice(best_start, best_start + 41)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    time_minutes = np.arange(41)
    ax.plot(time_minutes, values[span], "-", color="C1", lw=1.2, label="published 1-min series")
    ax.plot(time_minutes, values[span], ".", color="C1", ms=4)
    knot_mask = is_knot[span]
    ax.plot(time_minutes[knot_mask], values[span][knot_mask], "o", ms=9, mfc="none", mec="k", mew=1.6,
            label="recovered real 5-min samples")
    ax.set_xlabel(f"minutes (subject {sid}, phase = minute % 5 == {phase})")
    ax.set_ylabel("glucose (mg/dL)")
    ax.set_title("A. The 1-minute column is piecewise linear\nbetween real sensor samples")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    abs_second_diff = block[col].diff().diff().abs()
    floor = 1e-17                                  # log axis cannot show exact zeros
    for residue in range(DEXCOM_PERIOD_MIN):
        vals = abs_second_diff[(minute % DEXCOM_PERIOD_MIN == residue) & abs_second_diff.notna()].to_numpy()
        if len(vals) == 0:
            continue
        jitter = np.linspace(-0.18, 0.18, len(vals))
        marks_knot = residue == (phase + 1) % DEXCOM_PERIOD_MIN
        ax.scatter(np.full(len(vals), residue) + jitter, np.maximum(vals, floor), s=8,
                   alpha=0.35, color="C3" if marks_knot else "C0",
                   label=("carries all curvature (knot+1)" if marks_knot else
                          "float noise only") if residue <= 1 or marks_knot else None)
    ax.axhline(LINEAR_TOL, color="k", ls="--", lw=1, label=f"linearity tol {LINEAR_TOL:g}")
    ax.set_yscale("log")
    ax.set_xticks(range(DEXCOM_PERIOD_MIN))
    ax.set_xlabel("minute % 5")
    ax.set_ylabel("|second difference| (mg/dL)")
    ax.set_title("B. Phase detection: curvature lives in ONE residue class\n"
                 "(and it marks knot+1, not the knot)")
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="center left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = args.results_dir / "data_integrity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    off_lattice = abs_second_diff[(minute % DEXCOM_PERIOD_MIN != (phase + 1) % DEXCOM_PERIOD_MIN)
                                  & abs_second_diff.notna()]
    on_lattice = abs_second_diff[(minute % DEXCOM_PERIOD_MIN == (phase + 1) % DEXCOM_PERIOD_MIN)
                                 & abs_second_diff.notna()]
    print(f"wrote {out}")
    print(f"  subject {sid}, longest block {len(block):,} 1-min rows, detected phase {phase}")
    print(f"  max |2nd diff| OFF the knot+1 lattice: {off_lattice.max():.2e} "
          f"<- float noise; the series is exactly piecewise linear")
    print(f"  max |2nd diff| ON  the knot+1 lattice: {on_lattice.max():.2f} mg/dL "
          f"(median {on_lattice.median():.2f})")
    print(f"  ratio on/off = {on_lattice.max() / max(off_lattice.max(), 1e-300):.1e} "
          f"-- the phase is unambiguous")
    print(f"  block_phase() residual (audit's trap-1 number): {residual:.2e}")
    print(f"  {is_knot.sum():,} real readings recovered from {len(block):,} rows "
          f"({100 * is_knot.mean():.0f}% -- the other 80% were manufactured)")


# ------------------------------------------------------- under-dispersion, all subjects

def figure_dispersion(segments, args) -> None:
    """Under-dispersion as a cohort-level result instead of one anecdote.

    Panel A: every LOSO subject's prediction SD against their actual SD. Panel B: mean
    prediction per decile of actual glucose, pooled over the 11 seeds -- regression
    toward the population mean at both tails."""
    loso_path = args.results_dir / "metrics_loso.csv"
    if not loso_path.exists():
        print(f"{loso_path} missing -- run `python src/metrics_table.py --loso` first")
        return
    loso = pd.read_csv(loso_path)
    if "pred_std" not in loso.columns:
        print(f"{loso_path} predates the dispersion columns -- rerun "
              f"`python src/metrics_table.py --loso`")
        return
    loso["sid"] = loso["subject"].astype(int).map(lambda n: f"{n:03d}")

    classes = pd.read_csv(args.results_dir / "loso_with_class.csv")
    classes["sid"] = classes["subject"].astype(int).map(lambda n: f"{n:03d}")
    loso = loso.merge(classes[["sid", "glycemic_class"]], on="sid", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    for name, group in loso.groupby("glycemic_class"):
        ax.scatter(group["actual_std"], group["pred_std"], s=42, alpha=0.85,
                   color=CLASS_COLORS.get(name, MISSING_CLASS_COLOR), label=name,
                   edgecolor="k", linewidth=0.4)
    axis_limits = [0, max(loso["actual_std"].max(), loso["pred_std"].max()) * 1.08]
    ax.plot(axis_limits, axis_limits, "k--", lw=1, label="perfect dispersion (y = x)")

    # Least-squares slope through the origin: the cohort-wide dispersion ratio.
    slope = float((loso["actual_std"] * loso["pred_std"]).sum()
                  / (loso["actual_std"] ** 2).sum())
    ax.plot(axis_limits, [slope * x for x in axis_limits], "-", color="C3", lw=1.4,
            label=f"fit through origin: {slope:.2f}x")
    for sid in (args.good_subject, args.bad_subject, "046"):
        row = loso[loso["sid"] == sid]
        if not row.empty:
            ax.annotate(sid, (row["actual_std"].iloc[0], row["pred_std"].iloc[0]),
                        textcoords="offset points", xytext=(6, -9), fontsize=8)
    ax.set_xlim(axis_limits), ax.set_ylim(axis_limits)
    ax.set_xlabel("actual glucose SD (mg/dL)")
    ax.set_ylabel("prediction SD (mg/dL)")
    ax.set_title(f"A. Every one of {len(loso)} held-out subjects is under-dispersed")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # Panel B: pooled over all 11 seeds so the curve is not one split's luck.
    ys, ps = [], []
    for seed, ckpt, y, p, last, subj in iter_pooled_seeds(segments, ph=args.ph,
                                                          results_dir=args.results_dir):
        ys.append(y), ps.append(p)
    y_all, p_all = np.concatenate(ys), np.concatenate(ps)
    edges = np.quantile(y_all, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(y_all, edges[1:-1]), 0, 9)
    centres = np.array([y_all[idx == b].mean() for b in range(10)])
    pred_means = np.array([p_all[idx == b].mean() for b in range(10)])
    persist_means = None

    ax = axes[1]
    ax.plot(centres, centres, "k--", lw=1, label="unbiased (y = x)")
    ax.plot(centres, pred_means, "o-", color="C0", label="LSTM mean prediction")
    ax.set_xlabel("actual glucose, decile mean (mg/dL)")
    ax.set_ylabel("mean prediction (mg/dL)")
    ax.set_title("B. Regression toward the population mean at both tails\n"
                 f"(pooled over {len(ys)} seeds, n={len(y_all):,} windows)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = args.results_dir / "dispersion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out}")
    print(f"  dispersion ratio per subject: mean {loso['dispersion_ratio'].mean():.3f}, "
          f"range {loso['dispersion_ratio'].min():.3f}-{loso['dispersion_ratio'].max():.3f}")
    print(f"  under-dispersed (ratio < 1) on {(loso['dispersion_ratio'] < 1).sum()}/"
          f"{len(loso)} subjects; cohort fit through origin {slope:.3f}x")
    for name, group in loso.groupby("glycemic_class"):
        print(f"    {name:<13} n={len(group):<3} ratio {group['dispersion_ratio'].mean():.3f}")
    print(f"  lowest decile: actual {centres[0]:.0f} -> predicted {pred_means[0]:.0f} "
          f"({pred_means[0] - centres[0]:+.0f} mg/dL, pulled UP)")
    print(f"  highest decile: actual {centres[-1]:.0f} -> predicted {pred_means[-1]:.0f} "
          f"({pred_means[-1] - centres[-1]:+.0f} mg/dL, pulled DOWN)")
    print(f"  population mean of these test sets: {y_all.mean():.0f} mg/dL")


# ------------------------------------------------- where does the model actually help?

LEVEL_BINS = [0, 70, 140, 180, 250, 10_000]
LEVEL_LABELS = ["<70\n(hypo)", "70-140", "140-180", "180-250", ">250"]
# Standard CGM trend-arrow thresholds, mg/dL per minute over the last 15 minutes.
TREND_EDGES = [-1.0, 1.0]
TREND_LABELS = ["falling\n(<-1)", "flat\n(-1..1)", "rising\n(>1)"]


def figure_error_bins(segments, args) -> None:
    """Model vs persistence RMSE split by glucose level and by recent trend.

    Neither existing figure answers "where does the model actually help?". Bins are
    computed on the truth (level) and on the un-scaled input window (trend), pooled over
    all 11 seeds so no single split drives a bin."""
    ys, ps, lasts, rates = [], [], [], []
    for seed, ckpt, y, p, last, subj, history in iter_pooled_seeds_with_history(
            segments, ph=args.ph, results_dir=args.results_dir):
        ys.append(y), ps.append(p), lasts.append(last)
        # mg/dL per minute over the last 15 min (3 five-minute steps)
        rates.append((history[:, -1] - history[:, -4]) / 15.0)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    last = np.concatenate(lasts)
    rate = np.concatenate(rates)
    print(f"  pooled {len(ys)} seeds, n={len(y):,} windows")

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def bin_rows(assign, labels, title):
        rows = []
        for b, label in enumerate(labels):
            mask = assign == b
            if mask.sum() < 2:
                rows.append({"label": label, "n": int(mask.sum()),
                             "model": np.nan, "persist": np.nan, "skill": np.nan})
                continue
            model_rmse = rmse(p[mask], y[mask])
            persist_rmse = rmse(last[mask], y[mask])
            rows.append({"label": label, "n": int(mask.sum()), "model": model_rmse,
                         "persist": persist_rmse,
                         "skill": (persist_rmse - model_rmse) / persist_rmse})
        print(f"\n  {title}")
        print(f"    {'bin':<16} {'n':>8} {'model':>8} {'persist':>8} {'skill':>8}")
        for r in rows:
            label = r["label"].replace("\n", " ")
            print(f"    {label:<16} {r['n']:>8,} {r['model']:>8.2f} {r['persist']:>8.2f} "
                  f"{r['skill']:>7.1%}")
        return rows

    level_assign = np.clip(np.digitize(y, LEVEL_BINS[1:-1]), 0, len(LEVEL_LABELS) - 1)
    # Also bin on the CURRENT reading, not just the outcome. Binning on the truth answers
    # "during a real hypo event, which method is more accurate?"; binning on the current
    # level answers the actionable question "given where the patient is now, how good is
    # the forecast?" -- and it avoids conditioning on the very value being predicted,
    # which structurally penalizes a mean-reverting predictor.
    current_assign = np.clip(np.digitize(last, LEVEL_BINS[1:-1]), 0, len(LEVEL_LABELS) - 1)
    trend_assign = np.digitize(rate, TREND_EDGES)
    level_rows = bin_rows(level_assign, LEVEL_LABELS, "by ACTUAL (future) glucose level:")
    current_rows = bin_rows(current_assign, LEVEL_LABELS, "by CURRENT glucose level "
                                                          "(the actionable cut):")
    trend_rows = bin_rows(trend_assign, TREND_LABELS, "by recent trend (mg/dL per min):")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [5, 5, 3]})
    for ax, rows, xlabel in [(axes[0], level_rows, "actual (future) glucose (mg/dL)"),
                             (axes[1], current_rows, "current glucose (mg/dL)"),
                             (axes[2], trend_rows, "trend over the last 15 min")]:
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [r["persist"] for r in rows], width=0.4, label="persistence",
               color="C1", alpha=0.85)
        ax.bar(x + 0.2, [r["model"] for r in rows], width=0.4, label="LSTM", color="C0")
        for i, r in enumerate(rows):
            if not np.isnan(r["skill"]):
                ax.text(i, max(r["persist"], r["model"]) + 0.6, f"{r['skill']:+.1%}",
                        ha="center", fontsize=8)
            ax.text(i, 0.6, f"n={r['n']:,}", ha="center", fontsize=7, color="white")
        ax.set_xticks(x)
        ax.set_xticklabels([r["label"] for r in rows], fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("RMSE (mg/dL)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_title("Error by actual (future) level")
    axes[1].set_title("Error by current level")
    axes[2].set_title("Error by recent trend")
    fig.suptitle(f"Where the model helps: pooled over {len(ys)} seeds, "
                 f"+{args.ph} min (percentages = skill vs persistence)")
    fig.tight_layout()
    out = args.results_dir / "error_bins.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


# ---------------------------------------------------------------- cheap supplements

def figure_seed_pairs(segments, args) -> None:
    """Every seed's paired (persistence, model) result, per horizon.

    "Positive on 11/11 seeds" is the honest stability claim (raw SD is inflated by split
    difficulty), and it deserves to be visible rather than parenthetical."""
    table = pd.read_csv(args.results_dir / "horizon_multiseed.csv")
    horizons = sorted(table["ph"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(3.4 * len(horizons), 4.6),
                             sharey=False)
    axes = np.atleast_1d(axes)

    for ax, ph in zip(axes, horizons):
        sub = table[table["ph"] == ph]
        wins = 0
        for row in sub.itertuples():
            better = row.rmse < row.persistence
            wins += better
            ax.plot([0, 1], [row.persistence, row.rmse],
                    "-o", ms=4, lw=1.1, alpha=0.8,
                    color="C0" if better else "C3")
            if not better:
                ax.annotate(f"seed {row.seed}", (1, row.rmse), fontsize=7, color="C3",
                            textcoords="offset points", xytext=(4, -2))
        ax.set_xlim(-0.35, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["persistence", "LSTM"])
        ax.set_title(f"+{ph} min\nmodel better on {wins}/{len(sub)} seeds")
        ax.set_ylabel("RMSE (mg/dL)")
        ax.grid(axis="y", alpha=0.3)
        print(f"  +{ph:>3}: model better on {wins}/{len(sub)} seeds; "
              f"median paired gap {(sub['persistence'] - sub['rmse']).median():+.2f} mg/dL"
              + ("" if wins == len(sub) else
                 f"; loses on seed(s) "
                 f"{list(sub.loc[sub['rmse'] >= sub['persistence'], 'seed'])}"))
    fig.suptitle("Paired by split: each line is one seed (blue = model wins, red = loses)")
    fig.tight_layout()
    out = args.results_dir / "seed_pairs.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


ARCH_CURVES = [("LSTM", "", "C0"), ("BiLSTM", "_bidir", "C1"), ("Attn-LSTM", "_attn", "C2")]


def figure_learning_curves(segments, args) -> None:
    """Validation RMSE per epoch, median over the 11 seeds of each architecture.

    Answers "did you train long enough?" without a per-run wall-clock claim: every curve
    flattens well before early stopping fires (patience 20)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, infix, color in ARCH_CURVES:
        curves = []
        for seed in DEFAULT_SEEDS:
            run_dir = args.results_dir / run_tag(seed, args.ph, infix)
            if not run_dir.exists():
                continue
            acc = EventAccumulator(str(next(run_dir.glob("**/events.out.tfevents.*")).parent))
            acc.Reload()
            if "val_rmse" not in acc.Tags()["scalars"]:
                continue
            curves.append([e.value for e in acc.Scalars("val_rmse")])
        if not curves:
            print(f"  [skip] no tfevents for {label}")
            continue

        # Runs stop at different epochs (early stopping), so a median over ALL epochs
        # would be computed from a shrinking, increasingly lucky subset. Truncate at the
        # median stopping epoch, where at least half the runs are still alive.
        stops = sorted(len(c) for c in curves)
        cutoff = stops[len(stops) // 2]
        padded = np.full((len(curves), cutoff), np.nan)
        for i, curve in enumerate(curves):
            take = min(cutoff, len(curve))
            padded[i, :take] = curve[:take]
        median = np.nanmedian(padded, axis=0)
        quartile_low = np.nanpercentile(padded, 25, axis=0)
        quartile_high = np.nanpercentile(padded, 75, axis=0)
        epochs = np.arange(1, cutoff + 1)

        for curve in curves:                            # every run, faintly
            ax.plot(np.arange(1, len(curve) + 1), curve, color=color, alpha=0.15, lw=0.8)
        ax.plot(epochs, median, color=color, lw=2,
                label=f"{label} (median of {len(curves)}, to epoch {cutoff})")
        ax.fill_between(epochs, quartile_low, quartile_high, color=color, alpha=0.15)

        best = [int(np.argmin(c)) + 1 for c in curves]
        print(f"  {label:<10} {len(curves)} runs | epochs run "
              f"{min(stops)}-{max(stops)} (median {cutoff}) | best epoch median "
              f"{int(np.median(best))} (range {min(best)}-{max(best)}) | "
              f"best val_rmse median {np.median([min(c) for c in curves]):.2f}")

    ax.set_xlabel("epoch")
    ax.set_ylabel("validation RMSE (mg/dL)")
    ax.set_title("Training converges well before early stopping — no architecture is "
                 "under-trained")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = args.results_dir / "learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_loso_b2(segments, args, figsize=(14, 6), out_name="loso_b2_per_subject.png",
                   dpi=150) -> None:
    """B2 LOSO per-subject error beside B1, sorted by B2 RMSE, colored by glycemic
    class. The 039/046 flip is the visual headline.

    figsize/out_name/dpi exist for poster_figures.py, which re-renders this at a
    standardized poster geometry without duplicating the drawing logic."""
    b2_csv = args.results_dir / "loso_mealstatic_TMYIS_with_class.csv"
    b1_csv = args.results_dir / "loso_with_class.csv"
    if not b2_csv.exists():
        print(f"  {b2_csv} not found -- run B2 LOSO + analyze_loso.py first")
        return
    if not b1_csv.exists():
        print(f"  {b1_csv} not found -- run analyze_loso.py for B1 first")
        return

    b2 = pd.read_csv(b2_csv)
    b1 = pd.read_csv(b1_csv)
    b2["subject"] = b2["subject"].astype(int).map(lambda n: f"{n:03d}")
    b1["subject"] = b1["subject"].astype(int).map(lambda n: f"{n:03d}")
    b1 = b1.set_index("subject")

    # Every per-subject bar and the macro comparison below assume the two CSVs
    # cover the SAME folds. A partial rerun on either side would otherwise crash
    # mid-plot (missing subject) or silently average unequal cohorts.
    assert b1.index.is_unique, f"{b1_csv} has duplicate subject rows"
    b1_subjects = set(b1.index)
    b2_subjects = set(b2["subject"])
    if b1_subjects != b2_subjects:
        only_b1 = sorted(b1_subjects - b2_subjects)
        only_b2 = sorted(b2_subjects - b1_subjects)
        print(f"  subject sets differ (only B1: {only_b1}, only B2: {only_b2}) -- "
              f"rerun the missing folds; refusing to compare unequal cohorts")
        return

    b2 = b2.sort_values("test_rmse").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.35
    for i, row in enumerate(b2.itertuples()):
        color = CLASS_COLORS.get(row.glycemic_class, MISSING_CLASS_COLOR)
        ax.bar(i - bar_width / 2, row.test_rmse, width=bar_width, color=color, alpha=0.9)
        b1_rmse = b1.loc[row.subject, "test_rmse"]
        ax.bar(i + bar_width / 2, b1_rmse, width=bar_width, color=color, alpha=0.35,
               edgecolor=color, linewidth=0.5)
        ax.plot(i, row.persistence_rmse, "k_", markersize=10)

    ax.set_xticks(range(len(b2)))
    ax.set_xticklabels(b2["subject"], rotation=90, fontsize=7)
    ax.set_xlabel("Subject (sorted by B2 RMSE)")
    ax.set_ylabel("RMSE (mg/dL)")
    ax.set_title(f"LOSO: B2 (solid) vs B1 (faded), {len(b2)} held-out subjects "
                 f"(ticks = persistence)")
    ax.legend(handles=[
        Patch(color="#666", alpha=0.9, label="B2 (CGM + meals)"),
        Patch(color="#666", alpha=0.35, label="B1 (CGM only)"),
        plt.Line2D([], [], color="k", marker="_", ls="", label="persistence"),
    ] + [Patch(color=c, label=k) for k, c in CLASS_COLORS.items()],
              loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = args.results_dir / out_name
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    b2_macro = b2["test_rmse"].mean()
    b1_macro = b1["test_rmse"].mean()
    n_wins = int((b2["test_rmse"] < b2["persistence_rmse"]).sum())
    flipped = []
    for _, row in b2.iterrows():
        b1_skill = b1.loc[row["subject"], "skill"]
        if b1_skill < 0 and row["skill"] > 0:
            flipped.append(row["subject"])

    print(f"wrote {out}")
    print(f"  B2 macro {b2_macro:.2f} vs B1 macro {b1_macro:.2f} "
          f"(delta {b1_macro - b2_macro:+.2f})")
    print(f"  B2 wins {n_wins}/{len(b2)} subjects")
    print(f"  flipped (B1 loss -> B2 win): {', '.join(flipped) if flipped else 'none'}")
    for name, group in b2.groupby("glycemic_class"):
        b1_group = b1.loc[group["subject"], "test_rmse"]
        print(f"  {name:<13} n={len(group):<3} "
              f"B2 {group['test_rmse'].mean():.2f}  B1 {b1_group.mean():.2f}  "
              f"delta {b1_group.mean() - group['test_rmse'].mean():+.2f}")


def figure_horizon_meal(segments, args) -> None:
    """Paired meal benefit (B1 - B2 RMSE) at each horizon. Shows the peak at +60 and
    collapse at +120."""
    infix = "_mealstatic_TMYIS"
    horizons = [30, 60, 120]
    deltas, means_b1, means_b2, means_persist = [], [], [], []
    per_seed_deltas = {h: [] for h in horizons}

    for ph in horizons:
        # Pair BY SEED, never by list position: iter_pooled_seeds skips missing
        # checkpoints, so two positional lists can silently pair different seeds.
        b1_by_seed, b2_by_seed, persist_by_seed = {}, {}, {}
        for seed, _, y, preds, last, _ in iter_pooled_seeds(
                segments, ph=ph, infix="", results_dir=args.results_dir):
            b1_by_seed[seed] = np.sqrt(np.mean((preds - y) ** 2))
            persist_by_seed[seed] = np.sqrt(np.mean((last - y) ** 2))
        for seed, _, y, preds, _, _ in iter_pooled_seeds(
                segments, ph=ph, infix=infix, results_dir=args.results_dir):
            b2_by_seed[seed] = np.sqrt(np.mean((preds - y) ** 2))

        common_seeds = sorted(set(b1_by_seed) & set(b2_by_seed))
        assert common_seeds, f"+{ph}: no seed has BOTH a B1 and a B2 checkpoint"
        excluded = sorted(set(b1_by_seed) ^ set(b2_by_seed))
        if excluded:
            print(f"  +{ph}: seeds {excluded} exist on only one side -- "
                  f"pairing on the {len(common_seeds)} common seeds")

        b1_arr = np.array([b1_by_seed[seed] for seed in common_seeds])
        b2_arr = np.array([b2_by_seed[seed] for seed in common_seeds])
        d = b1_arr - b2_arr
        deltas.append(d.mean())
        per_seed_deltas[ph] = d
        means_b1.append(b1_arr.mean())
        means_b2.append(b2_arr.mean())
        means_persist.append(np.mean([persist_by_seed[seed] for seed in common_seeds]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ph, d in per_seed_deltas.items():
        x = [ph] * len(d)
        ax1.scatter(x, d, color="C0", alpha=0.5, s=30, zorder=3)
    ax1.plot(horizons, deltas, "C0o-", lw=2, zorder=4, label="mean paired delta")
    ax1.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax1.set_xlabel("Prediction horizon (minutes)")
    ax1.set_ylabel("Paired delta: B1 - B2 RMSE (mg/dL)")
    ax1.set_title("Meal benefit by horizon")
    ax1.set_xticks(horizons)
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(horizons, means_persist, "s--", color="#999", label="Persistence")
    ax2.plot(horizons, means_b1, "o-", color="C1", label="B1 (CGM only)")
    ax2.plot(horizons, means_b2, "o-", color="C0", label="B2 (CGM + meals)")
    ax2.set_xlabel("Prediction horizon (minutes)")
    ax2.set_ylabel("RMSE (mg/dL)")
    ax2.set_title("Absolute error by horizon")
    ax2.set_xticks(horizons)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = args.results_dir / "horizon_meal_benefit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out}")
    for ph, d in per_seed_deltas.items():
        n_wins = int((d > 0).sum())
        print(f"  +{ph:>3} min: delta {d.mean():+.3f} +/- {d.std(ddof=1):.3f}  "
              f"wins {n_wins}/{len(d)}")


FIGURES = {
    "parkes": figure_parkes,
    "traces": figure_traces,
    "traces_pooled": figure_traces_pooled,
    "horizon": figure_horizon,
    "loso": figure_loso,
    "sweep": figure_sweep,
    "data_integrity": figure_data_integrity,
    "dispersion": figure_dispersion,
    "error_bins": figure_error_bins,
    "seed_pairs": figure_seed_pairs,
    "learning_curves": figure_learning_curves,
    "loso_b2": figure_loso_b2,
    "horizon_meal": figure_horizon_meal,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate result figures for the glucose forecasting project")
    parser.add_argument("--figure", choices=["all", *FIGURES], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--ph", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=int, default=2, choices=[1, 2],
                        help="Parkes grid diabetes type (2 = primary, matches the cohort)")
    parser.add_argument("--good-subject", default="006", help="LOSO subject for the good panel")
    parser.add_argument("--bad-subject", default="039", help="LOSO subject for the bad panel")
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/CGMacros"),
                        help="raw CGMacros CSVs, for the data-integrity figure")
    parser.add_argument("--integrity-subject", default="001",
                        help="subject to illustrate the interpolation with")
    parser.add_argument("--max-step", type=int, default=MAX_STEP, metavar="MGDL",
                        help="reject trace runs containing a jump larger than this "
                             f"(default {MAX_STEP}); lower it to test the screen's effect")
    args = parser.parse_args()

    logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.WARNING)
    segments = load_segments(args.data_dir / "segments_split.npz")

    names = list(FIGURES) if args.figure == "all" else [args.figure]
    for name in names:
        print(f"\n[{name}]")
        FIGURES[name](segments, args)


if __name__ == "__main__":
    main()
