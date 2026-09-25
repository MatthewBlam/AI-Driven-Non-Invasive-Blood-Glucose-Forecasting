"""Post-meal glucose response, grouped by carbohydrate load.

The poster's "what does the meal's size do?" figure (guide/15-poster.md section
7b): mean glucose change from meal time over the following two hours, one curve
per carb bin, shaded +/- one standard error. Purely descriptive -- built from the
meal log and the corrected 5-minute CGM directly, no model involved.

Every meal is aligned at its own baseline (the reading at or just before the
logged meal time; within a segment that reading is at most 5 minutes old), so the
curves answer: "given that a meal of THIS size was just eaten, what does glucose
do next, on average?"

Meals are excluded, and counted, when the carbs are unknown (the 43 all-zero
"macros missing" log rows), when the CGM does not cover the meal or the full two
hours after it (segment gaps), or -- by default -- when another meal follows
within the two-hour horizon (the second meal would contaminate the first meal's
curve; keep them with --keep-overlaps to see the difference).

Usage (from the repo root):
    python src/plot_carb_response.py                 # -> results/carb_response.png
    python src/plot_carb_response.py --keep-overlaps

Renders through the Agg backend: files are written, no window opens.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")           # must precede the pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from meals import PHYSIOLOGICAL_CAPS, READING_INTERVAL_MIN, load_meals, \
    meals_by_subject, segment_times
from preprocess import Segment, load_segments

RESULTS_DIR = Path("results")

# Bin edges in grams of carbohydrate. The same cut points PROJECT_STATE's
# "signal shape" measurement used, so the two readings of the data stay
# comparable: 0-15 / 15-40 / 40-70 / 70+.
CARB_EDGES = (15.0, 40.0, 70.0)
BIN_LABELS = ("0-15 g", "15-40 g", "40-70 g", "70+ g")

FOLLOW_MIN = 120                                     # minutes of curve after the meal
FOLLOW_STEPS = FOLLOW_MIN // READING_INTERVAL_MIN    # 24 five-minute steps

# The bins are ORDERED (a magnitude, not four unrelated identities), so the ramp
# must still read light -> dark with the carb load -- but the professor's poster
# review (2026-09-22) asked for clearly DIFFERENT colors, which the single-hue
# Blues ramp lacked. Viridis samples give four distinct hues (yellow-green ->
# teal -> blue -> purple) that stay ordered and colorblind-safe; reversed so the
# biggest load draws the darkest, most prominent curve. C0..C3 stays off the
# table: blue/orange/green are the baseline identity colors (B1/B2/B3).
BIN_COLORS = plt.get_cmap("viridis")(np.linspace(0.85, 0.05, len(BIN_LABELS)))

EXCLUSION_REASONS = ("macros missing", "no CGM at meal time",
                     "curve leaves the segment", "another meal within 2 h")


@dataclass(frozen=True)
class MealTrace:
    """One meal's aligned response curve."""

    subject: str
    segment_index: int
    baseline_index: int          # reading index of the baseline within the segment
    carbs: float
    bin_index: int
    deltas: np.ndarray           # (FOLLOW_STEPS + 1,) glucose minus baseline, mg/dL


def collect_traces(segments, meals: pd.DataFrame, keep_overlaps: bool = False):
    """Align every usable meal to its segment's reading grid. Returns
    (traces, exclusion counts). Every meal lands in exactly one bucket --
    used or one exclusion reason -- and the caller asserts the partition.

    Exclusions are tested in a fixed order (macros -> coverage -> horizon ->
    overlap) so each count has one meaning: "another meal within 2 h" counts only
    meals that a curve could otherwise have been drawn for.
    """
    segments_by_subject: dict[str, list[tuple[int, Segment, np.ndarray]]] = {}
    for segment_index, segment in enumerate(segments):
        segments_by_subject.setdefault(segment.subject, []).append(
            (segment_index, segment, segment_times(segment)))

    traces: list[MealTrace] = []
    excluded = dict.fromkeys(EXCLUSION_REASONS, 0)

    for subject, subject_meals in meals_by_subject(meals).items():
        meal_timestamps = subject_meals["timestamp"].to_numpy()
        for row in range(len(subject_meals)):
            meal = subject_meals.iloc[row]
            if bool(meal["macros_missing"]):
                excluded["macros missing"] += 1
                continue

            meal_time = meal_timestamps[row]
            covering = [(segment_index, segment, times)
                        for segment_index, segment, times
                        in segments_by_subject.get(subject, [])
                        if times[0] <= meal_time <= times[-1]]
            if not covering:
                excluded["no CGM at meal time"] += 1
                continue
            # Segments never overlap in time (they are gap-separated runs of one
            # subject's readings), so a meal falls inside at most one.
            assert len(covering) == 1, f"meal at {meal_time} inside two segments"
            segment_index, segment, times = covering[0]

            baseline = int(np.searchsorted(times, meal_time, side="right") - 1)
            # Inside a segment, readings are 5 min apart, so the reading just
            # before the meal is necessarily fresh. State it rather than trust it.
            lag_min = float((meal_time - times[baseline]) / np.timedelta64(1, "m"))
            assert 0.0 <= lag_min < READING_INTERVAL_MIN, \
                f"stale baseline ({lag_min:.1f} min) for meal at {meal_time}"

            if baseline + FOLLOW_STEPS >= len(segment.glucose):
                excluded["curve leaves the segment"] += 1
                continue

            next_row = row + 1
            if (not keep_overlaps and next_row < len(subject_meals)
                    and (meal_timestamps[next_row] - meal_time)
                    <= np.timedelta64(FOLLOW_MIN, "m")):
                excluded["another meal within 2 h"] += 1
                continue

            curve = segment.glucose[baseline: baseline + FOLLOW_STEPS + 1]
            carbs = float(meal["Carbs"])
            traces.append(MealTrace(
                subject=subject, segment_index=segment_index,
                baseline_index=baseline, carbs=carbs,
                bin_index=int(np.digitize(carbs, CARB_EDGES)),
                deltas=curve - curve[0]))

    return traces, excluded


def verify_traces(traces, segments, n_sample: int = 25) -> int:
    """Recompute a sample of curves through an independent second path (pandas
    label slicing on a time-indexed series, against collect_traces' integer
    indexing) and assert bit-identical deltas. Returns curves checked.

    Same discipline as activity.verify_aggregation: two code paths that agree
    pin the alignment, and the negative check in main() proves this can fail.
    """
    assert traces, "verified an empty set of traces -- that proves nothing"
    stride = max(1, len(traces) // n_sample)
    sample = traces[::stride][:n_sample]

    for trace in sample:
        segment = segments[trace.segment_index]
        series = pd.Series(segment.glucose,
                           index=pd.DatetimeIndex(segment_times(segment)))
        start = series.index[trace.baseline_index]
        window = series.loc[start: start + pd.Timedelta(minutes=FOLLOW_MIN)]
        assert len(window) == FOLLOW_STEPS + 1, \
            f"label slice found {len(window)} readings, expected {FOLLOW_STEPS + 1}"
        expected = window.to_numpy() - window.iloc[0]
        assert np.array_equal(expected, trace.deltas), \
            f"deltas disagree for subject {trace.subject} at {start}"
    return len(sample)


def bin_curves(traces):
    """Per-bin mean curve, standard error, and n. Returns a list aligned with
    BIN_LABELS; a bin with fewer than 2 meals yields (None, None, n)."""
    results = []
    for bin_index in range(len(BIN_LABELS)):
        stacked = np.array([trace.deltas for trace in traces
                            if trace.bin_index == bin_index])
        if len(stacked) < 2:
            results.append((None, None, len(stacked)))
            continue
        mean = stacked.mean(axis=0)
        sem = stacked.std(axis=0, ddof=1) / np.sqrt(len(stacked))
        results.append((mean, sem, len(stacked)))
    return results


def draw_figure(curves, out_path: Path, dpi: int = 300,
                figsize: tuple[float, float] = (9.0, 5.5)) -> None:
    offsets = np.arange(FOLLOW_STEPS + 1) * READING_INTERVAL_MIN

    fig, ax = plt.subplots(figsize=figsize)
    for (mean, sem, n), label, color in zip(curves, BIN_LABELS, BIN_COLORS):
        if mean is None:
            continue
        ax.plot(offsets, mean, color=color, lw=2, label=f"{label}  (n={n})")
        ax.fill_between(offsets, mean - sem, mean + sem, color=color,
                        alpha=0.18, lw=0)
        ax.annotate(label, (offsets[-1], mean[-1]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=color)

    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("minutes after the meal")
    ax.set_ylabel("glucose change from meal time (mg/dL)")
    ax.set_title("Post-meal glucose response by carbohydrate load\n"
                 "(mean over meals, shaded +/- 1 standard error)")
    ax.legend(title="carbs in the meal", fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, FOLLOW_MIN + 14)          # room for the direct labels
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poster figure: post-meal glucose response by carb load")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/CGMacros"),
                        help="raw CGMacros CSVs (the meal log)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=None,
                        help="output PNG (default results/carb_response.png)")
    parser.add_argument("--keep-overlaps", action="store_true",
                        help="keep meals that are followed by another meal within "
                             "2 h (contaminates the tail of the curve)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="output resolution (default 300 -- poster print "
                             "quality; the shared paper figures use 150)")
    parser.add_argument("--figsize", type=float, nargs=2, default=(9.0, 5.5),
                        metavar=("W", "H"),
                        help="figure size in inches (poster_figures.py passes "
                             "the standardized 10-inch-wide geometry)")
    args = parser.parse_args()
    out_path = args.out or args.results_dir / "carb_response.png"

    segments = load_segments(args.data_dir / "segments_split.npz")
    meals = load_meals(args.raw_dir)

    traces, excluded = collect_traces(segments, meals,
                                      keep_overlaps=args.keep_overlaps)
    assert len(traces) + sum(excluded.values()) == len(meals), \
        "used + excluded does not partition the meal table"

    n_checked = verify_traces(traces, segments)
    print(f"verified: {n_checked} sampled curves match a label-sliced recompute "
          f"(independent second path)")

    # The check must be shown able to fail (same discipline as
    # activity.verify_aggregation): a corrupted curve must trip it.
    first = traces[0]
    corrupted = MealTrace(subject=first.subject, segment_index=first.segment_index,
                          baseline_index=first.baseline_index, carbs=first.carbs,
                          bin_index=first.bin_index, deltas=first.deltas + 1.0)
    try:
        verify_traces([corrupted], segments, n_sample=1)
        raise SystemExit("verify_traces PASSED a corrupted curve")
    except AssertionError:
        print("negative check: a corrupted curve trips the verification")

    print(f"\n{len(meals):,} logged meals -> {len(traces):,} curves")
    for reason in EXCLUSION_REASONS:
        print(f"  excluded, {reason}: {excluded[reason]}")
    over_cap = sum(trace.carbs > PHYSIOLOGICAL_CAPS["Carbs"] for trace in traces)
    if over_cap:
        print(f"  note: {over_cap} meals log more than "
              f"{PHYSIOLOGICAL_CAPS['Carbs']:.0f} g carbs (known corrupt rows); "
              f"they can only land in the top bin, so they were kept")

    curves = bin_curves(traces)
    draw_figure(curves, out_path, dpi=args.dpi, figsize=tuple(args.figsize))

    print(f"\nwrote {out_path}")
    print(f"  {'bin':<9} {'n':>5} {'+30 min':>9} {'+60 min':>9} "
          f"{'peak':>7} {'at':>6}")
    step_30 = 30 // READING_INTERVAL_MIN
    step_60 = 60 // READING_INTERVAL_MIN
    for (mean, sem, n), label in zip(curves, BIN_LABELS):
        if mean is None:
            print(f"  {label:<9} {n:>5}   too few meals to average")
            continue
        peak_step = int(np.argmax(mean))
        print(f"  {label:<9} {n:>5} {mean[step_30]:>+8.1f} {mean[step_60]:>+8.1f} "
              f"{mean[peak_step]:>+6.1f} {peak_step * READING_INTERVAL_MIN:>4d}m")
    print("  ^ meal-ALIGNED curves: not directly comparable to the window-level "
        "30-min deltas\n    in PROJECT_STATE (those condition on 'a meal in the "
        "prior 60 min', not on t=0).")


if __name__ == "__main__":
    main()
