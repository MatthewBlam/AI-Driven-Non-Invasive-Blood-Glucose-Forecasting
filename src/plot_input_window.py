"""One real hour of input, drawn as each baseline sees it.

Three stacked rows over the same 60 minutes -- the poster's "what each model sees"
figure (guide/15-poster.md section 7a), doubling as a visual definition of the
baselines:

    row 1   B1: the 12 glucose readings (60 min of history), the +30 min target starred
    row 2   B2: + the most recent meal, at its clock position, with its logged macros
    row 3   B3: + heart rate (and activity calories) over the same hour

The window is REAL and chosen deterministically: among all forecast-shaped windows
with a recent meal (15-30 min before "now", macros actually logged), full heart-rate
coverage, and a visible post-meal rise, take the strongest rise. `--rank` steps to
the runners-up; `--list` prints the shortlist instead of plotting. Nothing is
smoothed or invented -- the meal card shows the dietitian's logged numbers.

The right-edge verdict labels are derived from results/metrics_table.csv (mean
pooled RMSE per arm, deltas paired by seed), never hand-typed, so the figure and
the results table cannot drift apart.

Usage (from the repo root):
    python src/plot_input_window.py                  # -> results/input_window.png
    python src/plot_input_window.py --rank 1         # the runner-up window
    python src/plot_input_window.py --subject 029    # restrict to one subject
    python src/plot_input_window.py --list 10        # shortlist only, no figure

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
from numpy.lib.stride_tricks import sliding_window_view

from activity import SegmentActivity, aggregate_segment, load_activity, verify_aggregation
from meals import READING_INTERVAL_MIN, load_meals, meals_by_subject, segment_times, \
    verify_segment_times
from preprocess import HISTORY, Segment, load_segments

RESULTS_DIR = Path("results")

HORIZON_STEPS = 6               # +30 min at 5-min spacing, the headline horizon

# Selection defaults. The meal-age band matches the poster spec (a meal recent
# enough that its effect is still ahead of the model); the rise floor keeps the
# starred target visually interesting. All three are CLI-overridable.
MEAL_AGE_MIN, MEAL_AGE_MAX = 15.0, 30.0     # minutes before "now"
MIN_RISE = 15.0                              # mg/dL, now -> target

# Same physiology screen as figures.py's trace picker: glucose moves at most
# ~4-5 mg/dL per minute, so a jump beyond this between two readings 5 min apart
# is a sensor artifact, not physiology -- and ranking by rise would otherwise
# select exactly those artifacts (the unscreened top "rise" here was 219 mg/dL
# in 30 min).
MAX_STEP = 25                                # mg/dL between consecutive readings

# Fixed row colors, matching the identities the project's figures already use:
# C0 = the glucose signal / the model, C1 = meals, C2 = activity, C3 = highlight.
GLUCOSE_COLOR, MEAL_COLOR, HR_COLOR, TARGET_COLOR = "C0", "C1", "C2", "C3"


@dataclass(frozen=True)
class Candidate:
    """One forecast-shaped window that passed the glucose + meal criteria."""

    segment_index: int
    now_index: int              # index of the newest input reading in the segment
    rise: float                 # glucose[target] - glucose[now], mg/dL
    meal_row: int               # row in the subject's own meal frame
    minutes_since_meal: float


def find_candidates(segments, per_subject_meals, history: int = HISTORY,
                    horizon_steps: int = HORIZON_STEPS,
                    meal_age: tuple[float, float] = (MEAL_AGE_MIN, MEAL_AGE_MAX),
                    min_rise: float = MIN_RISE, max_step: float = MAX_STEP,
                    subject: str | None = None):
    """Every window satisfying the glucose and meal criteria, strongest rise first.

    Heart-rate coverage is NOT checked here -- it needs the per-minute activity
    files, so the caller screens the (much shorter) sorted list lazily instead of
    aggregating all 81 segments up front.

    The meal criterion requires the macros to be genuinely logged
    (`macros_missing == False`): the row-2 card displays those numbers, and a
    train-median fill on a poster would be an invented meal.
    """
    candidates = []
    span = history + horizon_steps          # readings drawn: i-history+1 .. i+horizon
    for segment_index, segment in enumerate(segments):
        if subject is not None and segment.subject != subject:
            continue
        n = len(segment.glucose)
        if n < span:
            continue
        subject_meals = per_subject_meals.get(segment.subject)
        if subject_meals is None:
            continue

        times = segment_times(segment)
        meal_times = subject_meals["timestamp"].to_numpy()
        macros_missing = subject_meals["macros_missing"].to_numpy()

        now_indices = np.arange(history - 1, n - horizon_steps)
        rise = segment.glucose[now_indices + horizon_steps] - segment.glucose[now_indices]

        # Largest consecutive-reading jump anywhere in the drawn span (history AND
        # the actual-future overlay), aligned so row j covers now_indices[j].
        steps = np.abs(np.diff(segment.glucose))
        biggest_step = sliding_window_view(steps, span - 1).max(axis=1)
        assert len(biggest_step) == len(now_indices)

        # Most recent meal at or before each "now" -- the same backward-only lookup
        # rule the B2 features use (side="right" then -1: nothing after now).
        meal_rows = np.searchsorted(meal_times, times[now_indices], side="right") - 1
        has_meal = meal_rows >= 0
        minutes_since = np.full(len(now_indices), np.inf)
        minutes_since[has_meal] = (
            (times[now_indices][has_meal] - meal_times[meal_rows[has_meal]])
            / np.timedelta64(1, "m"))

        macros_known = np.zeros(len(now_indices), dtype=bool)
        macros_known[has_meal] = ~macros_missing[meal_rows[has_meal]]

        eligible = ((minutes_since >= meal_age[0]) & (minutes_since <= meal_age[1])
                    & (rise >= min_rise) & macros_known & (biggest_step <= max_step))
        for j in np.flatnonzero(eligible):
            candidates.append(Candidate(segment_index, int(now_indices[j]),
                                        float(rise[j]), int(meal_rows[j]),
                                        float(minutes_since[j])))

    # Deterministic order: strongest rise first, remaining keys only to break ties,
    # so the same data always yields the same figure.
    candidates.sort(key=lambda c: (-c.rise, segments[c.segment_index].subject,
                                   c.segment_index, c.now_index))
    return candidates


def hr_window_complete(aggregate: SegmentActivity, now_index: int,
                       history: int = HISTORY) -> bool:
    """True when every reading in the input window has a heart-rate value."""
    window = aggregate.hr[now_index - history + 1: now_index + 1]
    return not np.isnan(window).any()


def screen_candidates(candidates, segments, activity_frames, n_needed: int,
                      history: int = HISTORY):
    """Walk the sorted candidates, keeping those with full HR coverage, until
    n_needed survive. Aggregates each segment at most once (cached), so only the
    handful of segments near the top of the list pay the aggregation cost."""
    aggregates: dict[int, SegmentActivity] = {}
    survivors = []
    for candidate in candidates:
        if candidate.segment_index not in aggregates:
            segment = segments[candidate.segment_index]
            aggregates[candidate.segment_index] = aggregate_segment(
                segment, activity_frames[segment.subject])
        if hr_window_complete(aggregates[candidate.segment_index],
                              candidate.now_index, history):
            survivors.append(candidate)
            if len(survivors) >= n_needed:
                break
    return survivors, aggregates


def verdict_labels(metrics_csv: Path):
    """The three right-edge labels, derived from the results table.

    Pairs the per-seed pooled RMSEs (B1 vs B2, B2 vs B3) on their common seeds --
    the project's pairing rule -- and prints the numbers it derived. Returns
    (labels, None-safe) with an empty string where an arm has no rows, or None
    when the CSV itself is absent (the figure then ships without verdicts).
    """
    if not metrics_csv.exists():
        print(f"  {metrics_csv} not found -- omitting the verdict labels "
              f"(run `python src/metrics_table.py` first to add them)")
        return None

    table = pd.read_csv(metrics_csv)
    pooled = table[(table["split"] == "pooled") & (table["ph"] == 30)]
    rmse_by_seed = {arm: group.set_index("seed")["rmse"]
                    for arm, group in pooled.groupby("architecture")}
    b1 = rmse_by_seed.get("LSTM")
    b2 = rmse_by_seed.get("B2-LSTM")
    b3 = rmse_by_seed.get("B3-LSTM")
    if b1 is None:
        print(f"  {metrics_csv} has no pooled LSTM rows -- omitting the verdict labels")
        return None

    labels = [f"pooled RMSE\n{b1.mean():.2f} mg/dL", "", ""]
    print(f"  verdicts (from {metrics_csv}):")
    print(f"    B1 pooled RMSE {b1.mean():.2f} mg/dL over {len(b1)} seeds")

    if b2 is not None:
        common = b1.index.intersection(b2.index)
        delta = (b1[common] - b2[common])            # positive = meals help
        wins = int((delta > 0).sum())
        labels[1] = (f"{b2.mean():.2f} mg/dL\nmeals help\n({delta.mean():.2f} better, "
                     f"{wins}/{len(delta)} seeds)")
        print(f"    B2 pooled RMSE {b2.mean():.2f}; paired delta vs B1 "
              f"{delta.mean():+.3f} mg/dL, better on {wins}/{len(delta)} seeds")
    if b3 is not None and b2 is not None:
        common = b2.index.intersection(b3.index)
        delta = (b2[common] - b3[common])            # positive = activity helps
        labels[2] = (f"{b3.mean():.2f} mg/dL\nactivity adds nothing\n"
                     f"({abs(delta.mean()):.2f}, a null)")
        print(f"    B3 pooled RMSE {b3.mean():.2f}; paired delta vs B2 "
              f"{delta.mean():+.3f} mg/dL (a null)")
    return labels


def draw_figure(segment: Segment, aggregate: SegmentActivity, candidate: Candidate,
                meal: pd.Series, labels, out_path: Path, dpi: int = 300,
                figsize: tuple[float, float] = (9.0, 7.5),
                history: int = HISTORY, horizon_steps: int = HORIZON_STEPS):
    """Render the three rows and save the figure. Returns caption numbers."""
    i = candidate.now_index
    glucose = segment.glucose
    times = segment_times(segment)
    now_time = pd.Timestamp(times[i])

    offsets = (np.arange(history) - (history - 1)) * READING_INTERVAL_MIN   # -55..0
    history_window = glucose[i - history + 1: i + 1]
    future_offsets = np.arange(horizon_steps + 1) * READING_INTERVAL_MIN    # 0..+30
    future = glucose[i: i + horizon_steps + 1]
    hr_window = aggregate.hr[i - history + 1: i + 1]
    calorie_window = aggregate.calories[i - history + 1: i + 1]

    fig, (ax_glucose, ax_meal, ax_hr) = plt.subplots(
        3, 1, sharex=True, figsize=figsize,
        gridspec_kw={"height_ratios": [3.0, 1.15, 2.0]})
    # Tighter chrome than the defaults (2026-09-24): pull the axes up toward
    # the suptitle (top 0.88 left a ~0.6 in gap under the title) and out to the
    # right edge (right 0.9 reserved a strip the optional verdict labels no
    # longer need -- with labels they draw past the spine and bbox_inches
    # still includes them).
    fig.subplots_adjust(top=0.92, right=0.97)

    # Row 1 -- B1: the glucose window and the starred target.
    ax_glucose.plot(offsets, history_window, "o-", color=GLUCOSE_COLOR, lw=2, ms=5,
                    label=f"glucose input ({history} readings, 60 min)")
    # Marked at every reading (smaller than the input dots) so it reads as six real
    # 5-minute measurements, not a drawn straight line. markevery skips the point at
    # 0: that reading is already the last blue input dot, and a grey marker on top
    # of it would double-count "now".
    ax_glucose.plot(future_offsets, future, ":", marker="o", ms=3.5, color="0.55",
                    lw=1.4, markevery=slice(1, None),
                    label="what actually happened (5-min readings)")
    # ms kept small, plus extra y-headroom below: the target sits at the
    # window's maximum, and the default 5% y-margin left the star's tip
    # poking past the top frame (ms=17 was visibly cut off, 2026-09-24).
    ax_glucose.plot(horizon_steps * READING_INTERVAL_MIN, future[-1], marker="*",
                    ms=12, color=TARGET_COLOR, ls="none", zorder=5,
                    label="target (+30 min)")
    y_lo, y_hi = ax_glucose.get_ylim()
    ax_glucose.set_ylim(y_lo, y_hi + 0.05 * (y_hi - y_lo))
    ax_glucose.set_ylabel("glucose (mg/dL)")
    ax_glucose.legend(loc="upper left", fontsize=8)
    ax_glucose.set_title("B1 -- CGM only", loc="left", fontsize=10)
    ax_glucose.annotate("now", xy=(0, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color="0.3")

    # Row 2 -- B2: the meal event and its logged summary.
    minutes_ago = candidate.minutes_since_meal
    # Timeline height: leaves breathing room above the what-was-entered note below.
    meal_y = 0.5
    ax_meal.plot([offsets[0], future_offsets[-1]], [meal_y, meal_y], color="0.85",
                 lw=1, zorder=1)
    # mew: an explicit drawn edge, so the outline antialiases smoothly. Without it
    # the flat top edge can rasterize one pixel higher on one side (the marker
    # center lands on a fractional pixel), which reads as a slight tilt.
    ax_meal.plot(-minutes_ago, meal_y, marker="v", ms=12, color=MEAL_COLOR,
                 mec=MEAL_COLOR, mew=1.5, zorder=3)
    card = (f"{meal['type']}: {meal['Carbs']:.0f} g carbs, "
            f"{meal['Protein']:.0f} g protein, {meal['Fat']:.0f} g fat\n"
            f"eaten {minutes_ago:.0f} min before now")
    ax_meal.text(-minutes_ago + 4, meal_y, card, va="center", ha="left", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=MEAL_COLOR))
    # Sized to end left of the "now" line (axes fraction ~0.63) -- the future
    # region stays clear for the card.
    ax_meal.text(0.01, 0.02, "entered as a static summary (timing, macronutrients, "
                             "meal type, multi-meal overlap)",
                 transform=ax_meal.transAxes, fontsize=7, color="0.45", va="bottom")
    ax_meal.set_ylim(0, 1)
    ax_meal.set_yticks([])
    ax_meal.set_title("B2 -- + meals", loc="left", fontsize=10)

    # Row 3 -- B3: heart rate over the same hour.
    ax_hr.plot(offsets, hr_window, ".-", color=HR_COLOR, lw=1.8, ms=6,
               label="heart rate (5-min mean)")
    ax_hr.set_ylabel("heart rate (bpm)")
    ax_hr.set_xlabel("minutes relative to prediction time")
    ax_hr.set_title("B3 -- + physiology (HR, activity)", loc="left", fontsize=10)
    # Clear a band under the HR trace for the what-was-entered note (the trace
    # runs close to the axis floor on the left of this window).
    y_low, y_high = ax_hr.get_ylim()
    ax_hr.set_ylim(y_low - 0.16 * (y_high - y_low), y_high)
    ax_hr.text(0.01, 0.02, "entered as a static summary (HR, HR vs own norm, "
                           "time since active, energy, wear gaps)",
               transform=ax_hr.transAxes, fontsize=7, color="0.45", va="bottom")
    mean_calories = (float(np.nanmean(calorie_window))
                     if not np.isnan(calorie_window).all() else None)
    hr_card = f"mean HR {hr_window.mean():.0f} bpm"
    if mean_calories is not None:
        hr_card += f", activity {mean_calories:.2f} kcal/min"
    # Centered between "now" (x=0) and the right edge of the axis; x in data
    # coordinates, y in axes fraction (get_xaxis_transform blends the two).
    hr_card_x = (future_offsets[-1] + 4) / 2      # xlim right edge is future+4
    ax_hr.text(hr_card_x, 0.06, hr_card, transform=ax_hr.get_xaxis_transform(),
               fontsize=8.5, va="bottom", ha="center",
               bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=HR_COLOR))

    for ax in (ax_glucose, ax_meal, ax_hr):
        ax.axvline(0, color="k", ls="--", lw=1, alpha=0.55, zorder=2)
        ax.grid(alpha=0.3)
    ax_meal.grid(False)
    ax_glucose.set_xlim(offsets[0] - 4, future_offsets[-1] + 4)

    if labels is not None:
        for ax, label in zip((ax_glucose, ax_meal, ax_hr), labels):
            if label:
                ax.text(1.02, 0.5, label, transform=ax.transAxes, fontsize=8.5,
                        va="center", ha="left", color="0.25", clip_on=False)

    fig.suptitle(f"What each baseline sees -- subject {segment.subject}, "
                 f"{now_time:%b %d, %H:%M}", y=0.995)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return now_time, future[-1], mean_calories


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poster figure: one real input window, one row per baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/CGMacros"),
                        help="raw CGMacros CSVs (meal log + per-minute activity)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=None,
                        help="output PNG (default results/input_window.png)")
    parser.add_argument("--metrics-csv", type=Path, default=None,
                        help="results table for the verdict labels "
                             "(default results/metrics_table.csv)")
    parser.add_argument("--no-verdicts", action="store_true",
                        help="omit the right-edge verdict labels (the poster "
                             "variant passes this -- there the results table "
                             "and findings sit beside the figure and already "
                             "carry the outcomes)")
    parser.add_argument("--rank", type=int, default=0,
                        help="0 = strongest post-meal rise, 1 = runner-up, ...")
    parser.add_argument("--subject", default=None, help="restrict to one subject id")
    parser.add_argument("--min-rise", type=float, default=MIN_RISE, metavar="MGDL",
                        help=f"minimum now->target rise (default {MIN_RISE:g})")
    parser.add_argument("--meal-age-min", type=float, default=MEAL_AGE_MIN,
                        metavar="MIN", help=f"meal at least this many minutes before "
                                            f"now (default {MEAL_AGE_MIN:g})")
    parser.add_argument("--meal-age-max", type=float, default=MEAL_AGE_MAX,
                        metavar="MIN", help=f"meal at most this many minutes before "
                                            f"now (default {MEAL_AGE_MAX:g})")
    parser.add_argument("--max-step", type=float, default=MAX_STEP, metavar="MGDL",
                        help="reject windows containing a consecutive-reading jump "
                             f"larger than this (default {MAX_STEP:g}, same "
                             f"physiology screen as figures.py)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="output resolution (default 300 -- poster print "
                             "quality; the shared paper figures use 150)")
    parser.add_argument("--figsize", type=float, nargs=2, default=(9.0, 7.5),
                        metavar=("W", "H"),
                        help="figure size in inches (default 9 7.5); "
                             "poster_figures.py standardizes it across figures")
    parser.add_argument("--list", type=int, default=0, metavar="N", dest="list_n",
                        help="print the top N eligible windows instead of plotting")
    args = parser.parse_args()
    out_path = args.out or args.results_dir / "input_window.png"
    metrics_csv = args.metrics_csv or args.results_dir / "metrics_table.csv"
    meal_age = (args.meal_age_min, args.meal_age_max)
    assert meal_age[0] <= meal_age[1], "--meal-age-min exceeds --meal-age-max"

    segments = load_segments(args.data_dir / "segments_split.npz")
    meals = load_meals(args.raw_dir)
    per_subject_meals = meals_by_subject(meals)

    candidates = find_candidates(segments, per_subject_meals, meal_age=meal_age,
                                 min_rise=args.min_rise, max_step=args.max_step,
                                 subject=args.subject)
    print(f"{len(candidates)} windows pass the glucose + meal criteria "
          f"(meal {meal_age[0]:g}-{meal_age[1]:g} min ago with logged macros, "
          f"rise >= {args.min_rise:g} mg/dL, no step > {args.max_step:g})")
    if not candidates:
        raise SystemExit("no eligible window -- loosen --min-rise or the meal-age band")

    activity_frames = load_activity(args.raw_dir)
    n_needed = args.list_n if args.list_n else args.rank + 1
    survivors, aggregates = screen_candidates(candidates, segments, activity_frames,
                                              n_needed)
    print(f"screened down to {len(survivors)} with full heart-rate coverage "
          f"(searched the strongest {n_needed} needed)")
    if len(survivors) < n_needed:
        raise SystemExit(f"only {len(survivors)} windows also have full HR coverage -- "
                         f"lower --rank/--list or loosen the criteria")

    if args.list_n:
        print(f"\n  {'#':>2} {'subject':>7} {'now':>16} {'rise':>6} {'meal ago':>8} "
              f"{'carbs':>6} type")
        for rank, candidate in enumerate(survivors):
            segment = segments[candidate.segment_index]
            now = pd.Timestamp(segment_times(segment)[candidate.now_index])
            meal = per_subject_meals[segment.subject].iloc[candidate.meal_row]
            print(f"  {rank:>2} {segment.subject:>7} {f'{now:%b %d %H:%M}':>16} "
                  f"{candidate.rise:>5.0f} {candidate.minutes_since_meal:>7.0f}m "
                  f"{meal['Carbs']:>5.0f}g {meal['type']}")
        return

    chosen = survivors[args.rank]
    segment = segments[chosen.segment_index]
    aggregate = aggregates[chosen.segment_index]
    meal = per_subject_meals[segment.subject].iloc[chosen.meal_row]

    # The window is about to be published, so prove every selection criterion on the
    # exact arrays being drawn rather than trusting the search that produced it.
    i, times = chosen.now_index, segment_times(segment)
    assert HISTORY - 1 <= i < len(segment.glucose) - HORIZON_STEPS
    minutes_since = float((times[i] - meal["timestamp"].to_datetime64())
                          / np.timedelta64(1, "m"))
    assert np.isclose(minutes_since, chosen.minutes_since_meal)
    assert meal_age[0] <= minutes_since <= meal_age[1]
    rise = float(segment.glucose[i + HORIZON_STEPS] - segment.glucose[i])
    assert np.isclose(rise, chosen.rise) and rise >= args.min_rise
    assert not bool(meal["macros_missing"]), "chosen meal has no logged macros"
    assert hr_window_complete(aggregate, i)
    drawn = segment.glucose[i - HISTORY + 1: i + HORIZON_STEPS + 1]
    assert np.abs(np.diff(drawn)).max() <= args.max_step, \
        "chosen window contains a sensor-artifact jump"

    # And verify the two data reconstructions this figure rests on, with the
    # project's own instruments (both raise on any mismatch):
    n_times = verify_segment_times([segment], args.raw_dir)
    n_hr = verify_aggregation([segment], activity_frames, readings_per_segment=50,
                              aggregates=[aggregate])
    print(f"verified: {n_times:,} reconstructed timestamps match the raw CSV; "
          f"{n_hr} sampled HR aggregates match a label-sliced recompute")

    if args.no_verdicts:
        print("  --no-verdicts: omitting the right-edge verdict labels")
        labels = None
    else:
        labels = verdict_labels(metrics_csv)
    now_time, target, mean_calories = draw_figure(segment, aggregate, chosen, meal,
                                                  labels, out_path, dpi=args.dpi,
                                                  figsize=tuple(args.figsize))

    hr_window = aggregate.hr[i - HISTORY + 1: i + 1]
    print(f"\nwrote {out_path}")
    print(f"  subject {segment.subject}, now = {now_time} "
          f"(rank {args.rank} of the eligible windows)")
    print(f"  glucose at now {segment.glucose[i]:.0f} -> target {target:.0f} mg/dL "
          f"(+{rise:.0f} in 30 min)")
    print(f"  meal: {meal['type']}, {meal['Carbs']:.0f} g carbs / "
          f"{meal['Protein']:.0f} g protein / {meal['Fat']:.0f} g fat, "
          f"{minutes_since:.0f} min before now")
    print(f"  heart rate over the window: mean {hr_window.mean():.0f} bpm "
          f"(range {hr_window.min():.0f}-{hr_window.max():.0f})"
          + ("" if mean_calories is None
             else f"; activity {mean_calories:.2f} kcal/min"))


if __name__ == "__main__":
    main()
