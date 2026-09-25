"""Audit the raw activity columns before building any activity features.

Read-only inspection of HR, METs, and Calories (Activity) across the 44 modelled
subjects: per-subject coverage and gap structure, the METs-Calories relationship,
and how sparse "being active" is on the modelled 5-minute grid.

    python src/activity_audit.py                     # the three report blocks
    python src/activity_audit.py --detail            # + every subject's coverage row
    python src/activity_audit.py --csv results/activity_audit_coverage.csv

This script ASSERTS NOTHING. It is for looking. The handling rules it justifies live
in ``src/activity.py``; the numbers it prints are recorded in ``PROJECT_STATE.md`` so
a future run can be diffed against them.

Design decisions
~~~~~~~~~~~~~~~~
- Coverage is measured over the MODELLED subject set (whoever has segments in the
  cache), not over the files on disk -- the audience for these numbers is the model,
  and the segments also provide the 5-minute grid that block 3 needs.
- The longest HR gap is measured in TIME between valid readings, including the gap
  between the file's edges and its first/last valid reading. Counting NaN rows misses
  minutes the file skips outright (018), and ignoring the edges misses a sensor that
  died before the study ended (028) -- both happen in this data.
- METs and Calories (Activity) are reported together because where both exist their
  correlation is exactly 1.0 -- Calories is METs times a per-subject constant, which
  is what lets ``activity.py`` read energy from Calories for all 44 subjects instead
  of losing the 11 whose METs column is empty.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from activity import MINUTES_PER_READING
from baselines import HISTORY
from meals import segment_times
from preprocess import Segment, find_subject_csvs, load_raw, load_segments

REPO_ROOT = Path(__file__).resolve().parent.parent

HR = "HR"
METS = "METs"
CALORIES_ACTIVITY = "Calories (Activity)"

# The METs column is stored at 10x the physiological scale: 10 = 1.0 MET (rest).
# "Active" means >= 3.0 real METs (a brisk walk), i.e. >= 30 in the raw column.
ACTIVE_METS_RAW = 30.0

# Prediction horizon used for the window arithmetic in block 3, in 5-min steps.
HORIZON_STEPS = 6


def collect_activity(raw_dir: Path,
                     subjects: set[str]) -> dict[str, pd.DataFrame]:
    """One minute-indexed frame per modelled subject, holding the activity columns.

    A subject whose file lacks the METs column entirely would get NaN there; today
    every file has the column (11 subjects just never have a value in it), and the
    coverage table distinguishes the two cases.
    """
    frames: dict[str, pd.DataFrame] = {}
    for subject_id, path in find_subject_csvs(raw_dir).items():
        if subject_id not in subjects:
            continue
        raw = load_raw(path)
        columns = ["Timestamp"]
        for column in (HR, METS, CALORIES_ACTIVITY):
            if column in raw.columns:
                columns.append(column)
        frame = raw[columns].set_index("Timestamp").sort_index()
        for column in (HR, METS, CALORIES_ACTIVITY):
            if column not in frame.columns:
                frame[column] = np.nan
        frames[subject_id] = frame
    return frames


def longest_reading_gap_minutes(series: pd.Series) -> float:
    """Longest stretch of the file with no valid reading, in minutes.

    Three ways to get this wrong, all found in this data: counting NaN ROWS misses
    minutes the file skips outright (018 chains NaN rows and skipped rows into a
    9.4-day hole); spacing between valid timestamps misses gaps at the file's edges
    (028's HR dies 2.6 days before its last row); and both at once. So: interior
    gaps from valid-timestamp spacing, plus the leading and trailing gaps against
    the file's own first and last row.
    """
    valid_times = series.dropna().index
    if len(valid_times) < 2:
        return float("nan")
    interior = np.diff(valid_times.to_numpy()) / np.timedelta64(1, "m") - 1.0
    leading = (valid_times[0] - series.index[0]).total_seconds() / 60
    trailing = (series.index[-1] - valid_times[-1]).total_seconds() / 60
    return float(max(interior.max(), leading, trailing))


def zero_second_difference_fraction(series: pd.Series) -> float:
    """Fraction of exactly-zero second differences among the valid readings.

    The interpolation tell from the Dexcom audit: a linearly interpolated series
    scores ~1.0 away from the knots. Computed over dropna(), so a handful of pairs
    straddle gaps -- irrelevant at the 0.1-vs-1.0 scale this screens for.
    """
    values = series.dropna().to_numpy()
    if len(values) < 100:
        return float("nan")
    second_difference = np.diff(values, n=2)
    return float((np.abs(second_difference) == 0).mean())


def coverage_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-subject coverage of the three activity columns."""
    rows = []
    for subject_id, frame in frames.items():
        hr = frame[HR]
        mets = frame[METS]
        calories = frame[CALORIES_ACTIVITY]

        skipped_rows = int((frame.index.to_series().diff()
                            .dt.total_seconds().div(60) > 1).sum())

        if mets.notna().any():
            mets_status = f"{100 * mets.isna().mean():.1f}% NaN"
        else:
            mets_status = "ALL NaN"

        rows.append({
            "subject": subject_id,
            "rows": len(frame),
            "skipped_rows": skipped_rows,
            "hr_nan_pct": 100 * hr.isna().mean(),
            "hr_longest_gap_min": longest_reading_gap_minutes(hr),
            "hr_d2_zero_frac": zero_second_difference_fraction(hr),
            "hr_min": hr.min(),
            "hr_max": hr.max(),
            "mets_status": mets_status,
            "mets_min": mets.min(),
            "mets_max": mets.max(),
            "cal_nan_pct": 100 * calories.isna().mean(),
        })
    return pd.DataFrame(rows).set_index("subject").sort_index()


def report_coverage(coverage: pd.DataFrame, detail: bool) -> None:
    """Block 1: who has data, where the holes are, and whether HR is real."""
    print(f"\n{'=' * 78}\n1. PER-SUBJECT COVERAGE\n{'=' * 78}")

    no_mets = sorted(coverage.index[coverage["mets_status"] == "ALL NaN"])
    print(f"{len(coverage)} modelled subjects, "
          f"{int(coverage['rows'].sum()):,} raw minute rows")
    print(f"\nHR NaN%:            min {coverage['hr_nan_pct'].min():.1f}   "
          f"median {coverage['hr_nan_pct'].median():.1f}   "
          f"max {coverage['hr_nan_pct'].max():.1f} "
          f"(subject {coverage['hr_nan_pct'].idxmax()})")
    print(f"longest HR gap:     median {coverage['hr_longest_gap_min'].median():.0f} min   "
          f"max {coverage['hr_longest_gap_min'].max():.0f} min "
          f"(subject {coverage['hr_longest_gap_min'].idxmax()})")
    print(f"HR d2==0 fraction:  {coverage['hr_d2_zero_frac'].min():.2f} to "
          f"{coverage['hr_d2_zero_frac'].max():.2f}   "
          f"(a linearly interpolated series would be ~1.0 -> HR is measured data)")
    print(f"HR range:           {coverage['hr_min'].min():.0f} to "
          f"{coverage['hr_max'].max():.0f} bpm")
    print(f"skipped minute rows: {int(coverage['skipped_rows'].sum())} across "
          f"{int((coverage['skipped_rows'] > 0).sum())} subjects")

    print(f"\nMETs is ALL NaN for {len(no_mets)} subjects: {', '.join(no_mets)}")
    print("(046 is on that list -- one of the two LOSO subjects meals flipped. Any")
    print(" METs-dependent analysis silently drops it; energy reads Calories instead.)")

    over_20 = sorted(coverage.index[coverage["hr_nan_pct"] > 20])
    print(f"\nsubjects with >20% HR missing: {', '.join(over_20)}")
    print("A median longest gap of "
          f"{coverage['hr_longest_gap_min'].median():.0f} min (~"
          f"{coverage['hr_longest_gap_min'].median() / 60:.0f} h) is a device on the"
          " charger overnight:")
    print("gaps are structural and informative (off wrist ~ asleep/resting), which is")
    print("why they become fill+flag pairs rather than interpolation.")

    if detail:
        print()
        formatted = coverage.copy()
        for column in ("hr_nan_pct", "hr_longest_gap_min", "hr_d2_zero_frac",
                       "cal_nan_pct"):
            formatted[column] = formatted[column].round(2)
        print(formatted.to_string())


def report_mets_calories(frames: dict[str, pd.DataFrame]) -> None:
    """Block 2: Calories (Activity) is METs times a per-subject constant."""
    print(f"\n{'=' * 78}\n2. METs vs CALORIES (ACTIVITY)\n{'=' * 78}")

    correlations = []
    ratios = []
    mets_minima = []
    zero_mets_subjects = []
    for subject_id, frame in frames.items():
        both = frame[[METS, CALORIES_ACTIVITY]].dropna()
        if len(both) < 100:
            continue
        correlations.append(float(np.corrcoef(both[METS],
                                              both[CALORIES_ACTIVITY])[0, 1]))
        ratios.append(float((both[CALORIES_ACTIVITY] / both[METS])
                            .replace(np.inf, np.nan).median()))
        mets_minima.append(float(both[METS].min()))
        if (both[METS] == 0).any():
            zero_mets_subjects.append(subject_id)

    print(f"{len(correlations)} subjects have both columns")
    print(f"corr(Calories, METs) per subject: min {min(correlations):.4f}, "
          f"max {max(correlations):.4f}")
    print(f"median Calories/METs ratio: {min(ratios):.3f} to {max(ratios):.3f} "
          f"-- a per-subject constant that scales like body mass")
    print("\nSo Calories (Activity) IS the METs signal, present for all 44 subjects.")
    print("The energy features read Calories; METs is only used below, for the")
    print("physiologically-anchored activity threshold.")

    print(f"\nMETs minimum per subject is {min(m for m in mets_minima if m > 0):.0f} "
          f"for most -- the column is stored at 10x scale (10 = 1.0 MET at rest,")
    print(f"so 'active >= 3 METs' means raw >= {ACTIVE_METS_RAW:.0f}).")
    if zero_mets_subjects:
        print(f"Raw zeros (0.0 METs, not a physiological value -- device artifact) "
              f"appear for: {', '.join(sorted(zero_mets_subjects))}")


def aggregate_mets_and_hr_presence(
        segment: Segment,
        frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-reading 5-min METs mean and HR-slot-empty flag for one segment.

    Each reading at time t aggregates the raw minutes (t-4 .. t) -- nothing after t.
    The window is selected by TIMESTAMP, not row position: near a skipped row, "the 5
    rows ending at t" silently reaches minutes older than t-4 (the activity.py
    verification caught exactly that on subject 003). This deliberately mirrors
    `activity.aggregate_segment` rather than calling it -- the audit needs METs,
    which the loader deliberately never reads -- but the window constant is shared
    so the two cannot drift. Returns (mets_mean, hr_slot_empty, n_unmatched_times);
    every reading time is expected to match a raw row exactly (block 3 prints the
    count).
    """
    reading_times = pd.DatetimeIndex(segment_times(segment))
    minute_times = frame.index.to_numpy()
    window_starts = (reading_times.to_numpy()
                     - np.timedelta64(MINUTES_PER_READING - 1, "m"))
    first_rows = np.searchsorted(minute_times, window_starts, side="left")
    last_rows = np.searchsorted(minute_times, reading_times.to_numpy(), side="right")
    matched = frame.index.get_indexer(reading_times) >= 0

    hr_values = frame[HR].to_numpy()
    mets_values = frame[METS].to_numpy()

    mets_mean = np.full(len(reading_times), np.nan)
    hr_slot_empty = np.ones(len(reading_times), dtype=bool)
    for reading_index in range(len(reading_times)):
        first_row = first_rows[reading_index]
        last_row = last_rows[reading_index]
        hr_window = hr_values[first_row:last_row]
        if (~np.isnan(hr_window)).any():
            hr_slot_empty[reading_index] = False
        mets_window = mets_values[first_row:last_row]
        valid = ~np.isnan(mets_window)
        if valid.any():
            mets_mean[reading_index] = mets_window[valid].mean()

    return mets_mean, hr_slot_empty, int((~matched).sum())


def report_sparsity(segments: list[Segment],
                    frames: dict[str, pd.DataFrame]) -> None:
    """Block 3: how sparse activity is where the model will actually look."""
    print(f"\n{'=' * 78}\n3. SPARSITY ON THE MODELLED 5-MINUTE GRID\n{'=' * 78}")

    reading_is_active = []      # readings on subjects that have METs
    window_active_30 = []       # per ph30 window: any active reading in last 30 min
    window_active_60 = []
    slot_empty = []             # per reading, all 44 subjects: no HR minute at all
    unmatched_total = 0
    reading_total = 0
    window_total = 0

    for segment in segments:
        frame = frames[segment.subject]
        mets_mean, hr_slot_empty, n_unmatched = \
            aggregate_mets_and_hr_presence(segment, frame)
        unmatched_total += n_unmatched
        reading_total += len(mets_mean)
        slot_empty.append(hr_slot_empty)

        has_mets = not np.isnan(mets_mean).all()
        if has_mets:
            active = mets_mean >= ACTIVE_METS_RAW
            reading_is_active.append(active)

        # Prediction-time rows of the ph30 windows: a window needs HISTORY readings
        # of history and HORIZON_STEPS of future, so its "now" rows are
        # [HISTORY-1, len-HORIZON_STEPS). Matches the modelled window count exactly.
        first_row = HISTORY - 1
        last_row = len(segment.glucose) - HORIZON_STEPS
        if last_row <= first_row:
            continue
        window_total += last_row - first_row

        if has_mets:
            recent_30 = (pd.Series(active.astype(float))
                         .rolling(6, min_periods=1).max().to_numpy())
            recent_60 = (pd.Series(active.astype(float))
                         .rolling(12, min_periods=1).max().to_numpy())
            window_active_30.append(recent_30[first_row:last_row])
            window_active_60.append(recent_60[first_row:last_row])

    print(f"reading times with no matching raw row: {unmatched_total} of "
          f"{reading_total:,} (the 5-min grid and the raw files agree exactly)")
    print(f"ph30 windows implied by the grid arithmetic: {window_total:,}")

    empty = np.concatenate(slot_empty)
    print(f"\n5-min slots with NO HR minute at all: {100 * empty.mean():.2f}% "
          f"of {len(empty):,}")

    active = np.concatenate(reading_is_active)
    print(f"readings with METs >= 3.0 (active):   {100 * active.mean():.1f}% "
          f"of {len(active):,} (subjects with METs only)")

    active_30 = np.concatenate(window_active_30)
    active_60 = np.concatenate(window_active_60)
    print(f"windows with an active reading in the last 30 min: "
          f"{100 * active_30.mean():.1f}%")
    print(f"windows with an active reading in the last 60 min: "
          f"{100 * active_60.mean():.1f}%")

    print("\nCompare the meal audit: 8.0% of windows had a meal in the prior 30 min.")
    print("Activity is more frequent but far fainter per occurrence -- mild walking,")
    print("not workouts -- so dilution cuts against it. The ridge screen quantifies")
    print("what survives: see ridge_activity_screen.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit the raw CGMacros activity columns.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                        help="directory holding segments_split.npz")
    parser.add_argument("--raw-dir", type=Path,
                        default=REPO_ROOT / "dataset" / "CGMacros",
                        help="directory holding the CGMacros-XXX/ subject folders")
    parser.add_argument("--detail", action="store_true",
                        help="also print every subject's coverage row")
    parser.add_argument("--csv", type=Path, default=None,
                        help="write the per-subject coverage table here")
    args = parser.parse_args()

    segments = load_segments(args.data_dir / "segments_split.npz")
    modelled_subjects = {segment.subject for segment in segments}
    print(f"{len(segments)} segments, {len(modelled_subjects)} modelled subjects")

    frames = collect_activity(args.raw_dir, modelled_subjects)
    coverage = coverage_table(frames)

    report_coverage(coverage, args.detail)
    report_mets_calories(frames)
    report_sparsity(segments, frames)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(args.csv)
        print(f"\nwrote {args.csv}")

    print("\nNo assertions -- this is a read-only inspection.")
