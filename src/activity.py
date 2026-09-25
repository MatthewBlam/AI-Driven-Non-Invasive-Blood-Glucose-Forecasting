"""Raw 1-minute activity rows -> per-reading 5-minute aggregates + frozen fill stats.

The activity equivalent of `meals.py`: read the HR and Calories (Activity) columns
once, put them on the same 5-minute grid the model lives on, and hand
`activity_features.py` clean arrays plus one frozen object holding every fitted
constant.

    python src/activity.py               # summary + every verification below
    python src/activity.py --seed 43     # show another split's fitted constants

Design decisions. The first three come from `activity_audit.py`; the fourth is the
missing-data policy the audit justified:

  1. A reading at time t aggregates the raw minutes (t-4 .. t) -- inclusive of t,
     nothing after t. That is the entire leakage rule for a dense signal: minute t is
     known at prediction time, exactly as the CGM value at t is itself an input.

  2. Energy is read from `Calories (Activity)`, not METs. Where both exist their
     correlation is 0.9999-1.0000 -- Calories IS METs times a per-subject constant --
     and Calories is present for all 44 subjects while METs is entirely absent for 11
     (including 046). METs is never loaded here.

  3. HR and Calories keep SEPARATE missingness. They fail differently: 032 is missing
     52% of HR but only 5% of Calories (poor sensor contact with the device still
     worn), so one flag cannot describe both. Each array carries its own NaNs;
     `hr_missing_fraction` is HR's because HR is the signal whose gaps are informative
     (device off ~ asleep/resting).

  4. Gaps are never interpolated. The audit found multi-day holes (028's HR dies 2.6
     days before its file ends, with CGM still recording); interpolating across those
     would manufacture a flat, confident vital sign out of nothing -- the same sin the
     Dexcom fix exists to punish. A missing slot stays NaN here and becomes a
     (train-median fill, flag) pair in `activity_features.py`.

What this module deliberately does NOT do: build features or z-score anything.
Fitted constants belong to a split, so they are produced by `fit_activity_stats` and
applied downstream -- the same shape as `fit_scaler` / `fit_macro_cleaning`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from meals import EXPECTED_SUBJECTS, segment_times
from preprocess import MALFORMED, Segment, find_subject_csvs, load_raw

REPO_ROOT = Path(__file__).resolve().parent.parent

HR_COLUMN = "HR"
CALORIES_COLUMN = "Calories (Activity)"

# Minutes aggregated into one reading: (t-4 .. t), see design decision 1.
MINUTES_PER_READING = 5

# Regression constant: what a correct load produces (preprocess.py asserts the same
# number when building the cache). EXPECTED_SUBJECTS is shared with meals.py.
EXPECTED_READINGS = 124_808


def load_activity(data_dir: Path) -> dict[str, pd.DataFrame]:
    """One minute-indexed frame per clean subject, holding HR and Calories (Activity).

    Keyed by SUBJECT, same argument as `meals_by_subject`: segments from one subject
    share a device record, and a reading near a segment's start still aggregates raw
    minutes that precede the segment.
    """
    frames: dict[str, pd.DataFrame] = {}

    for subject_id, csv_path in find_subject_csvs(data_dir).items():
        if subject_id in MALFORMED:                 # subject 007, irregular timestamps
            continue

        raw = load_raw(csv_path)
        missing_columns = [column for column in (HR_COLUMN, CALORIES_COLUMN)
                           if column not in raw.columns]
        assert not missing_columns, \
            f"subject {subject_id} lacks columns {missing_columns}"

        frame = (raw[["Timestamp", HR_COLUMN, CALORIES_COLUMN]]
                 .set_index("Timestamp").sort_index())

        # get_indexer below requires a unique, sorted index; state it rather than
        # trusting it (the audit found files that skip rows -- duplicates would be
        # the same class of surprise).
        assert frame.index.is_unique, f"subject {subject_id} has duplicate timestamps"
        frames[subject_id] = frame

    assert len(frames) == EXPECTED_SUBJECTS, \
        f"expected {EXPECTED_SUBJECTS} subjects, got {len(frames)}"
    return frames


@dataclass(frozen=True)
class SegmentActivity:
    """Per-reading 5-minute aggregates for one segment, aligned to its glucose array.

    All arrays have the segment's length. `hr` and `calories` are NaN where none of a
    reading's 5 source minutes had a value -- separately per signal (design decision
    3). `hr_missing_fraction` is the fraction of HR source minutes absent, 1.0 for an
    empty slot. `hour_of_day` is fractional (14:30 -> 14.5), for the clock features.
    """

    hr: np.ndarray
    calories: np.ndarray
    hr_missing_fraction: np.ndarray
    hour_of_day: np.ndarray

    def __len__(self) -> int:
        return len(self.hr)


def aggregate_segment(segment: Segment, minute_frame: pd.DataFrame) -> SegmentActivity:
    """Aggregate one subject's raw minutes onto one segment's 5-minute reading grid.

    The window is found by TIMESTAMP, not by row position. "The 5 rows ending at t"
    reads the wrong minutes wherever the file skips rows -- 285 skipped minute rows
    exist across 9 subjects, and near one of them the positional window silently
    reaches back past t-4 (caught by `verify_aggregation` on subject 003). Skipped
    rows also count as MISSING minutes: `hr_missing_fraction` is always out of
    MINUTES_PER_READING, so a reading with 2 valid minutes scores 0.6 whether the
    other three were NaN rows or no rows at all.
    """
    reading_times = pd.DatetimeIndex(segment_times(segment))
    minute_times = minute_frame.index.to_numpy()
    window_starts = reading_times.to_numpy() - np.timedelta64(MINUTES_PER_READING - 1, "m")

    # Rows inside (t-4 .. t): index range [first_row, last_row) per reading.
    first_rows = np.searchsorted(minute_times, window_starts, side="left")
    last_rows = np.searchsorted(minute_times, reading_times.to_numpy(), side="right")

    # The missing fraction divides by MINUTES_PER_READING, so a window may never
    # hold more rows than that. Whole-minute timestamps guarantee it (verified: no
    # file has sub-minute rows) -- but that is a property of the raw files, so state
    # it rather than trusting it.
    assert int((last_rows - first_rows).max(initial=0)) <= MINUTES_PER_READING, \
        f"subject {segment.subject}: a 5-min window holds more than " \
        f"{MINUTES_PER_READING} rows -- sub-minute timestamps?"

    hr_minutes = minute_frame[HR_COLUMN].to_numpy(dtype=float)
    calorie_minutes = minute_frame[CALORIES_COLUMN].to_numpy(dtype=float)

    n_readings = len(reading_times)
    hr = np.full(n_readings, np.nan)
    calories = np.full(n_readings, np.nan)
    hr_missing_fraction = np.ones(n_readings)

    for reading_index in range(n_readings):
        first_row = first_rows[reading_index]
        last_row = last_rows[reading_index]

        hr_window = hr_minutes[first_row:last_row]
        hr_present = ~np.isnan(hr_window)
        hr_missing_fraction[reading_index] = \
            1.0 - hr_present.sum() / MINUTES_PER_READING
        if hr_present.any():
            hr[reading_index] = hr_window[hr_present].mean()

        calorie_window = calorie_minutes[first_row:last_row]
        calories_present = ~np.isnan(calorie_window)
        if calories_present.any():
            calories[reading_index] = calorie_window[calories_present].mean()

    hour_of_day = (reading_times.hour.to_numpy()
                   + reading_times.minute.to_numpy() / 60.0)

    return SegmentActivity(hr=hr, calories=calories,
                           hr_missing_fraction=hr_missing_fraction,
                           hour_of_day=hour_of_day)


def aggregate_segments(segments: list[Segment],
                       frames: dict[str, pd.DataFrame]) -> list[SegmentActivity]:
    """One SegmentActivity per segment, in segment order."""
    aggregates = []
    for segment in segments:
        aggregates.append(aggregate_segment(segment, frames[segment.subject]))
    return aggregates


@dataclass(frozen=True)
class ActivityStats:
    """The fitted constants needed to turn raw aggregates into model input.

    Bundled in one object on purpose: scalers and fills must describe the SAME split,
    and passing them separately makes it possible to mix two by accident. Mirrors the
    `MacroCleaning` / `fit_scaler` contract.
    """

    hr_mean: float
    hr_std: float
    hr_fill: float               # train median, used where a slot has no HR at all
    calories_mean: float
    calories_std: float
    calories_fill: float
    n_train_readings: int        # how many non-NaN HR readings the stats were fit on


def fit_activity_stats(train_aggregates: list[SegmentActivity]) -> ActivityStats:
    """Fit scalers and fills on TRAINING segments only. NaN slots are excluded --
    they carry no information about the distribution, and a fill value computed from
    fill values would be circular."""
    assert train_aggregates, "no training aggregates given"

    hr_values = np.concatenate([aggregate.hr for aggregate in train_aggregates])
    hr_values = hr_values[~np.isnan(hr_values)]
    calorie_values = np.concatenate(
        [aggregate.calories for aggregate in train_aggregates])
    calorie_values = calorie_values[~np.isnan(calorie_values)]

    assert len(hr_values) > 0 and len(calorie_values) > 0, \
        "training segments contain no valid activity readings"

    stats = ActivityStats(
        hr_mean=float(hr_values.mean()),
        hr_std=float(hr_values.std()),
        hr_fill=float(np.median(hr_values)),
        calories_mean=float(calorie_values.mean()),
        calories_std=float(calorie_values.std()),
        calories_fill=float(np.median(calorie_values)),
        n_train_readings=len(hr_values),
    )
    assert stats.hr_std > 0 and stats.calories_std > 0, \
        "degenerate activity distribution: zero variance"
    return stats


def verify_aggregation(segments: list[Segment], frames: dict[str, pd.DataFrame],
                       readings_per_segment: int | None = 50,
                       aggregates: list[SegmentActivity] | None = None) -> int:
    """Check aggregates against a label-sliced recompute. Returns readings checked.

    Pass the `aggregates` actually in use to verify them; omitting them verifies a
    fresh `aggregate_segment` run. Injecting them is also what makes this check
    demonstrably able to fail -- a corrupted aggregate must trip it (exercised in
    __main__ below).

    An independent code path on purpose: `aggregate_segment` selects source minutes
    with numpy searchsorted over the index values, this recomputes them with pandas
    label slicing (`frame.loc[t-4min : t]`). The first version of the aggregation
    used index-POSITION arithmetic instead, and this check caught it reading minutes
    older than t-4 wherever the file skips rows (subject 003). Agreement also pins
    the leakage bound directly: the slice is closed at t and opens 4 minutes earlier.

    Defaults to 50 evenly spread readings per segment (~4,000 checks, a few seconds).
    Pass None to check every reading.
    """
    if aggregates is None:
        aggregates = aggregate_segments(segments, frames)
    checked = 0

    for segment, aggregate in zip(segments, aggregates):
        frame = frames[segment.subject]
        reading_times = pd.DatetimeIndex(segment_times(segment))

        if readings_per_segment is None:
            sampled_indices = range(len(reading_times))
        else:
            stride = max(1, len(reading_times) // readings_per_segment)
            sampled_indices = range(0, len(reading_times), stride)

        for reading_index in sampled_indices:
            reading_time = reading_times[reading_index]
            window_start = reading_time - pd.Timedelta(minutes=MINUTES_PER_READING - 1)
            window = frame.loc[window_start:reading_time]

            for column, aggregated in ((HR_COLUMN, aggregate.hr),
                                       (CALORIES_COLUMN, aggregate.calories)):
                values = window[column].dropna()
                if values.empty:
                    assert np.isnan(aggregated[reading_index]), (
                        f"subject {segment.subject} {column} at {reading_time}: "
                        f"aggregate has a value where the raw window is empty")
                else:
                    assert np.isclose(aggregated[reading_index], values.mean()), (
                        f"subject {segment.subject} {column} at {reading_time}: "
                        f"{aggregated[reading_index]} != {values.mean()}")

            expected_missing = 1.0 - window[HR_COLUMN].notna().sum() / MINUTES_PER_READING
            assert np.isclose(aggregate.hr_missing_fraction[reading_index],
                              expected_missing), (
                f"subject {segment.subject} missing fraction at {reading_time}: "
                f"{aggregate.hr_missing_fraction[reading_index]} != {expected_missing}")
            checked += 1

    assert checked > 0, "verified an empty set of readings -- that proves nothing"
    return checked


if __name__ == "__main__":
    from dataset import split_subjects
    from preprocess import load_segments

    parser = argparse.ArgumentParser(
        description="Load and aggregate the activity columns.")
    parser.add_argument("--data-dir", type=Path,
                        default=REPO_ROOT / "dataset" / "CGMacros",
                        help="directory holding the CGMacros-XXX/ subject folders")
    parser.add_argument("--segments", type=Path,
                        default=REPO_ROOT / "data" / "segments_split.npz",
                        help="B1 segment cache defining the 5-minute grid")
    parser.add_argument("--seed", type=int, default=42,
                        help="which split to show the fitted constants for")
    args = parser.parse_args()

    segments = load_segments(args.segments)
    frames = load_activity(args.data_dir)
    aggregates = aggregate_segments(segments, frames)

    print(f"{'=' * 78}\nACTIVITY AGGREGATES\n{'=' * 78}")
    n_readings = sum(len(aggregate) for aggregate in aggregates)
    assert n_readings == EXPECTED_READINGS, \
        f"expected {EXPECTED_READINGS:,} readings, got {n_readings:,}"

    all_hr = np.concatenate([aggregate.hr for aggregate in aggregates])
    all_calories = np.concatenate([aggregate.calories for aggregate in aggregates])
    all_missing = np.concatenate(
        [aggregate.hr_missing_fraction for aggregate in aggregates])

    hr_empty = np.isnan(all_hr)
    calories_empty = np.isnan(all_calories)
    print(f"{n_readings:,} readings across {len(segments)} segments, "
          f"{len(frames)} subjects")
    print(f"HR:       {100 * hr_empty.mean():.2f}% empty slots   "
          f"range {np.nanmin(all_hr):.1f} to {np.nanmax(all_hr):.1f} bpm "
          f"(5-min means, so tighter than the raw 30-176)")
    print(f"Calories: {100 * calories_empty.mean():.2f}% empty slots   "
          f"range {np.nanmin(all_calories):.2f} to {np.nanmax(all_calories):.2f} "
          f"kcal/min")
    print(f"slots with Calories but no HR: "
          f"{int((hr_empty & ~calories_empty).sum()):,} "
          f"(the two signals fail separately -- design decision 3)")

    # The flag and the value must tell the same story.
    assert np.array_equal(hr_empty, all_missing == 1.0), \
        "hr_missing_fraction == 1.0 disagrees with hr == NaN"
    partial = (all_missing > 0) & (all_missing < 1)
    print(f"partially covered slots (some of the 5 minutes absent): "
          f"{100 * partial.mean():.2f}%")

    print(f"\n{'=' * 78}\nFITTED CONSTANTS (seed {args.seed})\n{'=' * 78}")
    subjects = sorted(frames)
    train_subjects, val_subjects, test_subjects = split_subjects(
        subjects, seed=args.seed)
    train_aggregates = [aggregate for segment, aggregate in zip(segments, aggregates)
                        if segment.subject in set(train_subjects)]
    stats = fit_activity_stats(train_aggregates)

    print(f"{len(train_subjects)} train subjects, "
          f"{stats.n_train_readings:,} valid HR readings "
          f"(val {len(val_subjects)}, test {len(test_subjects)})")
    print(f"HR:       mean {stats.hr_mean:.2f}   std {stats.hr_std:.2f}   "
          f"fill {stats.hr_fill:.1f}")
    print(f"Calories: mean {stats.calories_mean:.3f}   std {stats.calories_std:.3f}   "
          f"fill {stats.calories_fill:.3f}")

    print(f"\n{'=' * 78}\nVERIFICATION\n{'=' * 78}")
    n_checked = verify_aggregation(segments, frames, aggregates=aggregates)
    print(f"aggregation: {n_checked:,} sampled readings match a label-sliced "
          f"recompute exactly (searchsorted vs .loc -- two independent paths)")

    # The check must be shown able to fail: a corrupted aggregate must trip it.
    # Corrupt a reading that HAS a value -- NaN + 1 is still NaN and would not trip,
    # so pick a segment with at least one valid HR reading rather than trusting
    # segment 0 to have one.
    target_index = next(index for index, aggregate in enumerate(aggregates)
                        if not np.isnan(aggregate.hr).all())
    target = aggregates[target_index]
    corrupted_hr = target.hr.copy()
    corrupted_hr[int(np.argmax(~np.isnan(corrupted_hr)))] += 1.0
    corrupted = SegmentActivity(hr=corrupted_hr, calories=target.calories,
                                hr_missing_fraction=target.hr_missing_fraction,
                                hour_of_day=target.hour_of_day)
    try:
        verify_aggregation([segments[target_index]], frames,
                           readings_per_segment=None, aggregates=[corrupted])
        raise SystemExit("verify_aggregation PASSED a corrupted aggregate")
    except AssertionError:
        print("negative check: a corrupted aggregate trips the verification")

    print("\nActivity aggregation complete. activity_features.py consumes "
          "load_activity + aggregate_segments + fit_activity_stats.")
