"""Activity state per reading, in named feature blocks.

The activity counterpart of `meal_features.py`: consume the 5-minute aggregates and
frozen stats from `activity.py`, emit one (L, K) matrix per segment. Row i uses only
readings at or before i -- every rolling statistic is trailing, the activity-recency
scan is causal, and the test below proves it against a perturbed raw file.

    python src/activity_features.py               # demo subject + all tests
    python src/activity_features.py --tests       # tests only
    python src/activity_features.py --coverage    # per-feature stats, all segments

Feature blocks, named so the ablation ladder is a set union rather than a rewrite:

    H  heart rate       hr_z, hr_mean30_z, hr_slope15
    R  subject-relative hr_rel, hr_rel30       (vs a causal per-subject baseline)
    F  missingness      hr_missing, hr_missfrac60
    A  activity timing  dt_active_norm, act_decay   (the meal T-block analog)
    E  energy           cal_z, cal30_z         (Calories (Activity), all 44 subjects)
    K  clock            sin_hour, cos_hour

K is a CONTROL, not an activity block: HR partly just tracks the clock, and any
activity gain must beat the clock-only arm before it may be called physiological.
It is therefore excluded from DEFAULT_BLOCKS and reported separately everywhere.

Design constants. Honesty labels, because a reviewer will ask:
    ELEVATED_HR_Z       CONVENTION -- "elevated" is hr_z >= 1 on a MEASURED slot. One
                        global threshold is crude (a fit subject's z=1 is a different
                        exertion than an unfit one's); block R exists to compensate.
    DT_ACTIVE_CLIP_MIN  BORROWED from the meal T block (300 min), for comparability
    ACT_DECAY_TAU_MIN   BORROWED from the meal T block (60 min), not re-derived
    30 / 60 min rolling windows   CONVENTION, chosen to match the evaluation strata
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from activity import ActivityStats, SegmentActivity, aggregate_segment

REPO_ROOT = Path(__file__).resolve().parent.parent

ACTIVITY_BLOCKS = {
    "H": ("hr_z", "hr_mean30_z", "hr_slope15"),
    "R": ("hr_rel", "hr_rel30"),
    "F": ("hr_missing", "hr_missfrac60"),
    "A": ("dt_active_norm", "act_decay"),
    "E": ("cal_z", "cal30_z"),
    "K": ("sin_hour", "cos_hour"),
}

# Everything except the K control -- see the module docstring.
DEFAULT_BLOCKS = ("H", "R", "F", "A", "E")

ELEVATED_HR_Z = 1.0
DT_ACTIVE_CLIP_MIN = 300.0
ACT_DECAY_TAU_MIN = 60.0

READING_INTERVAL_MIN = 5
READINGS_PER_30_MIN = 6
READINGS_PER_60_MIN = 12
SLOPE_LOOKBACK_READINGS = 3          # 15 min, matching the trend stratum's window


def block_names(blocks: tuple[str, ...] = DEFAULT_BLOCKS) -> list[str]:
    """Column names for a block list, in the same order `build_features` emits.

    Kept separate so callers can size a matrix or label a table without building
    one; the two are checked against each other in `test_names_match_matrix`.
    """
    names: list[str] = []
    for block in blocks:
        if block not in ACTIVITY_BLOCKS:
            raise ValueError(f"unknown block {block!r}; known: {sorted(ACTIVITY_BLOCKS)}")
        names += ACTIVITY_BLOCKS[block]
    return names


def trailing_mean(values: np.ndarray, window_readings: int) -> np.ndarray:
    """Mean over the last `window_readings` readings, this one included. Trailing by
    construction, which is the causality argument for every rolling feature here."""
    series = pd.Series(values)
    return series.rolling(window_readings, min_periods=1).mean().to_numpy()


def minutes_since_elevated(elevated: np.ndarray) -> np.ndarray:
    """Causal scan: 0 at an elevated reading, +5 per reading since, clipped at
    DT_ACTIVE_CLIP_MIN. Starts at the clip -- before any elevation is seen the state
    is "not recently active", the same choice the meal T block makes for fasting."""
    minutes_since = np.empty(len(elevated))
    since = DT_ACTIVE_CLIP_MIN
    for reading_index in range(len(elevated)):
        if elevated[reading_index]:
            since = 0.0
        else:
            since = min(since + READING_INTERVAL_MIN, DT_ACTIVE_CLIP_MIN)
        minutes_since[reading_index] = since
    return minutes_since


def build_features(aggregate: SegmentActivity, stats: ActivityStats,
                   blocks: tuple[str, ...] = DEFAULT_BLOCKS) -> tuple[np.ndarray, list[str]]:
    """(L, K) activity-state matrix for one segment. Row i uses only readings <= i.

    `stats` must come from `fit_activity_stats` on TRAINING segments -- passing stats
    fitted elsewhere is the leak this signature makes hard to write by accident.
    """
    missing = np.isnan(aggregate.hr)
    hr_filled = np.where(missing, stats.hr_fill, aggregate.hr)
    hr_z = (hr_filled - stats.hr_mean) / stats.hr_std

    # Causal per-subject baseline: the expanding median of everything seen so far in
    # this segment. Resets at segment boundaries, so the first hours of a segment
    # have an unstable baseline -- acceptable at ~5-day mean segment length, and the
    # honest alternative (a cross-segment state) would couple segments together.
    expanding_median = pd.Series(hr_filled).expanding(min_periods=1).median().to_numpy()
    hr_rel = (hr_filled - expanding_median) / stats.hr_std

    # A filled slot can never count as activity: the fill is a train median, and a
    # median crossing the threshold would invent exercise out of a charger gap.
    elevated = (~missing) & (hr_z >= ELEVATED_HR_Z)
    dt_active = minutes_since_elevated(elevated)

    hr_slope15 = np.zeros(len(hr_z))
    lookback = SLOPE_LOOKBACK_READINGS
    hr_slope15[lookback:] = ((hr_z[lookback:] - hr_z[:-lookback])
                             / (lookback * READING_INTERVAL_MIN))

    calories_filled = np.where(np.isnan(aggregate.calories),
                               stats.calories_fill, aggregate.calories)
    calories_z = (calories_filled - stats.calories_mean) / stats.calories_std

    hour_angle = 2.0 * np.pi * aggregate.hour_of_day / 24.0

    features_by_name = {
        "hr_z": hr_z,
        "hr_mean30_z": trailing_mean(hr_z, READINGS_PER_30_MIN),
        "hr_slope15": hr_slope15,
        "hr_rel": hr_rel,
        "hr_rel30": trailing_mean(hr_rel, READINGS_PER_30_MIN),
        "hr_missing": missing.astype(float),
        "hr_missfrac60": trailing_mean(aggregate.hr_missing_fraction,
                                       READINGS_PER_60_MIN),
        "dt_active_norm": dt_active / DT_ACTIVE_CLIP_MIN,
        "act_decay": np.exp(-dt_active / ACT_DECAY_TAU_MIN),
        "cal_z": calories_z,
        "cal30_z": trailing_mean(calories_z, READINGS_PER_30_MIN),
        "sin_hour": np.sin(hour_angle),
        "cos_hour": np.cos(hour_angle),
    }

    names = block_names(blocks)
    features = np.column_stack([features_by_name[name] for name in names])

    assert np.isfinite(features).all(), "non-finite activity feature"
    assert features.shape == (len(aggregate), len(names)), \
        f"shape {features.shape} does not match {len(names)} names"
    return features, names


def build_all_features(aggregates: list[SegmentActivity], stats: ActivityStats,
                       blocks: tuple[str, ...] = DEFAULT_BLOCKS) -> list[np.ndarray]:
    """One (L, K) matrix per segment aggregate, in the same order."""
    return [build_features(aggregate, stats, blocks)[0] for aggregate in aggregates]


# ---------------------------------------------------------------------------------------
# Tests. These run by default -- the leak test is the one thing in this module that
# must never silently regress.
# ---------------------------------------------------------------------------------------

def test_names_match_matrix(aggregate: SegmentActivity, stats: ActivityStats) -> None:
    """Every block list must produce a matrix whose width equals its name list."""
    for blocks in [DEFAULT_BLOCKS, ("H",), ("K",), ("H", "A", "K"),
                   tuple(ACTIVITY_BLOCKS)]:
        features, names = build_features(aggregate, stats, blocks)
        assert features.shape[1] == len(names) == sum(
            len(ACTIVITY_BLOCKS[block]) for block in blocks)

    try:
        block_names(("H", "Z"))
        raise AssertionError("unknown block 'Z' was accepted")
    except ValueError:
        pass
    print("  names-match-matrix ..... PASS (5 block combinations + unknown rejected)")


def test_no_future_leak(segment, minute_frame: pd.DataFrame,
                        stats: ActivityStats) -> None:
    """Overwriting every raw minute after t_i must leave feature rows <= i
    bit-identical, and must change at least one later row.

    Both directions are asserted, on non-empty subsets -- a leak test that checks
    zero rows, or that could not detect its own perturbation, proves nothing (the
    B1 Part 4 lesson).
    """
    from meals import segment_times

    original, _ = build_features(aggregate_segment(segment, minute_frame), stats,
                                 tuple(ACTIVITY_BLOCKS))

    reading_times = pd.DatetimeIndex(segment_times(segment))
    cutoff_index = len(reading_times) // 2
    cutoff_time = reading_times[cutoff_index]

    perturbed_frame = minute_frame.copy()
    later_minutes = perturbed_frame.index > cutoff_time
    assert later_minutes.sum() > 0, "perturbation window is empty -- test proves nothing"
    perturbed_frame.loc[later_minutes, "HR"] = 200.0
    perturbed_frame.loc[later_minutes, "Calories (Activity)"] = 20.0

    perturbed, _ = build_features(aggregate_segment(segment, perturbed_frame), stats,
                                  tuple(ACTIVITY_BLOCKS))

    unchanged_rows = cutoff_index + 1
    assert unchanged_rows > 0 and unchanged_rows < len(original)
    assert np.array_equal(original[:unchanged_rows], perturbed[:unchanged_rows]), \
        "feature rows at or before the cutoff changed when FUTURE minutes changed -- LEAK"
    assert not np.array_equal(original[unchanged_rows:], perturbed[unchanged_rows:]), \
        "no later row changed -- the perturbation never reached the features"

    print(f"  no-future-leak ......... PASS ({unchanged_rows} rows bit-identical, "
          f"later rows respond)")


def test_filled_slots_never_active(stats: ActivityStats) -> None:
    """A filled slot must not reset the activity clock, even when the fill value
    itself would cross the elevated threshold."""
    high_fill_stats = ActivityStats(
        hr_mean=stats.hr_mean, hr_std=stats.hr_std,
        hr_fill=stats.hr_mean + 2.0 * stats.hr_std,      # fill z-scores to +2.0
        calories_mean=stats.calories_mean, calories_std=stats.calories_std,
        calories_fill=stats.calories_fill, n_train_readings=stats.n_train_readings)

    elevated_hr = stats.hr_mean + 3.0 * stats.hr_std
    synthetic = SegmentActivity(
        hr=np.array([np.nan, elevated_hr, np.nan, np.nan]),
        calories=np.array([1.0, 1.0, 1.0, 1.0]),
        hr_missing_fraction=np.array([1.0, 0.0, 1.0, 1.0]),
        hour_of_day=np.array([8.0, 8.1, 8.2, 8.3]))

    features, names = build_features(synthetic, high_fill_stats,
                                     tuple(ACTIVITY_BLOCKS))
    dt_norm = features[:, names.index("dt_active_norm")]

    assert dt_norm[0] == 1.0, "a filled slot before any elevation read as active"
    assert dt_norm[1] == 0.0, "a genuinely elevated reading did not reset the clock"
    expected_after = np.array([5.0, 10.0]) / DT_ACTIVE_CLIP_MIN
    assert np.allclose(dt_norm[2:], expected_after), \
        f"filled slots after an elevation moved the clock wrongly: {dt_norm[2:]}"
    print("  filled-never-active .... PASS (fill at z=+2 cannot reset the clock)")


def test_missing_rows_use_the_fill(aggregates: list[SegmentActivity],
                                   stats: ActivityStats) -> None:
    """On every empty slot, hr_z must equal the z-scored fill exactly.

    Takes the full aggregate list and picks a segment that HAS empty slots -- run on
    a fully covered segment this would check zero rows and prove nothing (the
    empty-subset lesson).
    """
    aggregate = next((a for a in aggregates if np.isnan(a.hr).any()), None)
    assert aggregate is not None, \
        "no segment with an empty slot -- the fill path is untested"

    features, names = build_features(aggregate, stats, tuple(ACTIVITY_BLOCKS))
    empty_slots = np.isnan(aggregate.hr)
    fill_z = (stats.hr_fill - stats.hr_mean) / stats.hr_std
    observed = features[empty_slots, names.index("hr_z")]
    assert np.allclose(observed, fill_z), \
        f"empty slots read {observed[:3]}, expected the fill at z={fill_z:.3f}"

    flags = features[empty_slots, names.index("hr_missing")]
    assert np.all(flags == 1.0), "an empty slot is not flagged hr_missing"
    print(f"  missing-uses-fill ...... PASS ({int(empty_slots.sum())} empty slots "
          f"filled at z={fill_z:.3f} and flagged)")


def test_clock_features_on_unit_circle(aggregate: SegmentActivity,
                                       stats: ActivityStats) -> None:
    """sin/cos of the hour must sit on the unit circle and actually rotate."""
    features, names = build_features(aggregate, stats, ("K",))
    sin_hour = features[:, names.index("sin_hour")]
    cos_hour = features[:, names.index("cos_hour")]
    assert np.allclose(sin_hour ** 2 + cos_hour ** 2, 1.0)
    assert sin_hour.std() > 0, "sin_hour is constant -- hour never advanced"
    print("  clock-unit-circle ...... PASS")


def run_tests(segment, minute_frame: pd.DataFrame, aggregate: SegmentActivity,
              aggregates: list[SegmentActivity], stats: ActivityStats) -> None:
    print("tests:")
    test_names_match_matrix(aggregate, stats)
    test_no_future_leak(segment, minute_frame, stats)
    test_filled_slots_never_active(stats)
    test_missing_rows_use_the_fill(aggregates, stats)
    test_clock_features_on_unit_circle(aggregate, stats)


def coverage_report(aggregates: list[SegmentActivity],
                    stats: ActivityStats) -> pd.DataFrame:
    """Per-feature distribution across every reading. The sanity check on scale --
    everything should be O(1); dt_active_norm's mean is the "not recently active"
    share of the cohort."""
    stacked = np.vstack(build_all_features(aggregates, stats, tuple(ACTIVITY_BLOCKS)))
    names = block_names(tuple(ACTIVITY_BLOCKS))
    return pd.DataFrame({
        "feature": names,
        "mean": stacked.mean(axis=0),
        "std": stacked.std(axis=0),
        "min": stacked.min(axis=0),
        "max": stacked.max(axis=0),
    }).set_index("feature")


if __name__ == "__main__":
    from activity import aggregate_segments, fit_activity_stats, load_activity
    from dataset import split_subjects
    from preprocess import load_segments

    parser = argparse.ArgumentParser(
        description="Build activity-state features for CGM readings.")
    parser.add_argument("--data-dir", type=Path,
                        default=REPO_ROOT / "dataset" / "CGMacros")
    parser.add_argument("--segments", type=Path,
                        default=REPO_ROOT / "data" / "segments_split.npz")
    parser.add_argument("--seed", type=int, default=42,
                        help="split whose fitted stats to use")
    parser.add_argument("--subject", default=None,
                        help="subject to demo (default: the first segment's)")
    parser.add_argument("--tests", action="store_true", help="run the tests and stop")
    parser.add_argument("--coverage", action="store_true",
                        help="per-feature stats over all segments")
    args = parser.parse_args()

    segments = load_segments(args.segments)
    frames = load_activity(args.data_dir)
    aggregates = aggregate_segments(segments, frames)

    train_subjects, _, _ = split_subjects(sorted(frames), seed=args.seed)
    train_aggregates = [aggregate for segment, aggregate in zip(segments, aggregates)
                        if segment.subject in set(train_subjects)]
    stats = fit_activity_stats(train_aggregates)

    subject = args.subject or segments[0].subject
    segment_index = next(i for i, s in enumerate(segments) if s.subject == subject)
    segment = segments[segment_index]
    aggregate = aggregates[segment_index]

    print(f"{'=' * 78}\nACTIVITY FEATURES (subject {subject}, seed {args.seed})\n{'=' * 78}")
    features, names = build_features(aggregate, stats)
    print(f"segment: {len(aggregate)} readings")
    print(f"matrix:  {features.shape}   blocks {DEFAULT_BLOCKS}")

    run_tests(segment, frames[subject], aggregate, aggregates, stats)

    if args.tests:
        raise SystemExit(0)

    # Show the state around the first activity bout: the clearest demonstration that
    # the clock resets at elevation and decays after -- the meal-features demo, for
    # activity.
    all_features, all_names = build_features(aggregate, stats, tuple(ACTIVITY_BLOCKS))
    dt_column = all_features[:, all_names.index("dt_active_norm")]
    if not (dt_column == 0.0).any():
        print("\n(no elevated reading in this segment -- pick another with --subject)")
        raise SystemExit(0)
    first_elevated = int(np.argmax(dt_column == 0.0))
    show = ["hr_z", "hr_rel", "dt_active_norm", "act_decay", "cal_z", "hr_missing"]
    picked = [all_names.index(name) for name in show]

    print(f"\n{'=' * 78}\nAROUND THE FIRST ELEVATED READING\n{'=' * 78}")
    print("reading  " + "  ".join(f"{name:>14}" for name in show))
    for i in range(max(0, first_elevated - 2),
                   min(len(all_features), first_elevated + 5)):
        marker = " <- elevated" if dt_column[i] == 0 else ""
        print(f"{i:>7}  "
              + "  ".join(f"{all_features[i, k]:>14.3f}" for k in picked) + marker)

    if args.coverage:
        print(f"\n{'=' * 78}\nFEATURE COVERAGE (all {len(segments)} segments)\n{'=' * 78}")
        table = coverage_report(aggregates, stats)
        print(table.to_string(float_format=lambda value: f"{value:.4f}"))
        print("\nThe bulk of every feature is O(1); cal_z's tail reaches ~10 (real")
        print("exercise spikes -- state the range, don't imply every value is small).")
        print("hr_missing's mean must match the audit's empty-slot fraction (0.0671).")
