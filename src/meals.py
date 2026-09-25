"""Raw meal rows -> one clean, sorted meal table.

The meal equivalent of `preprocess.py`: read the CSVs once, apply the cleaning rules that
`meal_audit.py` justified, and hand `meal_features.py` a tidy frame it can index by time.

    python src/meals.py              # summary + every verification below
    python src/meals.py --detail     # + per-subject meal counts

Three cleaning decisions. The first two come from `meal_audit.py`; the third was decided
here, from a measurement this module makes:

  1. `Amount Consumed` is DROPPED. It is recorded in three incompatible units across
     subjects (percent / servings / percent-per-serving) and is absent for 028 and 029.
     Rescaling it would need a constant that can only be read off the held-out subject's
     own rows, which is a leak -- unlike the macro caps below, which are one global number
     fit on training subjects.

  2. All-zero macro rows are MISSING, not zero. 43 meals record Carbs=0 and Calories=0 and
     yet raised glucose ~43 mg/dL, so the food was eaten and the log row was simply never
     filled in. Feeding `carbs = 0` there would teach the model that a real meal had no
     carbohydrate. The meal is KEPT (its timestamp is valid, so every timing feature is
     correct), the macros become NaN, and `apply_macro_cleaning` fills them with a
     train-fitted median while `macros_missing` records that the value is a guess.

  3. Macros are capped at FIXED physiological ceilings, not at a fitted p99. The original
     plan was p99 over training subjects. Measuring it (`--cap-stability`) showed the Fiber
     cap swinging 53.3 -> 621.5 g across the 11 seeds, an 11.67x range, because the
     corrupted rows sit in only two subjects (016, 019) and the cap depends on whether they
     landed in train. That would make one feature's scale a function of the seed, in a
     project whose method is paired multi-seed comparison. A fixed ceiling cannot move.
     `cap_method="quantile"` keeps the original behaviour available for comparison.

What this module deliberately does NOT do: winsorize on load. Cleaning constants belong to
a split, so they are produced by `fit_macro_cleaning` and applied separately -- the same
shape as `fit_scaler` / `apply_scaler`.

Design decisions and their rationale are recorded in PROJECT_STATE.md.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import MALFORMED, Segment, find_subject_csvs, load_raw

REPO_ROOT = Path(__file__).resolve().parent.parent

# The four real classes. 'Snacks' is a fourth class the data dictionary never documents,
# and it is 20% of all meals, so it cannot be folded away or dropped.
MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")

MACROS = ("Carbs", "Protein", "Fat", "Fiber", "Calories")

# Divisors that put a typical meal in the same ORDER OF MAGNITUDE as 1. Fixed constants,
# NOT fitted -- a scaler fitted on the meal columns would be one more thing that could leak.
# Measured medians after scaling: Carbs 0.54, Calories 0.46, Protein 0.22, Fat 0.14,
# Fiber 0.04. So "near 1.0" is loose for Fat and plainly wrong for Fiber -- these are round
# numbers picked for interpretability (1.0 = 100 g), not a normalization claim.
MACRO_SCALE = {"Carbs": 100.0, "Protein": 100.0, "Fat": 100.0,
               "Fiber": 100.0, "Calories": 1000.0}

DROPPED_COLUMNS = ("Amount Consumed",)   # see the module docstring, decision 1

# Winsorization ceilings. FIXED constants chosen from physiology, not fitted from the data
# -- see the module docstring, decision 3, and `cap_stability` for the measurement that
# made this the default. A fixed number cannot vary with the split, which is the whole point.
PHYSIOLOGICAL_CAPS = {"Carbs": 300.0, "Protein": 200.0, "Fat": 200.0,
                      "Fiber": 100.0, "Calories": 3000.0}

CAP_METHODS = ("physiological", "quantile")
DEFAULT_CAP_METHOD = "physiological"
DEFAULT_QUANTILE = 0.99                  # only used when cap_method="quantile"

RAW_TIMESTAMP, RAW_MEAL_TYPE = "Timestamp", "Meal Type"

# Regression constants: what a correct load produces. Same idea as preprocess.py's
# 44 subjects / 124,808 readings asserts -- if a number moves, something upstream changed.
EXPECTED_MEALS, EXPECTED_SUBJECTS = 1_689, 44

READING_INTERVAL_MIN = 5                 # Dexcom native sampling, enforced by extract_segments


def normalize_meal_type(raw_types: pd.Series) -> pd.Series:
    """10 raw spellings -> the 4 classes in MEAL_TYPES.

    'Breakfast' -> 'breakfast' (case), 'Snacks' -> 'snack' (trailing s),
    'snack 1' -> 'snack' (trailing counter).
    """
    return (raw_types.astype(str).str.strip().str.lower()
            .str.replace(r"s$|\s*\d+$", "", regex=True))


def load_meals(data_dir: Path) -> pd.DataFrame:
    """Every logged meal for the 44 clean subjects, tidy and sorted by (subject, time).

    Columns: subject, timestamp, type, the 5 macros (raw mg/g -- NOT yet winsorized or
    scaled), and macros_missing.

    Sorting is not cosmetic: `meal_features.py` locates the most recent meal with
    `np.searchsorted`, which returns a plausible wrong answer on unsorted input rather
    than raising.
    """
    per_subject_frames = []

    for subject_id, csv_path in find_subject_csvs(data_dir).items():
        if subject_id in MALFORMED:                 # subject 007, irregular timestamps
            continue

        raw = load_raw(csv_path)

        # Subjects do not share a column schema (013 has an extra Sugar column, 028/029
        # have no Amount Consumed), so state the requirement instead of trusting it.
        required = [RAW_TIMESTAMP, RAW_MEAL_TYPE, *MACROS]
        missing_columns = [c for c in required if c not in raw.columns]
        assert not missing_columns, f"subject {subject_id} lacks columns {missing_columns}"

        meal_rows = raw[raw[RAW_MEAL_TYPE].notna()]

        subject_meals = meal_rows[[RAW_TIMESTAMP, *MACROS]].rename(
            columns={RAW_TIMESTAMP: "timestamp"})
        subject_meals.insert(0, "subject", subject_id)
        subject_meals.insert(2, "type", normalize_meal_type(meal_rows[RAW_MEAL_TYPE]))

        per_subject_frames.append(subject_meals)

    meals = pd.concat(per_subject_frames, ignore_index=True)

    for macro in MACROS:
        meals[macro] = pd.to_numeric(meals[macro], errors="coerce")

    # Decision 2: a row whose macros are ALL zero (or blank) is an unfilled log entry, not
    # a zero-calorie meal. Blank it out so it can never be read as "this meal had no carbs";
    # apply_macro_cleaning fills it later with a train-fitted value.
    macro_values = meals[list(MACROS)]
    meals["macros_missing"] = (macro_values.fillna(0.0) == 0.0).all(axis=1)
    meals.loc[meals["macros_missing"], list(MACROS)] = np.nan

    meals = meals.sort_values(["subject", "timestamp"]).reset_index(drop=True)

    unrecognized = set(meals["type"]) - set(MEAL_TYPES)
    assert not unrecognized, f"unrecognized meal types: {unrecognized}"
    assert len(meals) == EXPECTED_MEALS, f"expected {EXPECTED_MEALS:,} meals, got {len(meals):,}"
    assert meals["subject"].nunique() == EXPECTED_SUBJECTS, (
        f"expected {EXPECTED_SUBJECTS} subjects, got {meals['subject'].nunique()}")
    for subject_id, subject_meals in meals.groupby("subject"):
        assert subject_meals["timestamp"].is_monotonic_increasing, \
            f"subject {subject_id} meals are not sorted by time"

    return meals


@dataclass(frozen=True)
class MacroCleaning:
    """The constants needed to turn raw macros into model input.

    Bundled in one object on purpose: caps and fills must describe the SAME split, and
    passing them separately makes it possible to mix two by accident. Mirrors the
    (mean, std) contract of `preprocess.fit_scaler`.
    """

    caps: dict[str, float]       # winsorize above this, per macro
    fills: dict[str, float]      # value used for meals whose macro record is missing
    method: str                  # how the caps were chosen: "physiological" or "quantile"
    quantile: float | None       # the quantile used, or None for fixed caps
    n_train_meals: int           # how many meals the FILLS were fit on


def fit_macro_cleaning(meals: pd.DataFrame, train_subjects: list[str],
                       cap_method: str = DEFAULT_CAP_METHOD,
                       quantile: float = DEFAULT_QUANTILE) -> MacroCleaning:
    """Choose winsorization caps and fit missing-value fills. Fills are TRAIN-ONLY.

    Two cap methods:

      "physiological" (default) -- the fixed PHYSIOLOGICAL_CAPS. Nothing is fitted, so the
          cap cannot move with the split. Chosen because the quantile caps turned out to be
          badly unstable for Fiber (11.67x across the 11 seeds; run `cap_stability`), which
          made that one feature's scale a function of the seed.

      "quantile" -- p99 over the training subjects, the original plan. Kept so the choice
          stays measurable rather than asserted, and so `cap_stability` can demonstrate it.

    The FILLS are fitted from training subjects either way, so this stays a `fit_` function
    with the same leakage discipline as `fit_scaler`. Meals flagged `macros_missing` are
    excluded from the fill statistics -- they carry no information about the distribution,
    and counting them as zeros would drag every median down.
    """
    assert cap_method in CAP_METHODS, f"cap_method must be one of {CAP_METHODS}"
    assert train_subjects, "no training subjects given"

    is_train = meals["subject"].isin(train_subjects)
    train_meals = meals[is_train & ~meals["macros_missing"]]
    assert not train_meals.empty, "no usable training meals"

    if cap_method == "physiological":
        caps = dict(PHYSIOLOGICAL_CAPS)
        used_quantile = None
    else:
        caps = {macro: float(train_meals[macro].quantile(quantile)) for macro in MACROS}
        used_quantile = quantile

    fills = {macro: float(train_meals[macro].median()) for macro in MACROS}

    for macro in MACROS:
        assert np.isfinite(caps[macro]) and np.isfinite(fills[macro]), \
            f"non-finite constant for {macro}: cap={caps[macro]}, fill={fills[macro]}"
        assert fills[macro] <= caps[macro], \
            f"{macro}: fill {fills[macro]} exceeds cap {caps[macro]}"

    return MacroCleaning(caps=caps, fills=fills, method=cap_method,
                         quantile=used_quantile, n_train_meals=len(train_meals))


def apply_macro_cleaning(meals: pd.DataFrame, cleaning: MacroCleaning) -> pd.DataFrame:
    """Winsorize the real macro values, then fill the missing ones. Returns a new frame.

    Order is clip-then-fill so the fill values (train medians, far below any cap) pass
    through untouched -- filling first would work out to the same numbers, but this way
    the two steps stay independent and each is easy to reason about.
    """
    cleaned = meals.copy()

    for macro in MACROS:
        values = cleaned[macro]
        values = values.clip(upper=cleaning.caps[macro])       # NaNs survive clip()
        cleaned[macro] = values.fillna(cleaning.fills[macro])

    still_missing = cleaned[list(MACROS)].isna().any(axis=1).sum()
    assert still_missing == 0, f"{still_missing} macro values still NaN after cleaning"

    return cleaned


def scale_macros(meals: pd.DataFrame) -> pd.DataFrame:
    """Divide each macro by its fixed MACRO_SCALE divisor so a typical meal is near 1.0.

    Separate from cleaning because the divisors are constants, not fitted values -- there
    is no train/test distinction to get wrong here.
    """
    scaled = meals.copy()
    for macro in MACROS:
        scaled[macro] = scaled[macro] / MACRO_SCALE[macro]
    return scaled


def segment_times(segment: Segment) -> np.ndarray:
    """The timestamp of every reading in a segment, as datetime64.

    No timestamp array needs storing: `extract_segments` only groups readings whose
    consecutive difference is exactly 5 minutes, so within a segment the times are
    `start_time + 5 min * i` by construction. Verified against the raw CSVs by
    `verify_segment_times` below.
    """
    offsets = np.arange(len(segment.glucose)) * READING_INTERVAL_MIN
    return (segment.start_time + pd.to_timedelta(offsets, unit="m")).to_numpy()


def meals_by_subject(meals: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the meal table into one sorted frame per subject, for meal-feature lookups.

    NOTE: deliberately keyed by SUBJECT, not by segment. A meal that happened inside a data
    gap between two segments is still a real meal, and the first window of the next segment
    should know about it. Restricting meals to a segment's own time span would turn all 81
    segment boundaries into fake "nobody has ever eaten" regions.
    """
    return {subject_id: group.reset_index(drop=True)
            for subject_id, group in meals.groupby("subject")}


def cap_stability(meals: pd.DataFrame, seeds: range | list[int],
                  cap_method: str = "quantile",
                  quantile: float = DEFAULT_QUANTILE) -> pd.DataFrame:
    """How much each constant moves across splits. One row per seed, per macro column.

    This is the measurement that decided `DEFAULT_CAP_METHOD`. Fitting a constant per split
    is leak-safe, but it has a cost: if the constant depends on WHICH subjects landed in
    train, that feature's scale changes from seed to seed, and a multi-seed paired
    comparison is then averaging over inputs that were never on the same scale. A small
    max/min ratio means the cost is negligible; Fiber's was 11.67x.

    Defaults to "quantile" because fixed caps are constant by construction -- there would
    be nothing to measure. The `fill_*` columns are fitted under either method, so they are
    always worth checking.
    """
    from dataset import split_subjects

    subjects = sorted(meals["subject"].unique())
    rows = []
    for seed in seeds:
        train_subjects, _, _ = split_subjects(subjects, seed=seed)
        cleaning = fit_macro_cleaning(meals, train_subjects, cap_method, quantile)
        rows.append({"seed": seed, **cleaning.caps,
                     **{f"fill_{macro}": value for macro, value in cleaning.fills.items()},
                     "n_train_meals": cleaning.n_train_meals})
    return pd.DataFrame(rows).set_index("seed")


def spread_summary(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """min / max / max-over-min / coefficient of variation for each column. Reporting helper."""
    rows = []
    for column in columns:
        values = table[column].to_numpy(dtype=float)
        rows.append({"column": column, "min": values.min(), "max": values.max(),
                     "max_over_min": values.max() / values.min() if values.min() else np.inf,
                     "cv": values.std() / values.mean() if values.mean() else np.inf})
    return pd.DataFrame(rows).set_index("column")


def verify_segment_times(segments: list[Segment], data_dir: Path,
                         n_segments: int | None = None) -> int:
    """Check the reconstructed timestamps against the raw CSVs. Returns readings checked.

    Defaults to ALL segments. An earlier version sampled 5, but checking all 81 costs a
    few seconds and turns "the 5-minute spacing holds where we looked" into "it holds
    everywhere", which is what the
    rest of the pipeline actually assumes. If the reconstruction were off by even one
    interval, every meal feature would attach to the wrong reading and nothing downstream
    would complain.

    Pass `n_segments` to sample instead (evenly spread, not the first N -- those would all
    come from the same one or two subjects).
    """
    checked = 0
    raw_by_subject: dict[str, pd.DataFrame] = {}

    if n_segments is None:
        sampled = segments
    else:
        stride = max(1, len(segments) // n_segments)
        sampled = segments[::stride][:n_segments]

    for segment in sampled:
        if segment.subject not in raw_by_subject:
            csv_path = find_subject_csvs(data_dir)[segment.subject]
            raw = load_raw(csv_path).dropna(subset=["Dexcom GL"])
            raw_by_subject[segment.subject] = raw.set_index("Timestamp")["Dexcom GL"]

        raw_glucose = raw_by_subject[segment.subject]
        times = segment_times(segment)

        missing = [t for t in times if t not in raw_glucose.index]
        assert not missing, (f"subject {segment.subject}: {len(missing)} reconstructed "
                             f"timestamps are absent from the raw CSV, first {missing[:1]}")

        from_raw = raw_glucose.reindex(times).to_numpy(dtype=float)
        assert np.array_equal(from_raw, segment.glucose), (
            f"subject {segment.subject}: reconstructed times select the wrong readings")
        checked += len(times)

    return checked


if __name__ == "__main__":
    from preprocess import load_segments
    from dataset import split_subjects

    parser = argparse.ArgumentParser(description="Load and clean the meal table.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "dataset" / "CGMacros",
                    help="directory holding the CGMacros-XXX/ subject folders")
    parser.add_argument("--segments", type=Path, default=REPO_ROOT / "data" / "segments_split.npz",
                    help="B1 segment cache, used to verify the timestamp reconstruction")
    parser.add_argument("--seed", type=int, default=42,
                    help="which split to show the cleaning constants for")
    parser.add_argument("--cap-method", choices=CAP_METHODS, default=DEFAULT_CAP_METHOD,
                    help="fixed physiological ceilings (default) or a fitted p99")
    parser.add_argument("--detail", action="store_true", help="also print per-subject meal counts")
    parser.add_argument("--cap-stability", action="store_true",
                    help="compare both cap methods across the 11 B1 seeds")
    args = parser.parse_args()

    meals = load_meals(args.data_dir)

    print(f"{'=' * 78}\nMEAL TABLE\n{'=' * 78}")
    print(f"{len(meals):,} meals   {meals['subject'].nunique()} subjects   "
          f"{len(meals) / meals['subject'].nunique():.1f} per subject")
    print(f"types: {', '.join(f'{name} {count}' for name, count in meals['type'].value_counts().items())}")
    print(f"span:  {meals['timestamp'].min()}  ->  {meals['timestamp'].max()}")

    n_missing = int(meals["macros_missing"].sum())
    print(f"\nmacros_missing: {n_missing} meals ({n_missing / len(meals):.1%}) had an "
          f"all-zero macro record")
    print(f"  by type: {', '.join(f'{name} {count}' for name, count in meals.loc[meals.macros_missing, 'type'].value_counts().items())}")
    print(f"  subjects affected: {meals.loc[meals.macros_missing, 'subject'].nunique()}")

    print(f"\n{'=' * 78}\nCLEANING CONSTANTS (seed {args.seed}, caps = {args.cap_method})\n{'=' * 78}")
    train_subjects, val_subjects, test_subjects = split_subjects(
        sorted(meals["subject"].unique()), seed=args.seed)
    cleaning = fit_macro_cleaning(meals, train_subjects, cap_method=args.cap_method)

    print(f"{len(train_subjects)} train subjects, {cleaning.n_train_meals:,} usable meals "
          f"(val {len(val_subjects)}, test {len(test_subjects)})")
    print("caps are FIXED constants; only the fills are fitted, on training subjects only\n"
          if cleaning.method == "physiological" else
          f"caps AND fills fitted on training subjects only (p{cleaning.quantile:.0%})\n")

    # The p99 columns are shown even when they are not in use, because the gap between the
    # cap actually applied and the data's own p99 is what a reviewer will ask about.
    train_only = meals[meals["subject"].isin(train_subjects) & ~meals["macros_missing"]]
    print(f"{'macro':<10} {'cap':>9} {'train p99':>10} {'fill':>8} {'clipped':>8}")
    for macro in MACROS:
        n_clipped = int((meals[macro] > cleaning.caps[macro]).sum())
        print(f"{macro:<10} {cleaning.caps[macro]:>9.1f} "
              f"{train_only[macro].quantile(DEFAULT_QUANTILE):>10.1f} "
              f"{cleaning.fills[macro]:>8.1f} {n_clipped:>8}")

    cleaned = apply_macro_cleaning(meals, cleaning)
    print(f"\nafter cleaning: {int(cleaned[list(MACROS)].isna().sum().sum())} NaN remaining, "
          f"max Fiber {cleaned['Fiber'].max():.1f} g (raw max was {meals['Fiber'].max():.1f})")

    scaled = scale_macros(cleaned)
    print(f"\nscaled, median meal (1.0 = 100 g, or 1000 kcal): "
          f"{', '.join(f'{macro} {scaled[macro].median():.2f}' for macro in MACROS)}")
    print("  these are interpretable round divisors, NOT a normalization -- Fiber's median")
    print("  sits at 0.04, so do not describe the features as centred near 1.")

    print(f"\n{'=' * 78}\nVERIFICATION\n{'=' * 78}")
    segments = load_segments(args.segments)
    n_checked = verify_segment_times(segments, args.data_dir)
    print(f"segment timestamps: {n_checked:,} readings across all {len(segments)} segments "
          f"match the raw CSVs exactly")

    by_subject = meals_by_subject(meals)
    assert sum(len(group) for group in by_subject.values()) == len(meals)
    assert all(group["timestamp"].is_monotonic_increasing for group in by_subject.values())
    print(f"meals_by_subject: {len(by_subject)} subjects, all sorted, "
          f"{sum(len(group) for group in by_subject.values()):,} meals total")

    # A meal falling in a data gap belongs to no segment, but must still be reachable.
    segment_subjects = {seg.subject for seg in segments}
    assert segment_subjects == set(by_subject), "segment and meal subject sets disagree"
    print(f"subject sets match between segments and meals ({len(segment_subjects)})")

    if args.cap_stability:
        print(f"\n{'=' * 78}\nCAP STABILITY ACROSS THE 11 B1 SEEDS\n{'=' * 78}")
        print("This is the measurement that made 'physiological' the default.\n")

        quantile_caps = cap_stability(meals, range(42, 53), cap_method="quantile")
        print("p99 caps, one row per seed:")
        print(quantile_caps[list(MACROS)].to_string(float_format=lambda v: f"{v:.1f}"))

        spread = spread_summary(quantile_caps, list(MACROS))
        print(f"\n{'macro':<10} {'min':>9} {'max':>9} {'max/min':>9} {'CV':>7}")
        for macro, row in spread.iterrows():
            print(f"{macro:<10} {row['min']:>9.1f} {row['max']:>9.1f} "
                  f"{row['max_over_min']:>8.2f}x {row['cv']:>7.1%}")

        print("\nEvery macro is stable except Fiber, whose cap swings ~12x. Cause: the")
        print("corruption sits in only subjects 016 and 019, so the cap depends on whether")
        print("they landed in train. That would make one feature's scale a function of the")
        print("seed, in a project whose method is paired multi-seed comparison.")

        print(f"\nFixed physiological caps remove this entirely: "
              f"{', '.join(f'{macro} {cap:.0f}' for macro, cap in PHYSIOLOGICAL_CAPS.items())}")

        # The fills stay fitted under either method, so their spread still matters. Medians
        # are robust statistics, so this should be small -- but check rather than assume.
        fill_columns = [f"fill_{macro}" for macro in MACROS]
        fill_spread = spread_summary(quantile_caps, fill_columns)
        print(f"\nfills (train medians) are fitted under BOTH methods -- their spread:")
        for column, row in fill_spread.iterrows():
            print(f"  {column:<16} {row['min']:>8.1f} -> {row['max']:>8.1f}   "
                  f"{row['max_over_min']:>5.2f}x  CV {row['cv']:>5.1%}")

    if args.detail:
        print(f"\n{'subject':<9} {'meals':>6} {'missing':>8}  types")
        for subject_id, group in meals.groupby("subject"):
            types = ", ".join(f"{name} {count}" for name, count in group["type"].value_counts().items())
            print(f"{subject_id:<9} {len(group):>6} {int(group.macros_missing.sum()):>8}  {types}")

    print("\nMeal cleaning complete. meal_features.py consumes load_meals + meals_by_subject + segment_times.")
