"""Meal state at every CGM reading.

`meals.py` answers "what meals happened?". This module answers "what does the meal state
look like AT reading i?", for every reading in a segment, using only meals at or before it.

    python src/meal_features.py              # build for one subject + run every test
    python src/meal_features.py --tests      # just the correctness tests
    python src/meal_features.py --coverage   # how often each block is non-zero, all subjects

The output is one `(L, K)` matrix per segment, parallel to `segment.glucose`, so the same
`sliding_window_view` machinery already in use works unchanged. The matrix is sliced two
ways: per-timestep channels are `features[i : i+HISTORY]`, and the static vector for a
window is row `i + HISTORY - 1` -- the state at the last input reading.

THE LEAKAGE RULE, which is the whole point of this file:

    last_meal_index = np.searchsorted(meal_times, t, side="right") - 1

At prediction time `t` you forecast `t + PH`, so meals in `(t, t + PH]` are the future and
must be invisible. `side="right"` keeps a meal at exactly `t` (it is not the future);
`- 1` steps back to it; `last_meal_index == -1` means no meal has ever happened yet.
`test_no_future_leak` proves it holds by deleting every later meal and checking the row is
bit-identical.

INPUT CONTRACT: `build_features` takes meals that are ALREADY cleaned and scaled --

    cleaning = fit_macro_cleaning(meals, train_subjects)      # meals.py, train-only
    ready    = scale_macros(apply_macro_cleaning(meals, cleaning))
    features, names = build_features(segment, ready[ready.subject == sid])

Cleaning therefore lives in exactly one place (`meals.py`) and cannot drift. Passing raw
meals is caught by an assert rather than silently zeroing the macros (see design decision 3).

Design decisions (all verified before changing):

  1. NO `len(meal_times) == 0` early return. A naive implementation returns
     `np.zeros((L, K))` when the meal list is empty, but an all-zero row means
     `dt_norm = 0` = "a meal just happened", the exact opposite of the truth. The no-meal
     state is `minutes_since_meal = inf` -> `dt_norm = 1.0`. The empty case now flows
     through the same code path and comes out right by construction.

  2. That bug also broke the leak test. Truncating meals to `<= times[0]` empties the
     frame, so the naive version produced 0.0 where the full build produced 1.0 and
     `test_no_future_leak` failed spuriously at i=0. Fixing (1) fixes the test.

  3. NO `.fillna(0.0)` on the macro columns. `load_meals` deliberately stores NaN for the 43
     meals whose macro record was never filled in; `fillna(0.0)` would turn them straight
     back into "this meal had no carbohydrate" and undo the cleaning fix. This module asserts
     the macros are finite instead.

  4. NUTRITION IS FORCED TO ZERO PAST 5 H OF FASTING. The undecayed macros (block M's raw
     half) and heavy kernel tails never decay to zero on their own. Measured over all 124,808
     readings: 38.5% are fasting, and on those rows the undecayed `carbs` averaged 0.504
     (non-zero on 95.1% of them) and `impulse_120` 0.143. Worse, 14% of the variance in
     fasting `carbs` sits BETWEEN subjects -- during a long fast those features partly encode
     WHO the subject is, which is the exact failure the fasting negative control is meant to
     detect. Past `FASTING_CUTOFF_MIN` every nutrition column is zeroed, the timing block is
     left alone. Cost is near zero -- the ridge oracle's fasting delta was +0.082, so meal
     features contributed nothing there anyway.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from meals import (MACROS, MEAL_TYPES, apply_macro_cleaning, fit_macro_cleaning,
                   load_meals, meals_by_subject, scale_macros, segment_times)
from preprocess import Segment

REPO_ROOT = Path(__file__).resolve().parent.parent

# Feature blocks, named so the ablation ladder is a set union rather than a rewrite.
# M is a SUPERSET of C (both carry Carbs), so a block list must never contain both.
BLOCK_SIZES = {"T": 3, "C": 2, "M": 10, "Y": 4, "I": 3, "S": 1}
DEFAULT_BLOCKS = ("T", "M", "Y", "I", "S")

# Block T only. These describe WHEN the last meal was, not what was in it, so they stay
# live during a fast (that is their job) and are exempt from the fasting gate below.
TIMING_FEATURES = frozenset({"dt_norm", "postprandial", "decay"})

# Design constants. Be honest about which are grounded and which are conventions, because a
# reviewer will ask and "it was in a tutorial" is not an answer:
#   POSTPRANDIAL_MIN  GROUNDED -- 2 h is the clinical postprandial window (ADA/IDF targets,
#                     the 2-hour OGTT).
#   DT_CLIP_MIN       DERIVED -- see `cutoff_evidence` / `--why-cutoff`. Also the fasting gate.
#   DECAY_TAU_MIN     CONVENTION -- a 1-hour exponential is a plausible recency shape, NOT
#                     fitted to this data. It only reshapes a feature (it never zeroes
#                     anything), so the stakes are far lower than the gate; the network can
#                     reweight it. Do not describe it as measured.
#   IMPULSE_PEAKS     CONVENTION -- three fixed timescales instead of a per-patient fitted
#                     absorption curve, so the network can learn its own mixture. The
#                     literature this is simplified from fits these curves per subject.
DECAY_TAU_MIN = 60.0            # decay = exp(-dt / 60): 1.0 at the meal, 0.37 an hour later
DT_CLIP_MIN = 300.0             # time-since-meal saturates at 5 h
POSTPRANDIAL_MIN = 120.0        # "am I digesting" flag
IMPULSE_PEAKS = (30.0, 60.0, 120.0)     # carb-absorption timescales, minutes

# Beyond this, every NUTRITION feature is forced to 0 -- see design decision 4. The timing block
# is exempt: it still reports dt_norm = 1.0 (saturated) and decay = 0, which is how a
# fasting row stays distinguishable from a row where a meal just happened.
FASTING_CUTOFF_MIN = DT_CLIP_MIN

READING_INTERVAL_MIN = 5


def gamma_kernel(dt_min: np.ndarray, t_peak: float) -> np.ndarray:
    """Crude carbohydrate appearance curve: 0 at the meal, peaks at `t_peak`, then decays.

    `t * exp(1 - t)` where `t = dt / t_peak`, so the peak value is exactly 1.0 at
    `dt == t_peak` (verified). Negative `dt` means the meal is in the FUTURE relative to this
    reading and returns 0 -- that is this function's share of the leakage rule.
    """
    scaled = np.clip(dt_min, 0, None) / t_peak
    return np.where(dt_min < 0, 0.0, scaled * np.exp(1.0 - scaled))


def block_names(blocks: tuple[str, ...] = DEFAULT_BLOCKS) -> list[str]:
    """Column names for a block list, in the same fixed order `build_features` emits.

    Kept separate so callers can size a feature matrix, or label an ablation table, without
    building one. The two are checked against each other in `test_names_match_matrix`.
    """
    names: list[str] = []
    for block in blocks:
        if block == "T":
            names += ["dt_norm", "postprandial", "decay"]
        elif block == "C":
            names += ["carbs", "carbs_x_decay"]
        elif block == "M":
            for macro in MACROS:
                names += [macro.lower(), f"{macro.lower()}_x_decay"]
        elif block == "Y":
            names += [f"type_{meal_type}" for meal_type in MEAL_TYPES]
        elif block == "I":
            names += [f"impulse_{int(peak)}" for peak in IMPULSE_PEAKS]
        elif block == "S":
            names += ["carb_bin"]
        else:
            raise ValueError(f"unknown block {block!r}; known: {sorted(BLOCK_SIZES)}")
    return names


def build_features(segment: Segment, meals: pd.DataFrame,
                   blocks: tuple[str, ...] = DEFAULT_BLOCKS) -> tuple[np.ndarray, list[str]]:
    """(L, K) meal-state matrix for one segment. Row i uses ONLY meals at or before reading i.

    `meals` must be this subject's rows, already cleaned and scaled (see the module
    docstring), sorted by timestamp. An empty frame is valid and yields the correct
    "nobody has ever eaten" state, not zeros.
    """
    assert "C" not in blocks or "M" not in blocks, \
        "blocks C and M both carry Carbs -- use one or the other, never both"
    for block in blocks:
        if block not in BLOCK_SIZES:
            raise ValueError(f"unknown block {block!r}; known: {sorted(BLOCK_SIZES)}")

    n_readings = len(segment.glucose)
    times = segment_times(segment)
    meal_times = meals["timestamp"].to_numpy()

    assert pd.Series(meal_times).is_monotonic_increasing, \
        "meals must be sorted by timestamp -- np.searchsorted is silently wrong otherwise"

    # --- the leakage rule -------------------------------------------------------------
    # For each reading, the index of the most recent meal at or before it (-1 = none yet).
    last_meal = np.searchsorted(meal_times, times, side="right") - 1
    has_meal = last_meal >= 0

    # Minutes since that meal. inf where no meal has happened yet, which every block below
    # handles explicitly -- there is no early return for the empty case (design decision 1).
    minutes_since = np.full(n_readings, np.inf)
    if has_meal.any():
        elapsed = times[has_meal] - meal_times[last_meal[has_meal]]
        minutes_since[has_meal] = elapsed / np.timedelta64(1, "m")
    assert (minutes_since >= 0).all(), "negative time-since-meal -- searchsorted is wrong"

    # exp(-inf/60) would be 0.0 anyway, but numpy warns on inf arithmetic, so substitute a
    # large finite number and let it underflow to 0 quietly.
    decay = np.exp(-np.where(np.isfinite(minutes_since), minutes_since, 1e6) / DECAY_TAU_MIN)

    def at_last_meal(values: np.ndarray) -> np.ndarray:
        """Broadcast a per-MEAL array to a per-READING array, using the most recent meal.

        Readings before the subject's first meal keep 0.0, which is the right neutral value
        for every block that uses this (macros, one-hots): no meal means no nutrition.
        """
        out = np.zeros(n_readings)
        out[has_meal] = values[last_meal[has_meal]]
        return out

    # --- macros ------------------------------------------------------------------------
    macro_values = {}
    for macro in MACROS:
        values = meals[macro].to_numpy(dtype=float)
        # Design decision 3: NOT fillna(0.0). load_meals stores NaN for the 43 unfilled
        # records and apply_macro_cleaning replaces them with a train median; a NaN here
        # means the caller skipped that step, and zeroing it would silently undo the fix.
        assert np.isfinite(values).all() if len(values) else True, (
            f"{macro} contains NaN/inf -- pass meals through apply_macro_cleaning() first "
            "(see the input contract in this module's docstring)")
        macro_values[macro] = values

    columns: list[np.ndarray] = []

    for block in blocks:
        if block == "T":
            columns += [
                np.minimum(minutes_since, DT_CLIP_MIN) / DT_CLIP_MIN,   # inf -> 1.0
                (minutes_since <= POSTPRANDIAL_MIN).astype(float),      # inf -> 0.0
                decay,
            ]

        elif block == "C":
            carbs = at_last_meal(macro_values["Carbs"])
            columns += [carbs, carbs * decay]

        elif block == "M":
            for macro in MACROS:
                value = at_last_meal(macro_values[macro])
                columns += [value, value * decay]

        elif block == "Y":
            for meal_type in MEAL_TYPES:
                is_type = (meals["type"] == meal_type).to_numpy(dtype=float)
                # x decay so "I ate breakfast" fades instead of persisting all day
                columns.append(at_last_meal(is_type) * decay)

        elif block == "I":
            # Sum over ALL past meals, not just the most recent -- this is the block that
            # handles the 99 meal pairs less than 30 min apart. (L, n_meals) is a few hundred
            # KB for one subject; do NOT build this for every subject at once.
            #
            # NOTE ON SCALE: because this SUMS over meals it is the only block that can
            # exceed the per-meal cap. Measured cohort-wide the impulse columns reach 3.6 /
            # 4.8 / 5.6 while every other feature stays <= 3.0. Their standard deviations
            # (0.23-0.43) are in line with the rest, so it is the tail that differs, not the
            # bulk -- but state the range rather than implying every feature is O(1).
            if len(meal_times):
                offsets = (times[:, None] - meal_times[None, :]) / np.timedelta64(1, "m")
                # Future meals -> -1 so gamma_kernel returns 0 for them.
                offsets = np.where(offsets >= 0, offsets, -1.0)
                for peak in IMPULSE_PEAKS:
                    weights = gamma_kernel(offsets, peak)
                    columns.append((weights * macro_values["Carbs"][None, :]).sum(axis=1))
            else:
                columns += [np.zeros(n_readings) for _ in IMPULSE_PEAKS]

        elif block == "S":
            # Grams of carbohydrate landing in the 5-min bin ENDING at each reading, i.e.
            # each meal is attributed to the first reading at or after it. The only block
            # that is genuinely per-timestep; it is what makes the "channels" encoding
            # mean anything.
            carb_bin = np.zeros(n_readings)
            if len(meal_times):
                bin_index = np.searchsorted(times, meal_times, side="left")
                # A meal BEFORE times[0] also yields bin_index == 0, but it did not land in
                # any bin of this segment -- without this guard every meal from the data gap
                # preceding the segment is dumped onto reading 0. Blocks T/M/Y/I still see
                # those meals through minutes_since, which is the correct treatment.
                inside = (meal_times >= times[0]) & (bin_index < n_readings)
                # add.at accumulates instead of overwriting, so two meals in one bin sum.
                np.add.at(carb_bin, bin_index[inside], macro_values["Carbs"][inside])
            columns.append(carb_bin)

    features = np.column_stack(columns)
    names = block_names(blocks)

    # --- fasting gate: force nutrition to zero during long fasts -----------------------
    # The undecayed macros (block M's raw half) and impulse_120 (a heavy kernel tail) stay
    # non-zero indefinitely after a meal. Measured over all 124,808 readings: 38.5% are
    # fasting, and on those rows `carbs` averaged 0.504 with 14% of its variance sitting
    # BETWEEN subjects -- i.e. during a long fast the features partly encode WHO this is.
    # That is exactly the failure the fasting negative control exists to detect, so zeroing
    # them keeps that control honest. The cost is near zero: the ridge oracle's fasting
    # delta was +0.082, so meal features were already contributing nothing on these rows.
    # `>=`, not `>`: dt_norm saturates at dt >= 300, so gating at `>` would leave 151
    # readings (0.12%) whose timing block reads "fasting" while their nutrition columns are
    # still live. Using `>=` makes the invariant an exact biconditional that is easy to
    # state and to test -- dt_norm == 1.0 if and only if every nutrition column is 0.
    fasting = minutes_since >= FASTING_CUTOFF_MIN
    nutrition = [k for k, name in enumerate(names) if name not in TIMING_FEATURES]
    if fasting.any() and nutrition:
        features[np.ix_(fasting, nutrition)] = 0.0

    assert np.isfinite(features).all(), "non-finite meal feature -- an inf was not clipped"
    assert features.shape == (n_readings, len(names)), \
        f"shape {features.shape} does not match {len(names)} names"
    return features, names


def build_all_features(segments: list[Segment], meals_ready: pd.DataFrame,
                       blocks: tuple[str, ...] = DEFAULT_BLOCKS) -> list[np.ndarray]:
    """One (L, K) matrix per segment, in the same order as `segments`.

    `meals_ready` is the whole cleaned+scaled table; it is grouped by subject once here
    rather than per segment, because 81 segments share 44 subjects.
    """
    by_subject = meals_by_subject(meals_ready)
    empty = meals_ready.iloc[:0]
    return [build_features(segment, by_subject.get(segment.subject, empty), blocks)[0]
            for segment in segments]


def _partial_corr(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]) -> float:
    """corr(x, y) after regressing BOTH on `controls` (an intercept is added automatically).

    Answers "what does x tell us about y that the controls do not already?" -- which is
    exactly B2's question when the controls are what the CGM model can already see.
    """
    design = np.column_stack([np.ones(len(x))] + list(controls))
    residual_x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    residual_y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def cutoff_evidence(segments: list[Segment], meals_ready: pd.DataFrame,
                    horizon_steps: int = 6,
                    trend_steps: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The measurement behind FASTING_CUTOFF_MIN. Returns (per-band, cumulative) tables.

    For every reading it collects: minutes since the last meal, that meal's carbs, the
    current glucose level, the recent trend, and the next 30 min of glucose change. Then it
    correlates carbs with that change -- raw, and PARTIAL on (level, trend).

    The partial version is the one that decides the cutoff. Raw correlation cannot separate
    "the meal is still being absorbed" from "glucose is high so it is coming back down", and
    the CGM history already carries the second. Measured on the real data the raw number has
    already gone negative by 90-120 min while the partial number is still +0.082 at 60-90 --
    so picking the cutoff from raw correlation would gate away real meal signal.

    Result: residual partial r beyond the cutoff is -0.010 at 4 h and +0.001 at 5 h, so 5 h
    is where the last of the meal information is genuinely gone.
    """
    by_subject = meals_by_subject(meals_ready)
    records = []

    for segment in segments:
        subject_meals = by_subject.get(segment.subject)
        if subject_meals is None or subject_meals.empty:
            continue
        times = segment_times(segment)
        meal_times = subject_meals["timestamp"].to_numpy()
        glucose = segment.glucose
        n_readings = len(glucose)

        last_meal = np.searchsorted(meal_times, times, side="right") - 1
        has_meal = last_meal >= 0
        minutes_since = np.full(n_readings, np.inf)
        if has_meal.any():
            minutes_since[has_meal] = (
                (times[has_meal] - meal_times[last_meal[has_meal]]) / np.timedelta64(1, "m"))
        carbs = np.zeros(n_readings)
        carbs[has_meal] = subject_meals["Carbs"].to_numpy()[last_meal[has_meal]]

        # Needs `trend_steps` of history behind it and `horizon_steps` of future ahead.
        for i in range(trend_steps, n_readings - horizon_steps):
            records.append((minutes_since[i], carbs[i], glucose[i],
                            glucose[i] - glucose[i - trend_steps],
                            glucose[i + horizon_steps] - glucose[i]))

    elapsed, carbs, level, trend, delta = np.asarray(records, dtype=float).T

    bands = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 180),
             (180, 240), (240, 300), (300, 360), (360, 480), (480, np.inf)]
    per_band = []
    for low, high in bands:
        mask = (elapsed >= low) & (elapsed < high)
        if mask.sum() < 200:
            continue
        per_band.append({
            "band": f"{low:.0f}-{high:.0f} min" if np.isfinite(high) else f"{low:.0f}+ min",
            "n": int(mask.sum()),
            "raw_r": float(np.corrcoef(carbs[mask], delta[mask])[0, 1]),
            "partial_r": _partial_corr(carbs[mask], delta[mask], [level[mask], trend[mask]]),
        })

    cumulative = []
    for cutoff in (90, 120, 180, 240, 300, 360):
        mask = elapsed >= cutoff
        residual = _partial_corr(carbs[mask], delta[mask], [level[mask], trend[mask]])
        # A correlation's standard error is ~1/sqrt(n-3). At n ~ 50,000 that is ~0.0045, so
        # anything under ~0.01 is noise -- WITHOUT this column the table invites reading a
        # difference between 0.003 and 0.009 as meaningful when it is not.
        stderr = 1.0 / np.sqrt(mask.sum() - 3)
        cumulative.append({
            "cutoff_min": cutoff,
            "hours": cutoff / 60.0,
            "n_beyond": int(mask.sum()),
            "residual_partial_r": residual,
            "stderr": stderr,
            "n_stderr": abs(residual) / stderr,
        })

    return (pd.DataFrame(per_band).set_index("band"),
            pd.DataFrame(cumulative).set_index("cutoff_min"))


# ---------------------------------------------------------------------------------------
# Tests. These run by default -- the leak test in particular is the one thing in this
# module that must never silently regress.
# ---------------------------------------------------------------------------------------

def test_no_future_leak(segment: Segment, meals: pd.DataFrame,
                        blocks: tuple[str, ...] = DEFAULT_BLOCKS) -> None:
    """Deleting every meal after t_i must leave row i bit-identical.

    Catches all three ways this goes wrong: merge_asof(direction="nearest"), an off-by-one
    in searchsorted, and a side="left" typo.
    """
    full, _ = build_features(segment, meals, blocks)
    n_readings = len(segment.glucose)
    times = segment_times(segment)

    checked = 0
    for i in (0, n_readings // 3, n_readings // 2, n_readings - 1):
        cutoff = pd.Timestamp(times[i])
        truncated = meals[meals["timestamp"] <= cutoff]
        partial, _ = build_features(segment, truncated, blocks)
        assert np.allclose(full[i], partial[i]), (
            f"row {i} changed when future meals were removed -- FUTURE LEAK\n"
            f"  full:      {full[i]}\n  truncated: {partial[i]}")
        checked += 1

    # Stronger version: drop every meal after the segment starts. Rows whose state depends
    # only on earlier meals must be untouched.
    before_only = meals[meals["timestamp"] < pd.Timestamp(times[0])]
    partial, _ = build_features(segment, before_only, blocks)
    first_meal_in = np.searchsorted(meals["timestamp"].to_numpy(), times[0], side="left")
    if first_meal_in < len(meals):
        first_affected = np.searchsorted(
            times, meals["timestamp"].to_numpy()[first_meal_in], side="left")
        if first_affected > 0:
            assert np.allclose(full[:first_affected], partial[:first_affected]), \
                "rows before the first in-segment meal changed -- FUTURE LEAK"
            checked += int(first_affected)

    print(f"  no-future-leak ......... PASS ({checked} rows verified)")


def test_fasting_rows_are_neutral(segment: Segment, meals: pd.DataFrame) -> None:
    """A row with no meal for >5 h must have ~0 in every nutrition column.

    The cheap invariant: if the meal blocks are non-zero during fasting, something is
    carrying meal information into windows that should not have any.
    """
    features, names = build_features(segment, meals)
    times = segment_times(segment)
    meal_times = meals["timestamp"].to_numpy()

    last_meal = np.searchsorted(meal_times, times, side="right") - 1
    minutes_since = np.full(len(times), np.inf)
    has_meal = last_meal >= 0
    if has_meal.any():
        minutes_since[has_meal] = (
            (times[has_meal] - meal_times[last_meal[has_meal]]) / np.timedelta64(1, "m"))

    fasting = minutes_since >= FASTING_CUTOFF_MIN
    if not fasting.any():
        print("  fasting-rows-neutral ... SKIP (no fasting row in this segment)")
        return

    nutrition = [k for k, name in enumerate(names) if name not in TIMING_FEATURES]
    worst = np.abs(features[np.ix_(fasting, nutrition)]).max()
    assert worst == 0.0, f"fasting rows carry nutrition signal, max |value| = {worst}"

    # The timing block must still be live, or a fasting row would be indistinguishable from
    # a row where a meal just happened -- the bug that a naive all-zero early return causes.
    dt_norm = features[fasting, names.index("dt_norm")]
    assert np.allclose(dt_norm, 1.0), f"fasting dt_norm should be 1.0, got {dt_norm.min()}"

    # decay is exp(-dt/60), so it approaches 0 asymptotically rather than reaching it: at
    # the 5 h cutoff it is exp(-5) = 0.0067. Bound it by that value rather than by 0.
    decay_ceiling = np.exp(-FASTING_CUTOFF_MIN / DECAY_TAU_MIN)
    worst_decay = features[fasting, names.index("decay")].max()
    assert worst_decay <= decay_ceiling + 1e-12, \
        f"fasting decay {worst_decay} exceeds exp(-cutoff/tau) = {decay_ceiling}"

    print(f"  fasting-rows-neutral ... PASS ({int(fasting.sum())} fasting rows, "
          f"all nutrition exactly 0, dt_norm saturated at 1.0)")


def test_fasting_gate_is_biconditional(segment: Segment, meals: pd.DataFrame) -> None:
    """`dt_norm == 1.0` must hold EXACTLY when every nutrition column is 0, and vice versa.

    Guards the `>=` vs `>` boundary. Gating at `>` left 151 cohort readings (0.12%) whose
    timing block read "fasting" while their nutrition was still live -- invisible in a
    single-segment test, and exactly the kind of edge that survives into a paper.
    """
    features, names = build_features(segment, meals)
    saturated = features[:, names.index("dt_norm")] >= 1.0

    nutrition = [k for k, name in enumerate(names) if name not in TIMING_FEATURES]
    all_zero = (features[:, nutrition] == 0.0).all(axis=1)

    # One direction only: a postprandial row CAN legitimately be all-zero (a meal with no
    # recorded carbs at reading 0, say), so `all_zero -> saturated` is not required.
    violations = int((saturated & ~all_zero).sum())
    assert violations == 0, (
        f"{violations} rows have dt_norm == 1.0 but non-zero nutrition -- the fasting gate "
        "and dt_norm disagree at the boundary")
    print(f"  fasting-gate-exact ..... PASS (dt_norm==1.0 -> nutrition==0 on all "
          f"{int(saturated.sum())} saturated rows)")


def test_empty_meals_is_fasting(segment: Segment, meals: pd.DataFrame) -> None:
    """A subject with NO meals must look like permanent fasting, not like a fresh meal.

    Validates design decision 1: a naive all-zero early return would read as dt_norm = 0 =
    "a meal just happened" -- the exact opposite of the truth.
    """
    features, names = build_features(segment, meals.iloc[:0])

    assert np.allclose(features[:, names.index("dt_norm")], 1.0), \
        "no meals should give dt_norm = 1.0 (saturated), not 0.0"
    assert np.allclose(features[:, names.index("postprandial")], 0.0)
    assert np.allclose(features[:, names.index("decay")], 0.0)

    nutrition = [k for k, name in enumerate(names) if name not in ("dt_norm",)]
    assert np.allclose(features[:, nutrition], 0.0), "no meals should give no nutrition"
    print("  empty-meals-is-fasting . PASS (dt_norm saturates at 1.0, not 0.0)")


def test_raw_macros_rejected(segment: Segment, raw_meals: pd.DataFrame) -> None:
    """Passing uncleaned meals must fail loudly, not silently zero the macros."""
    has_nan = raw_meals[list(MACROS)].isna().any().any()
    if not has_nan:
        print("  raw-macros-rejected .... SKIP (this subject has no unfilled macro rows)")
        return
    try:
        build_features(segment, raw_meals)
    except AssertionError:
        print("  raw-macros-rejected .... PASS (NaN macros raise instead of becoming 0.0)")
        return
    raise AssertionError("build_features accepted NaN macros -- the cleaning contract is bypassed")


def test_names_match_matrix(segment: Segment, meals: pd.DataFrame) -> None:
    """Every block's declared size must match the columns it actually emits."""
    for block in sorted(BLOCK_SIZES):
        features, names = build_features(segment, meals, blocks=(block,))
        assert features.shape[1] == BLOCK_SIZES[block], (
            f"block {block}: BLOCK_SIZES says {BLOCK_SIZES[block]}, got {features.shape[1]}")
        assert len(names) == BLOCK_SIZES[block]
        assert names == block_names((block,))
    combined, names = build_features(segment, meals, DEFAULT_BLOCKS)
    assert combined.shape[1] == sum(BLOCK_SIZES[b] for b in DEFAULT_BLOCKS)
    print(f"  names-match-matrix ..... PASS (all 6 blocks; default K = {combined.shape[1]})")


def test_impulse_handles_overlap(segment: Segment, meals: pd.DataFrame) -> None:
    """Two meals 20 min apart must give a larger impulse than either alone.

    The reason block I exists: 99 meal pairs are <30 min apart, so "the last meal" is not a
    complete description of the state.
    """
    if len(meals) < 2:
        print("  impulse-handles-overlap  SKIP (subject has <2 meals)")
        return

    base = pd.Timestamp(segment_times(segment)[0]) + pd.Timedelta(minutes=10)
    template = meals.iloc[[0]].copy()

    def synth(offsets_min):
        rows = []
        for offset in offsets_min:
            row = template.copy()
            row["timestamp"] = base + pd.Timedelta(minutes=offset)
            row["Carbs"] = 0.5
            rows.append(row)
        return pd.concat(rows, ignore_index=True).sort_values("timestamp")

    one, names = build_features(segment, synth([0]), blocks=("I",))
    two, _ = build_features(segment, synth([0, 20]), blocks=("I",))

    column = names.index("impulse_60")
    assert two[:, column].max() > one[:, column].max() * 1.2, (
        "two overlapping meals did not raise the impulse above one meal -- "
        "block I is not summing over past meals")
    print(f"  impulse-handles-overlap  PASS (1 meal {one[:, column].max():.3f} -> "
          f"2 meals {two[:, column].max():.3f})")


def test_gamma_kernel_shape() -> None:
    """The kernel must peak at exactly t_peak, be 0 at the meal, and 0 for future meals."""
    for peak in IMPULSE_PEAKS:
        grid = np.linspace(0, 600, 6001)
        values = gamma_kernel(grid, peak)
        assert np.isclose(values.max(), 1.0, atol=1e-6), f"peak value {values.max()}"
        assert abs(grid[values.argmax()] - peak) < 0.2, "peak in the wrong place"
        assert gamma_kernel(np.array([0.0]), peak)[0] == 0.0, "non-zero at the meal itself"
        assert gamma_kernel(np.array([-30.0]), peak)[0] == 0.0, "FUTURE MEAL LEAKED"
    print("  gamma-kernel-shape ..... PASS (peaks at t_peak, 0 at dt=0, 0 for future)")


def run_tests(segment: Segment, ready_meals: pd.DataFrame, raw_meals: pd.DataFrame) -> None:
    print("tests:")
    test_gamma_kernel_shape()
    test_names_match_matrix(segment, ready_meals)
    test_no_future_leak(segment, ready_meals)
    test_fasting_rows_are_neutral(segment, ready_meals)
    test_fasting_gate_is_biconditional(segment, ready_meals)
    test_empty_meals_is_fasting(segment, ready_meals)
    test_raw_macros_rejected(segment, raw_meals)
    test_impulse_handles_overlap(segment, ready_meals)


def coverage_report(segments: list[Segment], ready_meals: pd.DataFrame) -> pd.DataFrame:
    """How often each feature is non-zero across every reading. A sanity check on sparsity.

    Expect block S to be near-zero almost everywhere (one reading per meal) and the timing
    block to be active nearly always -- if S were dense, meals would be landing in the wrong
    bins.
    """
    matrices = build_all_features(segments, ready_meals)
    stacked = np.vstack(matrices)
    names = block_names(DEFAULT_BLOCKS)
    return pd.DataFrame({
        "feature": names,
        "nonzero_share": (np.abs(stacked) > 1e-9).mean(axis=0),
        "mean": stacked.mean(axis=0),
        "max": stacked.max(axis=0),
    }).set_index("feature")


if __name__ == "__main__":
    from preprocess import load_segments
    from dataset import split_subjects

    ap = argparse.ArgumentParser(description="Build meal-state features for CGM readings.")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "dataset" / "CGMacros")
    ap.add_argument("--segments", type=Path, default=REPO_ROOT / "data" / "segments_split.npz")
    ap.add_argument("--seed", type=int, default=42, help="split whose cleaning constants to use")
    ap.add_argument("--subject", default=None, help="subject to demo (default: first with meals)")
    ap.add_argument("--tests", action="store_true", help="run the tests and stop")
    ap.add_argument("--coverage", action="store_true", help="non-zero share of every feature")
    ap.add_argument("--why-cutoff", action="store_true",
                    help="the partial-correlation evidence behind FASTING_CUTOFF_MIN")
    args = ap.parse_args()

    raw_meals = load_meals(args.data_dir)
    segments = load_segments(args.segments)

    # The cleaning contract: fit on training subjects only, then apply, then scale.
    train_subjects, _, _ = split_subjects(sorted(raw_meals["subject"].unique()), seed=args.seed)
    cleaning = fit_macro_cleaning(raw_meals, train_subjects)
    ready_meals = scale_macros(apply_macro_cleaning(raw_meals, cleaning))

    subject = args.subject or segments[0].subject
    subject_meals = ready_meals[ready_meals["subject"] == subject].reset_index(drop=True)
    subject_raw = raw_meals[raw_meals["subject"] == subject].reset_index(drop=True)
    segment = next(s for s in segments if s.subject == subject)

    print(f"{'=' * 78}\nMEAL FEATURES (subject {subject}, seed {args.seed})\n{'=' * 78}")
    features, names = build_features(segment, subject_meals)
    print(f"segment: {len(segment.glucose)} readings, {len(subject_meals)} meals for subject")
    print(f"matrix:  {features.shape}   blocks {DEFAULT_BLOCKS}")

    run_tests(segment, subject_meals, subject_raw)

    if args.tests:
        raise SystemExit(0)

    # Show the state around the first meal that falls inside this segment: the clearest
    # possible demonstration that the features switch on at the meal and not before.
    times = segment_times(segment)
    inside = subject_meals[(subject_meals["timestamp"] >= times[0])
                           & (subject_meals["timestamp"] <= times[-1])]
    if not inside.empty:
        first = inside.iloc[0]
        at = int(np.searchsorted(times, first["timestamp"].to_datetime64(), side="left"))
        show = ["dt_norm", "postprandial", "decay", "carbs", "carbs_x_decay",
                f"type_{first['type']}", "impulse_60", "carb_bin"]
        picked = [names.index(n) for n in show]

        print(f"\n{'=' * 78}\nAROUND THE FIRST MEAL IN THE SEGMENT\n{'=' * 78}")
        print(f"{first['type']} at {first['timestamp']}, "
              f"carbs {first['Carbs']:.2f} (scaled)\n")
        print("reading  time    " + "  ".join(f"{n:>13}" for n in show))
        for i in range(max(0, at - 2), min(len(times), at + 5)):
            marker = " <- meal lands here" if i == at else ""
            print(f"{i:>7}  {str(times[i])[11:16]}  "
                  + "  ".join(f"{features[i, k]:>13.3f}" for k in picked) + marker)
        print("\nEvery column is 0 (and dt_norm saturated) before the meal, then switches on.")

    if args.why_cutoff:
        print(f"\n{'=' * 78}\nWHY {FASTING_CUTOFF_MIN:.0f} MIN? "
              f"(evidence behind FASTING_CUTOFF_MIN)\n{'=' * 78}")
        print("corr(last-meal carbs, next 30-min glucose change).")
        print("PARTIAL controls for current level + 15-min trend -- i.e. what the meal adds")
        print("BEYOND what the CGM already shows, which is the question B2 actually asks.\n")

        per_band, cumulative = cutoff_evidence(segments, ready_meals)
        print(per_band.to_string(float_format=lambda v: f"{v:+.3f}"))
        print("\nNote 60-90 min: raw says +0.037 (looks dead), partial says +0.082 (alive).")
        print("Mean reversion MASKS real meal signal, so a raw-correlation cutoff gates too early.")

        print(f"\nResidual meal information BEYOND each candidate cutoff:")
        print(cumulative.to_string(float_format=lambda v: f"{v:.4f}"))
        print("\nRead the n_stderr column, NOT the raw r. A correlation's SE is ~1/sqrt(n),")
        print("so at n ~ 50,000 anything under ~0.01 is indistinguishable from zero.")
        print("  2-3 h: residual is 4-6 SE from zero -> real signal still there, too early")
        print("  4-6 h: all within ~3 SE, r^2 < 0.02% -> the data does NOT discriminate here")
        print("So the data sets a FLOOR of 4 h and is indifferent above it. 5 h is chosen as")
        print("the conservative end (deletes less real signal) and because it is already")
        print("DT_CLIP_MIN, so one constant serves dt_norm, the gate, and the stratum.")
        print("Literature brackets this: clinical postprandial ~2 h, carb-absorption")
        print("durations 3-6 h, physiological meal models (Dalla Man / UVA-Padova) 4-6 h.")

    if args.coverage:
        print(f"\n{'=' * 78}\nFEATURE COVERAGE (all {len(segments)} segments)\n{'=' * 78}")
        table = coverage_report(segments, ready_meals)
        print(table.to_string(float_format=lambda v: f"{v:.4f}"))
        print("\ncarb_bin should be non-zero on roughly one reading per meal; the timing")
        print("block should be active nearly everywhere.")

    print("\nDone. These matrices are sliced two ways (channels / static) for model input.")
