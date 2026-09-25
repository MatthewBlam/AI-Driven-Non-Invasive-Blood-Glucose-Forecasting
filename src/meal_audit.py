"""Audit the raw meal columns before building any meal features.

Read-only inspection of the 1,689 meal rows: column schemas, macro outliers, Amount
Consumed unit inconsistencies, meal timing, and (optionally) a glucose excursion check.
audit.py flags that Amount Consumed exceeds its range and Meal Type casing is
inconsistent -- the real problems are worse.

    python src/meal_audit.py                     # the six report blocks
    python src/meal_audit.py --detail            # + every subject's Amount Consumed row
    python src/meal_audit.py --csv results/meal_audit_amount_consumed.csv

This script ASSERTS NOTHING. It is for looking. The cleaning rules it justifies live in
``src/meals.py``; the numbers it prints are recorded in ``PROJECT_STATE.md`` so a future
run can be diffed against them.

Design decisions
~~~~~~~~~~~~~~~~
- Subjects do not share a column schema, so a plain ``pd.concat`` fills a missing column
  with NaN and makes "this subject never had the column" look identical to "this subject
  left the field blank". Column PRESENCE is therefore recorded per subject before
  concatenating, and reported separately from NaN counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import MALFORMED, find_subject_csvs, load_raw

REPO_ROOT = Path(__file__).resolve().parent.parent

MACROS = ("Carbs", "Protein", "Fat", "Fiber", "Calories")
AMOUNT = "Amount Consumed"
MEAL_TYPE = "Meal Type"

# Gap buckets for consecutive meals, in minutes. <30 is the one that matters: two meals
# inside half an hour mean "time since last meal" is not a complete description of state.
GAP_BUCKETS = (15, 30, 60)

# --- glucose-response check (--glucose-check) ----------------------------------------
RESPONSE_MIN = 120      # look-ahead window after a meal, in minutes
FASTING_HOURS = 4       # a control anchor must be this far after the previous logged meal
FASTING_STRIDE = 37     # subsample CGM rows for control anchors; fixed so the count is stable
MIN_ROWS = 60           # need most of the look-ahead present (raw rows are 1 min apart)


def normalize_meal_type(s: pd.Series) -> pd.Series:
    """10 raw spellings -> 4 classes. 'Snacks' / 'Snack' / 'snack 1' all -> 'snack'.

    Same rule used in ``meals.py``; duplicated here deliberately so the audit
    stays runnable independently.
    """
    return (s.astype(str).str.strip().str.lower()
             .str.replace(r"s$|\s*\d+$", "", regex=True))


def collect_meals(data_dir: Path) -> tuple[pd.DataFrame, dict[tuple, list[str]], pd.DataFrame]:
    """Every logged meal across the clean subjects, plus the schema survey.

    Returns:
        meals    one row per logged meal, with a `subject` column prepended
        schemas  {column tuple: [subject ids]} over EVERY subject on disk, 007 included
                 (007 is excluded from modelling but its schema is part of the finding)
        present  boolean frame, subject x column, True where the file HAS that column
    """
    csvs = find_subject_csvs(data_dir)
    if not csvs:
        raise SystemExit(f"no subject CSVs under {data_dir} -- pass --data-dir")

    schemas: dict[tuple, list[str]] = {}
    presence_rows: dict[str, dict[str, bool]] = {}
    frames = []

    for sid, path in csvs.items():
        df = load_raw(path)                      # strips whitespace, drops the Unnamed zombie
        schemas.setdefault(tuple(df.columns), []).append(sid)

        if sid in MALFORMED:
            continue

        presence_rows[sid] = {col: True for col in df.columns}

        if MEAL_TYPE not in df.columns:          # no subject hits this today; fail loudly if one ever does
            print(f"  ! subject {sid} has no '{MEAL_TYPE}' column -- no meals collected")
            continue

        meals = df[df[MEAL_TYPE].notna()].copy()
        meals.insert(0, "subject", sid)
        frames.append(meals)

    meals = pd.concat(frames, ignore_index=True)
    present = pd.DataFrame(presence_rows).T.fillna(False).sort_index()
    return meals, schemas, present


def report_schemas(schemas: dict[tuple, list[str]]) -> None:
    """Distinct column layouts across subjects -- not all files have the same columns."""
    print(f"\n{'=' * 78}\n1. COLUMN SCHEMAS\n{'=' * 78}")
    print(f"{sum(len(v) for v in schemas.values())} subject files, "
          f"{len(schemas)} distinct schemas\n")

    baseline = max(schemas, key=lambda cols: len(schemas[cols]))   # the modal layout
    for cols, subjects in sorted(schemas.items(), key=lambda kv: -len(kv[1])):
        tag = "  <- the modal one" if cols == baseline else ""
        shown = ", ".join(subjects) if len(subjects) <= 12 else f"{', '.join(subjects[:12])}, ..."
        print(f"n={len(subjects):<3} {shown}{tag}")
        if cols != baseline:
            missing = [c for c in baseline if c not in cols]
            extra = [c for c in cols if c not in baseline]
            moved = [c for c in cols if c in baseline and
                     list(baseline).index(c) != list(cols).index(c)]
            if missing:
                print(f"       missing: {missing}")
            if extra:
                print(f"       extra:   {extra}")
            if not missing and not extra and moved:
                print(f"       same columns, DIFFERENT ORDER: {moved[:4]}")
        print()

    print("Order-only differences are harmless *because* load_raw selects by name --")
    print("a positional loader would silently read the wrong column for those subjects.")


def report_meal_type(meals: pd.DataFrame) -> None:
    """Raw spellings, then what they collapse to."""
    print(f"\n{'=' * 78}\n2. MEAL TYPE\n{'=' * 78}")

    raw = meals[MEAL_TYPE].value_counts()
    print(f"raw ({len(raw)} distinct spellings):")
    for value, n in raw.items():
        print(f"    {str(value)!r:<16} {n:>5}")

    norm = normalize_meal_type(meals[MEAL_TYPE]).value_counts()
    print(f"\nafter .strip().lower() and stripping a trailing 's' or ' <digit>' "
          f"({len(norm)} classes):")
    for value, n in norm.items():
        print(f"    {value:<16} {n:>5}   {100 * n / len(meals):5.1f}%")
    print(f"    {'TOTAL':<16} {len(meals):>5}")

    n_subjects = meals["subject"].nunique()
    per_subject = len(meals) / n_subjects
    print(f"\n{len(meals):,} meals / {n_subjects} subjects = {per_subject:.1f} per subject")


def report_macros(meals: pd.DataFrame, present: pd.DataFrame) -> None:
    """NaN count, spread, and how many values sit above 2x p99 (the corruption tell)."""
    print(f"\n{'=' * 78}\n3. MACRONUTRIENT COLUMNS\n{'=' * 78}")
    print(f"{'column':<10} {'NaN':>6} {'min':>8} {'median':>9} {'p99':>9} "
          f"{'max':>9} {'>2xp99':>7}")

    for col in MACROS:
        if col not in meals.columns:
            print(f"{col:<10}  column absent from every file")
            continue
        values = pd.to_numeric(meals[col], errors="coerce")
        clean = values.dropna()
        p99 = clean.quantile(0.99)
        n_extreme = int((clean > 2 * p99).sum())
        print(f"{col:<10} {values.isna().sum():>6} {clean.min():>8.1f} "
              f"{clean.median():>9.1f} {p99:>9.1f} {clean.max():>9.1f} {n_extreme:>7}")

    # The rows that make the case for winsorizing, ranked by the worst offender.
    if "Fiber" in meals.columns:
        fiber = pd.to_numeric(meals["Fiber"], errors="coerce")
        worst = meals.loc[fiber.nlargest(5).index]
        print("\nworst Fiber rows (fiber is ~2 kcal/g -- check these against Calories):")
        for _, row in worst.iterrows():
            kcal = pd.to_numeric(pd.Series([row.get("Calories")]), errors="coerce").iloc[0]
            fiber_kcal = 2 * float(row["Fiber"])
            flag = "  <- impossible" if pd.notna(kcal) and fiber_kcal > kcal else ""
            print(f"  subject {row['subject']}  {str(row[MEAL_TYPE]).lower():<10} "
                  f"{kcal:>7.0f} kcal  {float(row.get('Carbs', np.nan)):>6.0f} g carbs  "
                  f"{float(row['Fiber']):>7.0f} g fiber  "
                  f"{float(row.get('Fat', np.nan)):>6.0f} g fat{flag}")

    absent = [c for c in MACROS if c in present.columns and not present[c].all()]
    for col in absent:
        missing = sorted(present.index[~present[col]])
        print(f"\n! {col} is absent from the file entirely for: {missing}")


def amount_consumed_table(meals: pd.DataFrame, present: pd.DataFrame) -> pd.DataFrame:
    """Per-subject unit fingerprint for `Amount Consumed`: max, share ==100, share <=10."""
    rows = []
    for sid, group in meals.groupby("subject"):
        has_column = bool(present.loc[sid, AMOUNT]) if AMOUNT in present.columns else False
        values = (pd.to_numeric(group[AMOUNT], errors="coerce").dropna()
                  if AMOUNT in group.columns else pd.Series(dtype=float))

        if not has_column or values.empty:
            rows.append({"subject": sid, "n_meals": len(group), "has_column": has_column,
                         "n_values": len(values), "max": np.nan,
                         "frac_eq_100": np.nan, "frac_le_10": np.nan, "group": "absent"})
            continue

        frac_100 = float((values == 100).mean())
        frac_le10 = float((values <= 10).mean())
        vmax = float(values.max())

        # Three fingerprints, checked in order: never above 10 -> a servings/1-5 scale;
        # above 100 -> out of the documented range; otherwise a percentage.
        if vmax <= 10:
            unit = "small integers"
        elif vmax > 100:
            unit = "out of range"
        else:
            unit = "percent"

        rows.append({"subject": sid, "n_meals": len(group), "has_column": True,
                     "n_values": len(values), "max": vmax,
                     "frac_eq_100": frac_100, "frac_le_10": frac_le10, "group": unit})

    return pd.DataFrame(rows).set_index("subject").sort_index()


def report_amount_consumed(table: pd.DataFrame, detail: bool) -> None:
    """The column that gets dropped -- units are inconsistent across subjects."""
    print(f"\n{'=' * 78}\n4. AMOUNT CONSUMED  (documented 0-100)\n{'=' * 78}")

    counts = table["group"].value_counts()
    print(f"{'group':<16} {'n subj':>7}  subjects")
    for unit in ("percent", "small integers", "out of range", "absent"):
        if unit not in counts:
            continue
        subjects = sorted(table.index[table["group"] == unit])
        shown = ", ".join(subjects) if len(subjects) <= 16 else f"{', '.join(subjects[:16])}, ..."
        print(f"{unit:<16} {counts[unit]:>7}  {shown}")

    print(f"\ntotal subjects accounted for: {int(counts.sum())} "
          f"(must equal {len(table)})")

    ok = table[table["group"] != "absent"]
    if not ok.empty:
        print(f"\nacross subjects that have the column: max ranges "
              f"{ok['max'].min():.0f} to {ok['max'].max():.0f}")
        print(f"share of all values exactly 100: "
              f"{ok['frac_eq_100'].mean():.2f} mean per subject")

    if detail:
        print(f"\n{'subject':<9} {'meals':>6} {'values':>7} {'max':>7} "
              f"{'==100':>7} {'<=10':>7}  group")
        for sid, row in table.iterrows():
            fmt = lambda v: "     --" if pd.isna(v) else f"{v:>7.2f}"
            print(f"{sid:<9} {row['n_meals']:>6} {row['n_values']:>7} "
                  f"{'     --' if pd.isna(row['max']) else format(row['max'], '>7.0f')} "
                  f"{fmt(row['frac_eq_100'])} {fmt(row['frac_le_10'])}  {row['group']}")


def report_alignment(meals: pd.DataFrame) -> None:
    """Do meal timestamps land on the 5-minute CGM lattice? (No.)"""
    print(f"\n{'=' * 78}\n5. TIMESTAMP ALIGNMENT\n{'=' * 78}")

    minute_mod = meals["Timestamp"].dt.minute % 5
    counts = minute_mod.value_counts().sort_index()
    print("Timestamp.dt.minute % 5:   " +
          "   ".join(f"{k}:{v}" for k, v in counts.items()))

    expected = len(meals) / 5
    spread = (counts.max() - counts.min()) / expected
    verdict = "essentially uniform" if spread < 0.35 else "NOT uniform -- investigate"
    print(f"\nuniform would be {expected:.0f} each; observed spread {spread:.0%} -> {verdict}.")
    print("So a meal at 14:23 has to be snapped to a CGM reading (meals.py handles this).")


def report_gaps(meals: pd.DataFrame) -> None:
    """How often meals overlap. Motivates the impulse-sum feature in meal_features.py."""
    print(f"\n{'=' * 78}\n6. CONSECUTIVE-MEAL GAPS\n{'=' * 78}")

    ordered = meals.sort_values(["subject", "Timestamp"])
    gaps = (ordered.groupby("subject")["Timestamp"].diff()
            .dt.total_seconds().div(60).dropna())

    print(f"{len(gaps):,} within-subject gaps, median {gaps.median():.0f} min")
    for cutoff in GAP_BUCKETS:
        n = int((gaps < cutoff).sum())
        print(f"    < {cutoff:>2} min: {n:>4}   ({100 * n / len(gaps):4.1f}%)")

    print(f"\n{int((gaps < 30).sum())} meal pairs are less than half an hour apart, so")
    print('"time since the last meal" is not a complete description of the meal state.')


def _window_stats(times: np.ndarray, glucose: np.ndarray, t) -> tuple[float, float] | None:
    """(peak rise, +60 min delta) over the RESPONSE_MIN window starting at ``t``.

    Returns None when the window is missing or too sparse. Both statistics are relative to
    the reading at ``t``, so they describe the excursion, not the level.
    """
    start_index = int(np.searchsorted(times, np.datetime64(t)))
    end_index = int(np.searchsorted(times, np.datetime64(pd.Timestamp(t) + pd.Timedelta(minutes=RESPONSE_MIN))))
    if start_index == 0 or start_index >= len(times) or end_index - start_index < MIN_ROWS:
        return None
    window = glucose[start_index:end_index]
    delta_60_index = min(start_index + 60, len(glucose) - 1)
    return float(window.max() - glucose[start_index]), float(glucose[delta_60_index] - glucose[start_index])


def glucose_response(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Post-meal glucose excursions, plus a matched no-meal control.

    Answers "was this meal actually eaten?" from the CGM signal instead of from the log.
    The control is not optional: `peak rise = max(window) - start` cannot go below zero, so
    it is biased upward and a raw "+43 mg/dL" is uninterpretable until you know what a
    no-meal window scores.

    Uses the published 1-minute Dexcom column rather than the recovered 5-minute knots
    from B1's deconvolution. That column is linearly interpolated, which would be fatal
    for training but is harmless for a 2-hour max/delta -- interpolation cannot invent
    an excursion.
    """
    meal_rows, fasting_rows = [], []

    for sid, path in find_subject_csvs(data_dir).items():
        if sid in MALFORMED:
            continue
        df = load_raw(path)
        cgm = df[["Timestamp", "Dexcom GL"]].dropna().sort_values("Timestamp")
        if cgm.empty or MEAL_TYPE not in df.columns:
            continue
        times = cgm["Timestamp"].to_numpy()
        glucose = cgm["Dexcom GL"].to_numpy(dtype=float)

        meals = df[df[MEAL_TYPE].notna()]
        meal_times = meals["Timestamp"].to_numpy()
        amounts = (pd.to_numeric(meals[AMOUNT], errors="coerce") if AMOUNT in meals.columns
                   else pd.Series(np.nan, index=meals.index))

        for (_, meal), amount in zip(meals.iterrows(), amounts):
            stats = _window_stats(times, glucose, meal["Timestamp"])
            if stats is None:
                continue
            macros = pd.to_numeric(pd.Series([meal.get(c) for c in MACROS]), errors="coerce")
            meal_rows.append({
                "subject": sid,
                "meal_type": normalize_meal_type(pd.Series([meal[MEAL_TYPE]])).iloc[0],
                "amount": amount,
                "has_amount": AMOUNT in meals.columns,
                # NaN counts as zero here: an empty macro cell is the same defect as a 0 one
                "all_macros_zero": bool((macros.fillna(0) == 0).all()),
                "peak_rise": stats[0], "d60": stats[1],
            })

        # Control anchors: >= FASTING_HOURS since the previous meal AND no meal inside the
        # look-ahead, so the window is genuinely meal-free at both ends.
        for t in times[::FASTING_STRIDE]:
            previous = int(np.searchsorted(meal_times, t, side="right")) - 1
            if previous < 0:
                continue
            if pd.Timestamp(t) - pd.Timestamp(meal_times[previous]) < pd.Timedelta(hours=FASTING_HOURS):
                continue
            next_meal_index = previous + 1
            if (next_meal_index < len(meal_times) and
                    pd.Timestamp(meal_times[next_meal_index]) - pd.Timestamp(t) < pd.Timedelta(minutes=RESPONSE_MIN)):
                continue
            stats = _window_stats(times, glucose, pd.Timestamp(t))
            if stats:
                fasting_rows.append({"subject": sid, "peak_rise": stats[0], "d60": stats[1]})

    return pd.DataFrame(meal_rows), pd.DataFrame(fasting_rows)


def report_glucose_check(meals: pd.DataFrame, fasting: pd.DataFrame) -> None:
    """Did the suspicious rows correspond to food actually being eaten?"""
    print(f"\n{'=' * 78}\n7. GLUCOSE RESPONSE CHECK\n{'=' * 78}")
    print(f"'peak rise' = max(next {RESPONSE_MIN} min) - reading at the meal. It is bounded")
    print("below by 0, so it is biased upward -- read every row against the control.\n")

    def line(name, group):
        if not len(group):
            print(f"{name:<36} n=0")
            return
        print(f"{name:<36} n={len(group):<6} peak rise median {group.peak_rise.median():6.1f}"
              f"  mean {group.peak_rise.mean():6.1f}   +60 delta median {group.d60.median():6.1f}")

    with_amount = meals[meals.has_amount & meals.amount.notna()]

    line(f"CONTROL: no meal for {FASTING_HOURS}h+", fasting)
    print()
    line("meals, amount > 0", with_amount[with_amount.amount > 0])
    line("meals, amount == 0", with_amount[with_amount.amount == 0])
    line("meals, no Amount column (028/029)", meals[~meals.has_amount])
    print()
    zero = with_amount[with_amount.amount == 0]
    line("  amount==0, macros ALL zero", zero[zero.all_macros_zero])
    line("  amount==0, macros present", zero[~zero.all_macros_zero])
    print()
    line("all-macros-zero rows (any amount)", meals[meals.all_macros_zero])
    line("  of those, amount > 0", meals[meals.all_macros_zero & (meals.amount > 0)])

    print(f"\nall-macros-zero: {int(meals.all_macros_zero.sum())} of {len(meals)} meals with a "
          f"usable window ({meals.all_macros_zero.mean():.1%})")

    if len(zero):
        by_type = zero.meal_type.value_counts()
        print(f"amount==0 by meal type: "
              f"{', '.join(f'{k} {v}' for k, v in by_type.items())}"
              f"   (all meals: {', '.join(f'{k} {v}' for k, v in meals.meal_type.value_counts().items())})")

    print("\nIf amount==0 rows matched the fasting control, they weren't eaten and could be")
    print("dropped. They match the ordinary meal rows instead -- they were eaten, and the")
    print("all-zero macros are missing data, not zero-carb meals. Timing is still valid;")
    print("all-zero macros are filled with the train median in meals.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit the raw CGMacros meal columns.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "dataset" / "CGMacros",
                        help="directory holding the CGMacros-XXX/ subject folders")
    parser.add_argument("--detail", action="store_true",
                        help="also print the per-subject Amount Consumed table")
    parser.add_argument("--csv", type=Path, default=None,
                        help="write the per-subject Amount Consumed table here")
    parser.add_argument("--glucose-check", action="store_true",
                        help="post-meal glucose response vs a fasting control "
                             "(second pass over the CSVs, ~20 s)")
    args = parser.parse_args()

    meals, schemas, present = collect_meals(args.data_dir)

    print(f"loaded {len(meals):,} logged meals from {meals['subject'].nunique()} subjects "
          f"({sorted(MALFORMED)} excluded)")

    report_schemas(schemas)
    report_meal_type(meals)
    report_macros(meals, present)

    amount = amount_consumed_table(meals, present)
    report_amount_consumed(amount, args.detail)

    report_alignment(meals)
    report_gaps(meals)

    if args.glucose_check:
        report_glucose_check(*glucose_response(args.data_dir))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        amount.to_csv(args.csv)
        print(f"\nwrote {args.csv}")

    print("\nNo assertions -- this is a read-only inspection.")
