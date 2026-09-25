"""Table 1 -- participant characteristics, stratified by glycemic class.

Joins bio.csv (age/sex/BMI/A1c/labs) to the CGM record actually modelled
(segments_split.npz), so the table describes the analysed cohort, not the raw dataset.

    python src/cohort_table.py                    # print + write results/cohort_table.md
    python src/cohort_table.py --detail           # also print every subject
    python src/cohort_table.py --data-dir data --raw-dir dataset/CGMacros

Outputs:
    results/cohort_table.md        paper-ready Table 1
    results/cohort_per_subject.csv one row per subject (feeds later per-subject figures)

Notes:
  - bio.csv stores subject as an integer; segments use zero-padded strings. Both sides
    are normalized before merging or it silently drops every subject.
  - A1c bins are LEFT-CLOSED (right=False): [0, 5.7), [5.7, 6.5), [6.5, inf). Four
    subjects sit exactly on a threshold, so right=True would misclassify ~9% of the cohort.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import load_segments

A1C_NORMAL, A1C_DIABETIC = 5.7, 6.5          # ADA cutoffs, NGSP %
CLASS_ORDER = ["normal", "pre-diabetic", "diabetic"]
HISTORY, HORIZON = 12, 6                      # H=12, PH=+30 -> a window needs 18 readings


def glucose_stats(segments) -> pd.DataFrame:
    """Per-subject CGM description, computed from the exact segments the model consumes."""
    by_subject: dict[str, list[np.ndarray]] = {}
    counts: dict[str, int] = {}
    windows: dict[str, int] = {}
    for segment in segments:
        by_subject.setdefault(segment.subject, []).append(segment.glucose)
        counts[segment.subject] = counts.get(segment.subject, 0) + 1
        # A window needs history + horizon readings, so a segment of length n yields
        # n - (history + horizon) + 1 of them (0 if the segment is too short).
        windows[segment.subject] = windows.get(segment.subject, 0) + max(
            0, len(segment.glucose) - (HISTORY + HORIZON) + 1)

    rows = []
    for sid, arrays in sorted(by_subject.items()):
        glucose = np.concatenate(arrays)
        rows.append({
            "sid": sid,
            "n_segments": counts[sid],
            "n_readings": len(glucose),
            "n_windows_ph30": windows[sid],
            "cgm_days": len(glucose) * 5 / 1440,          # 5-min readings -> days
            "glucose_mean": float(glucose.mean()),
            "glucose_sd": float(glucose.std(ddof=1)),
            "glucose_min": float(glucose.min()),
            "glucose_max": float(glucose.max()),
            "pct_above_180": 100 * float((glucose > 180).mean()),
            "pct_below_70": 100 * float((glucose < 70).mean()),
        })
    return pd.DataFrame(rows)


def load_bio(raw_dir: Path) -> pd.DataFrame:
    """bio.csv, normalized. Column names carry stray spaces ('Body weight ') and the file
    has two identical 'Time (t)' headers, so pandas de-duplicates them -- harmless here,
    but it is why this cannot be read with a stricter CSV reader."""
    bio = pd.read_csv(raw_dir / "bio.csv")
    bio.columns = [c.strip() for c in bio.columns]
    # int -> zero-padded string, the id the segments use
    bio["sid"] = bio["subject"].astype(int).map(lambda n: f"{n:03d}")
    return bio


def summarize(values, digits=1, unit="") -> str:
    """mean +/- SD over subjects, or '--' when the field is entirely missing."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if v.empty:
        return "--"
    if len(v) == 1:
        return f"{v.iloc[0]:.{digits}f}{unit}"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}{unit}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Table 1: participant characteristics")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/CGMacros"),
                        help="directory holding bio.csv")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--detail", action="store_true", help="print every subject")
    args = parser.parse_args()

    segments = load_segments(args.data_dir / "segments_split.npz")
    cgm = glucose_stats(segments)
    bio = load_bio(args.raw_dir)

    merged = cgm.merge(bio, on="sid", how="left", indicator=True)
    unmatched = merged.loc[merged["_merge"] != "both", "sid"].tolist()
    if unmatched:
        print(f"WARNING: no bio.csv row for {unmatched} -- check the id normalization")
    enrolled_only = sorted(set(bio["sid"]) - set(cgm["sid"]))

    merged["a1c"] = pd.to_numeric(merged["A1c PDL (Lab)"], errors="coerce")
    merged["glycemic_class"] = pd.cut(
        merged["a1c"], bins=[0, A1C_NORMAL, A1C_DIABETIC, 100],
        labels=CLASS_ORDER, right=False)          # LEFT-closed: see the module docstring
    merged["fasting_glucose"] = pd.to_numeric(
        merged.get("Fasting GLU - PDL (Lab)"), errors="coerce")
    merged["age"] = pd.to_numeric(merged.get("Age"), errors="coerce")
    merged["bmi"] = pd.to_numeric(merged.get("BMI"), errors="coerce")

    gender = merged.get("Gender", pd.Series(dtype=object)).astype(str).str.strip().str.lower()
    merged["is_female"] = gender.str.startswith("f")
    print(f"gender values found: {dict(gender.value_counts())}")

    # ------------------------------------------------------------------ Table 1
    groups = [(name, merged[merged["glycemic_class"] == name]) for name in CLASS_ORDER]
    groups.append(("All", merged))

    def row(label, fn):
        return [label] + [fn(sub) for _, sub in groups]

    table_rows = [
        row("Subjects, n", lambda d: f"{len(d)}"),
        row("Age, years", lambda d: summarize(d["age"])),
        row("Female, n (%)", lambda d: f"{int(d['is_female'].sum())} "
                                       f"({100 * d['is_female'].mean():.0f}%)" if len(d) else "--"),
        row("BMI, kg/m²", lambda d: summarize(d["bmi"])),
        row("HbA1c, %", lambda d: summarize(d["a1c"], digits=2)),
        row("Fasting glucose, mg/dL", lambda d: summarize(d["fasting_glucose"], digits=0)),
        row("CGM recording, days", lambda d: summarize(d["cgm_days"])),
        row("Segments, n", lambda d: f"{int(d['n_segments'].sum())}"),
        row("Dexcom readings (5-min), n", lambda d: f"{int(d['n_readings'].sum()):,}"),
        row("Windows (H=12, PH=+30), n", lambda d: f"{int(d['n_windows_ph30'].sum()):,}"),
        row("Mean glucose, mg/dL", lambda d: summarize(d["glucose_mean"], digits=0)),
        row("Within-subject glucose SD, mg/dL", lambda d: summarize(d["glucose_sd"], digits=1)),
        row("Time > 180 mg/dL, %", lambda d: summarize(d["pct_above_180"])),
        row("Time < 70 mg/dL, %", lambda d: summarize(d["pct_below_70"], digits=2)),
    ]

    header = "| Characteristic | " + " | ".join(
        f"{name} (n={len(sub)})" if name != "All" else f"All (n={len(sub)})"
        for name, sub in groups) + " |"
    lines = [header, "|" + "---|" * (len(groups) + 1)]
    lines += ["| " + " | ".join(cells) + " |" for cells in table_rows]
    table = "\n".join(lines)

    print("\n" + "=" * 78)
    print("  TABLE 1 -- participant characteristics (analysed cohort)")
    print("=" * 78)
    print(table)

    note = (f"\n_Glycemic class from HbA1c using ADA cutoffs (normal < {A1C_NORMAL}%, "
            f"pre-diabetic {A1C_NORMAL}–{A1C_DIABETIC}%, diabetic ≥ {A1C_DIABETIC}%), "
            f"left-closed bins. Values are mean ± SD across subjects unless noted; counts "
            f"are totals. CGM figures describe the real 5-minute Dexcom readings actually "
            f"used for modelling, recovered from the published 1-minute series by "
            f"second-difference phase detection. "
            + (f"Subject(s) {', '.join(enrolled_only)} are in bio.csv but excluded from "
               f"analysis (irregular timestamps). " if enrolled_only else "")
            + f"{len(bio)} participants enrolled, {len(merged)} analysed._\n")
    print(note)

    out_md = args.results_dir / "cohort_table.md"
    out_md.write_text(f"# Table 1 — participant characteristics\n\n{table}\n{note}",
                      encoding="utf-8")

    keep = ["sid", "glycemic_class", "a1c", "age", "bmi", "is_female", "fasting_glucose",
            "n_segments", "n_readings", "n_windows_ph30", "cgm_days", "glucose_mean",
            "glucose_sd", "glucose_min", "glucose_max", "pct_above_180", "pct_below_70"]
    out_csv = args.results_dir / "cohort_per_subject.csv"
    merged[keep].to_csv(out_csv, index=False)

    if args.detail:
        print(merged[keep].to_string(index=False))

    # ------------------------------------------------------------------ checks
    print("consistency checks against known cohort totals:")
    total_readings = int(merged["n_readings"].sum())
    total_windows = int(merged["n_windows_ph30"].sum())
    total_segments = int(merged["n_segments"].sum())
    print(f"  subjects {len(merged)} (expect 44)   segments {total_segments} (expect 81)")
    print(f"  readings {total_readings:,} (expect 124,808)   "
          f"windows {total_windows:,} (expect 123,435)")
    # Each segment loses (history + horizon - 1) = 17 readings to the window edges, EXCEPT
    # segments shorter than 18 readings, which yield 0 windows rather than a negative count.
    # One CGMacros segment is 13 readings (65 min) -- kept because 13 is enough for a PH=+5
    # window, but it contributes nothing at PH=+30, which is the whole 4-window difference.
    lengths = [len(s.glucose) for s in segments]
    edge_loss = (HISTORY + HORIZON - 1) * total_segments
    too_short = [n for n in lengths if n < HISTORY + HORIZON]
    clamped = sum(HISTORY + HORIZON - 1 - n for n in too_short)
    naive = total_readings - edge_loss
    print(f"  windows = readings - 17*segments ({naive:,}) + {clamped} clamped from "
          f"{len(too_short)} segment(s) shorter than {HISTORY + HORIZON} readings "
          f"{too_short} -> {naive + clamped:,} "
          f"{'consistent' if naive + clamped == total_windows else 'MISMATCH'}")
    counts = merged["glycemic_class"].value_counts()
    print(f"  classes {[f'{c} {int(counts.get(c, 0))}' for c in CLASS_ORDER]} "
          f"(expect 15 / 15 / 14)")
    print(f"  A1c range {merged['a1c'].min():.1f}-{merged['a1c'].max():.1f} % "
          f"(expect 4.6-8.5, NGSP % -- the data dictionary's 'mmol/mol' is wrong)")
    print(f"\nwrote {out_md}\nwrote {out_csv}")


if __name__ == "__main__":
    main()
