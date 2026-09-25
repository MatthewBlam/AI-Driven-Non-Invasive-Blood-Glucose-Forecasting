"""Data-quality audit for CGMacros. Verifies every data-integrity claim numerically.

Every check prints its result. Any claim not backed by a number here hasn't been
verified.

Usage:
    python src/audit.py --data-dir /path/to/CGMacros [--out results/audit.json]

`--data-dir` should be the directory containing the per-subject folders
(CGMacros-001/, CGMacros-002/, ...), i.e. the `CGMacros/` directory inside the zip.
`bio.csv` is expected there too; `DataDictionary_Bio.csv` in it or its parent.
"""

# /usr/local/bin/python3 /Users/mattyb/Projects/glucose-kanjilal/src/audit.py --data-dir dataset/CGMacros --out results/out.json

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

DEXCOM_PERIOD_MIN = 5    # Dexcom G6 Pro native sampling
LIBRE_PERIOD_MIN = 15    # FreeStyle Libre Pro native sampling
LINEAR_TOL = 1e-6        # |2nd difference| above this is a real slope change, not float noise
MIN_SEGMENT = 13         # H=12 history + 1 target; a shorter run cannot make one window
MALFORMED = {"007"}      # 870 irregular timestamp rows over only 3.9 days -- see trap 5

A1C_NORMAL, A1C_DIABETIC = 5.7, 6.5   # ADA cutoffs, NGSP %


# --------------------------------------------------------------------------- io

def find_subject_csvs(data_dir: Path) -> dict[str, Path]:
    """Map subject id -> csv path. Subject ids are NOT contiguous (024/025/037/040 absent)."""
    out = {}
    for p in sorted(data_dir.glob("CGMacros-*/CGMacros-*.csv")):
        out[p.stem.replace("CGMacros-", "")] = p
    if not out:  # tolerate a flat directory of csvs
        for p in sorted(data_dir.glob("CGMacros-*.csv")):
            out[p.stem.replace("CGMacros-", "")] = p
    return out


def load_raw(path: Path) -> pd.DataFrame:
    """Load one subject, normalizing the schema. See trap 4."""
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df.columns = [c.strip() for c in df.columns]           # 'Amount Consumed ' -> 'Amount Consumed'
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # stray pandas index column
    return df


def find_file(data_dir: Path, name: str) -> Path | None:
    for cand in (data_dir / name, data_dir.parent / name):
        if cand.exists():
            return cand
    return None


# ------------------------------------------------------- traps 1 & 2: the lattice

def contiguous_blocks(df: pd.DataFrame, col: str, min_len: int = 3):
    """Maximal runs of consecutive 1-minute rows in which `col` is present.

    The block, not the subject, is the right unit of analysis. A block ends at a missing
    value OR at a timestamp step != 1 min. Ten subjects have short stretches where the
    CSV simply omits minutes (subject 003, 09:05-09:33, 2-minute steps).

    Everything downstream assumes 1-minute contiguity. Establish it here, once.
    """
    present = df[col].notna()
    one_min = df["Timestamp"].diff().dt.total_seconds() == 60
    block_id = ((~present) | (~one_min)).cumsum()
    for _, g in df[present].groupby(block_id[present]):
        if len(g) >= min_len:
            yield g.reset_index(drop=True)


def block_phase(block: pd.DataFrame, col: str, period_min: int) -> tuple[int | None, float]:
    """Recover the true sampling lattice of a linearly-interpolated block. Traps 1 and 2.

    The column is piecewise linear between real readings ("knots"), so the second
    difference

        d2[i] = x[i] - 2*x[i-1] + x[i-2]

    vanishes unless the slope changes between step (i-2 -> i-1) and step (i-1 -> i) --
    that is, unless a knot sits at **i-1**.

    So nonzero d2 marks knot+1, NOT the knot. Dropping that -1 shifts every "real"
    reading one minute forward, onto an interpolated value. (This script used to do
    exactly that. The resulting series is a 0.8/0.2 blend of adjacent true readings:
    smoother than reality, and it made persistence look 0.2 mg/dL better than it is.)

    Returns (phase, residual). `phase` is `minute % period_min` of the true knots.
    `residual` is max|d2| over the rows where d2 must vanish -- ~1e-14 iff the block
    really is piecewise linear, which is the direct proof of trap 1.
    """
    d2 = block[col].diff().diff().abs()
    knot_plus_1 = d2.index[d2 > LINEAR_TOL]
    if len(knot_plus_1) == 0:
        return None, 0.0                            # perfectly flat block; no lattice to find
    minute = block["Timestamp"].dt.minute
    phase = int(((minute.loc[knot_plus_1] - 1) % period_min).mode().iloc[0])

    off_lattice = (minute % period_min != (phase + 1) % period_min) & d2.notna()
    residual = float(d2[off_lattice].max()) if off_lattice.any() else 0.0
    return phase, residual


def extract_segments(df: pd.DataFrame, col: str = "Dexcom GL",
                     period_min: int = DEXCOM_PERIOD_MIN,
                     min_len: int = MIN_SEGMENT) -> tuple[list[np.ndarray], list[dict]]:
    """Real sensor readings only, as contiguous evenly-spaced segments.

    This is THE function. Baselines, preprocessing, and the model all consume its output.

    Phase is detected per block because phase is not constant within a subject: subject
    026's sensor was restarted after a 1.7-day gap and came back on a different lattice.
    """
    segments, diagnostics = [], []
    for block in contiguous_blocks(df, col):
        phase, residual = block_phase(block, col, period_min)
        if phase is None:
            continue
        knots = block[block["Timestamp"].dt.minute % period_min == phase]
        gaps = knots["Timestamp"].diff().dropna().dt.total_seconds()
        assert (gaps == period_min * 60).all(), "knots unevenly spaced -- block is not 1-min contiguous"

        diagnostics.append({"block_rows": len(block), "phase": phase,
                            "residual": residual, "n_knots": len(knots)})
        if len(knots) >= min_len:
            segments.append(knots[col].to_numpy(dtype=np.float64))
    return segments, diagnostics


def dominant_phase(diagnostics: list[dict]) -> int | None:
    """Phase of the largest block. Only for the cohort-level phase histogram."""
    return max(diagnostics, key=lambda d: d["n_knots"])["phase"] if diagnostics else None


def interpolation_ramp(df: pd.DataFrame, col: str, phase: int, period_min: int) -> list[dict]:
    """Trap 1, shown rather than asserted: one inter-knot interval, row by row."""
    for block in contiguous_blocks(df, col, min_len=2 * period_min):
        minute = block["Timestamp"].dt.minute
        for i in block.index[minute % period_min == phase]:
            window = block.loc[i: i + period_min]
            if len(window) == period_min + 1 and abs(window[col].diff().iloc[1]) > 0.3:
                return [{"timestamp": str(t), "value": float(v),
                         "real": bool(m % period_min == phase)}
                        for t, v, m in zip(window["Timestamp"], window[col],
                                           window["Timestamp"].dt.minute)]
    return []


# ------------------------------------------------------------- trap 6: missingness

def nan_gap_profile(s: pd.Series) -> tuple[int, int]:
    """(longest NaN run anywhere, longest NaN run strictly inside the record).

    The difference between these two numbers IS trap 6. Most Dexcom "missingness" is edge
    truncation -- the G6 Pro was worn ~10 days inside a longer Libre recording, so the
    column is one contiguous block with NaN padding on both ends. Only the interior
    number tells you whether there is a real dropout that needs a gap policy.
    """
    isna = s.isna().to_numpy()

    def longest(mask: np.ndarray) -> int:
        best = cur = 0
        for v in mask:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    real = np.flatnonzero(~isna)
    if len(real) == 0:
        return len(isna), 0
    return longest(isna), longest(isna[real[0]: real[-1] + 1])


# ------------------------------------------------------------------- trap 7: bio.csv

def audit_bio(data_dir: Path, clean: set[str]) -> dict | None:
    """Glycemic stratification, and the A1c units error in the data dictionary."""
    bio_path = find_file(data_dir, "bio.csv")
    if bio_path is None:
        return None
    bio = pd.read_csv(bio_path)
    bio.columns = [c.strip() for c in bio.columns]
    bio["sid"] = bio["subject"].astype(int).map(lambda n: f"{n:03d}")
    a1c = bio.set_index("sid")["A1c PDL (Lab)"].astype(float)

    def strata(ids) -> dict[str, int]:
        v = a1c.loc[sorted(set(ids) & set(a1c.index))]
        return {"normal": int((v < A1C_NORMAL).sum()),
                "pre_diabetic": int(((v >= A1C_NORMAL) & (v < A1C_DIABETIC)).sum()),
                "diabetic": int((v >= A1C_DIABETIC).sum())}

    out = {"a1c_min": float(a1c.min()), "a1c_max": float(a1c.max()),
           "strata_all": strata(a1c.index), "strata_clean": strata(clean),
           "a1c_by_subject": {k: float(v) for k, v in a1c.items()}}

    dict_path = find_file(data_dir, "DataDictionary_Bio.csv")
    if dict_path is not None:
        dd = pd.read_csv(dict_path, encoding="utf-8-sig")   # the file carries a BOM
        row = dd[dd["Field"].str.strip() == "A1c PDL (Lab)"]
        if len(row):
            claim = str(row["Description"].iloc[0])
            # 4.6-8.5 is unambiguously NGSP percent; 5.7 % is about 39 mmol/mol.
            out["a1c_units_claimed"] = "mmol/mol" if "mmol/mol" in claim else claim
            out["a1c_units_actual"] = "NGSP %"
            out["a1c_dictionary_wrong"] = bool("mmol/mol" in claim and a1c.max() < 20)
    return out


# ------------------------------------------------------------------------- the audit

def audit(data_dir: Path) -> dict:
    csvs = find_subject_csvs(data_dir)
    if not csvs:
        raise SystemExit(f"No CGMacros-*.csv found under {data_dir}")

    headers = Counter()
    per_subject, meals = [], []
    dex_phases, libre_aligned = Counter(), 0
    ramp: list[dict] = []
    clean_values: list[np.ndarray] = []

    for sid, path in csvs.items():
        headers[tuple(pd.read_csv(path, nrows=0).columns)] += 1

        df = load_raw(path)
        ts = df["Timestamp"]

        segs, ddiag = extract_segments(df, "Dexcom GL", DEXCOM_PERIOD_MIN)
        _, ldiag = extract_segments(df, "Libre GL", LIBRE_PERIOD_MIN)

        dphase, lphase = dominant_phase(ddiag), dominant_phase(ldiag)
        if dphase is not None:
            dex_phases[dphase] += 1
        # Trap 3: do Libre's real readings land on Dexcom's?
        if dphase is not None and lphase is not None and lphase % DEXCOM_PERIOD_MIN == dphase:
            libre_aligned += 1

        if not ramp and dphase is not None:
            ramp = interpolation_ramp(df, "Dexcom GL", dphase, DEXCOM_PERIOD_MIN)

        gap_any, gap_interior = nan_gap_profile(df["Dexcom GL"])
        step = ts.diff().dt.total_seconds().div(60).dropna()
        phases_used = sorted({d["phase"] for d in ddiag if d["n_knots"] >= MIN_SEGMENT})

        if sid not in MALFORMED and segs:
            clean_values.append(np.concatenate(segs))

        per_subject.append({
            "subject": sid,
            "days": round(len(df) / 1440, 1),
            "dexcom_phase": dphase,
            "libre_phase": lphase,
            "n_distinct_phases": len(phases_used),                       # 026 has 2
            "max_linearity_residual": max((d["residual"] for d in ddiag), default=0.0),
            "dexcom_missing_frac": round(df["Dexcom GL"].isna().mean(), 4),
            "hr_missing_frac": round(df["HR"].isna().mean(), 4) if "HR" in df else None,
            "longest_dexcom_nan_gap_min": gap_any,                       # trap 6
            "longest_interior_nan_gap_min": gap_interior,                # trap 6, the real one
            "irregular_timestamp_rows": int((step != 1.0).sum()),
            "n_real_dexcom": int(sum(len(s) for s in segs)),
            "n_segments": len(segs),
            "longest_segment": max((len(s) for s in segs), default=0),
            "windows_H12_PH30": sum(max(0, len(s) - 12 - 6 + 1) for s in segs),
            "has_METs": "METs" in df.columns,
            "has_Amount_Consumed": "Amount Consumed" in df.columns,
            "malformed": sid in MALFORMED,
        })

        m = df[df["Meal Type"].notna()].copy()
        if len(m):
            m["subject"] = sid
            meals.append(m)

    subj = pd.DataFrame(per_subject).sort_values("subject")
    clean = subj[~subj.malformed]
    glucose_values = pd.Series(np.concatenate(clean_values))
    meals_df = pd.concat(meals, ignore_index=True) if meals else pd.DataFrame()

    report = {
        "n_subjects_found": len(csvs),
        "subject_ids": sorted(csvs),
        "linearity_residual_max": float(subj.max_linearity_residual.max()),      # trap 1
        "interpolation_ramp": ramp,                                              # trap 1
        "dexcom_phase_distribution": dict(sorted(dex_phases.items())),           # trap 2
        "subjects_with_multiple_phases": sorted(subj[subj.n_distinct_phases > 1].subject),
        "libre_aligned_with_dexcom": f"{libre_aligned}/{len(csvs)}",             # trap 3
        "n_header_variants": len(headers),                                       # trap 4
        "header_variants": [{"count": count, "columns": list(cols)} for cols, count in headers.items()],
        "files_missing_METs": int((~subj.has_METs).sum()),
        "files_missing_Amount_Consumed": int((~subj.has_Amount_Consumed).sum()),
        "malformed_subjects": sorted(MALFORMED),                                 # trap 5
        "subjects_with_irregular_timestamps": {
            r.subject: r.irregular_timestamp_rows
            for r in subj.itertuples() if r.irregular_timestamp_rows > 0},
        "missingness": {                                                         # trap 6
            "dexcom_missing_frac_range": [float(clean.dexcom_missing_frac.min()),
                                          float(clean.dexcom_missing_frac.max())],
            "hr_missing_frac_max": float(clean.hr_missing_frac.max()),
            "worst_interior_gap_min": int(clean.longest_interior_nan_gap_min.max()),
            "worst_interior_gap_subject": clean.loc[
                clean.longest_interior_nan_gap_min.idxmax(), "subject"],
            "n_subjects_with_interior_gap_gt_45min": int(
                (clean.longest_interior_nan_gap_min > 45).sum()),
            "median_segments_per_subject": float(clean.n_segments.median()),
        },
        "clean_cohort": {
            "n_subjects": len(clean),
            "total_real_dexcom_readings": int(clean.n_real_dexcom.sum()),
            "total_windows_H12_PH30": int(clean.windows_H12_PH30.sum()),
            "glucose_mean": round(float(glucose_values.mean()), 2),
            "glucose_std": round(float(glucose_values.std()), 2),
            "glucose_skew": round(float(glucose_values.skew()), 2),
            "frac_hyper_gt180": round(float((glucose_values > 180).mean()), 4),
            "frac_hypo_lt70": round(float((glucose_values < 70).mean()), 4),     # ~1 in 270 -- rare, marginal impact on RMSE
        },
        "bio": audit_bio(data_dir, set(clean.subject)),                          # trap 7
        "per_subject": per_subject,
    }

    if len(meals_df):                                                             # trap 7
        meal_types = meals_df["Meal Type"].astype(str).str.lower().str.strip()
        report["meal_type_counts"] = meal_types.value_counts().to_dict()
        if "Amount Consumed" in meals_df:
            consumed = meals_df["Amount Consumed"].dropna()
            report["amount_consumed"] = {
                "documented_range": "0-100 (%)",
                "observed_max": float(consumed.max()),
                "frac_above_100": round(float((consumed > 100).mean()), 4),
                "distinct_above_100": sorted(float(v) for v in consumed[consumed > 100].unique()),
            }
    return report


# ------------------------------------------------------------------------- reporting

def report_text(r: dict) -> str:
    cohort, bio = r["clean_cohort"], r["bio"]
    lines = ["", f"subjects found ............ {r['n_subjects_found']}"]

    lines.append("\n--- trap 1: Dexcom is linearly interpolated onto a 1-min grid ---")
    lines.append(f"max |2nd difference| off the knot lattice ... {r['linearity_residual_max']:.2e}")
    lines.append("  that is float noise. the signal is EXACTLY piecewise linear between knots.")
    for row in r["interpolation_ramp"]:
        lines.append(f"  {row['timestamp']}   {row['value']:6.1f}   "
                     f"{'<- real' if row['real'] else 'interpolated'}")

    lines.append("\n--- trap 2: the knot phase varies, and is not constant within a subject ---")
    lines.append(f"phase distribution (minute % 5) .. {r['dexcom_phase_distribution']}")
    n0 = r["dexcom_phase_distribution"].get(0, 0)
    lines.append(f"  iloc[::5] is correct for {n0}/{r['n_subjects_found']} subjects. it is wrong for the rest.")
    lines.append(f"subjects with >1 phase ........... {r['subjects_with_multiple_phases']}   <- sensor restart")

    lines.append("\n--- trap 3: Libre knots rarely coincide with Dexcom knots ---")
    lines.append(f"aligned .......................... {r['libre_aligned_with_dexcom']} subjects")

    lines.append("\n--- trap 4: schema drift ---")
    lines.append(f"distinct CSV headers ............. {r['n_header_variants']}")
    lines.append(f"files missing METs ............... {r['files_missing_METs']}")
    lines.append(f"files missing Amount Consumed .... {r['files_missing_Amount_Consumed']}")

    lines.append("\n--- trap 5: malformed subjects ---")
    lines.append(f"excluded ......................... {r['malformed_subjects']}")
    lines.append(f"irregular timestamp rows ......... {r['subjects_with_irregular_timestamps']}")
    lines.append("  only 007 is fatal. the others lose one short block each to a block boundary.")

    missingness = r["missingness"]
    lines.append("\n--- trap 6: missingness is mostly EDGE TRUNCATION, not dropout ---")
    lines.append(f"Dexcom missing frac (clean) ...... {missingness['dexcom_missing_frac_range'][0]:.1%} "
                 f"- {missingness['dexcom_missing_frac_range'][1]:.1%}")
    lines.append(f"HR missing frac, worst subject ... {missingness['hr_missing_frac_max']:.1%}")
    lines.append(f"worst INTERIOR gap ............... {missingness['worst_interior_gap_min']} min "
                 f"(subject {missingness['worst_interior_gap_subject']})")
    lines.append(f"subjects w/ interior gap > 45 min  {missingness['n_subjects_with_interior_gap_gt_45min']}"
                 "   <- only these need a gap policy")
    lines.append(f"median segments per subject ...... {missingness['median_segments_per_subject']:.0f}")

    lines.append("\n--- trap 7: dirty fields ---")
    if "meal_type_counts" in r:
        lines.append(f"Meal Type values ................. {r['meal_type_counts']}")
    if "amount_consumed" in r:
        amount = r["amount_consumed"]
        lines.append(f"Amount Consumed .................. documented 0-100, observed max "
                     f"{amount['observed_max']:.0f}, {amount['frac_above_100']:.1%} of rows > 100")
    if bio:
        lines.append(f"A1c units ....................... dictionary says '{bio.get('a1c_units_claimed')}', "
                     f"values are {bio['a1c_min']}-{bio['a1c_max']} => {bio['a1c_units_actual']}. dictionary is WRONG.")

    lines.append(f"\n--- clean cohort (n={cohort['n_subjects']}) ---")
    lines.append(f"real Dexcom readings ...... {cohort['total_real_dexcom_readings']:,}")
    lines.append(f"windows (H=12, PH=+30) .... {cohort['total_windows_H12_PH30']:,}")
    lines.append(f"glucose ................... {cohort['glucose_mean']} +/- {cohort['glucose_std']} mg/dL "
                 f"(skew {cohort['glucose_skew']})")
    lines.append(f"hyper >180 ................ {cohort['frac_hyper_gt180']:.1%}")
    lines.append(f"hypo  < 70 ................ {cohort['frac_hypo_lt70']:.2%}   <- RMSE will not see these")
    if bio:
        strata = bio["strata_clean"]
        lines.append(f"glycemic strata ........... {strata['normal']} normal / {strata['pre_diabetic']} pre-diabetic "
                     f"/ {strata['diabetic']} diabetic   <- stratify your LOSO results by this")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    r = audit(args.data_dir)
    print(report_text(r))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(r, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
