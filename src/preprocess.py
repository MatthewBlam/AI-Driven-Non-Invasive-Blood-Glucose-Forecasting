"""Raw CGMacros CSVs -> clean, windowable segments.

Pipeline order matters:

    load -> drop 007 -> detect phase -> real knots -> segments
      -> SPLIT -> fit scaler on TRAIN ONLY -> window each split
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd


HISTORY = 12  # 60 min at 5-min spacing
MIN_SEGMENT = 13  # floor; each horizon prunes further in make_windows (need H+k)

DEXCOM_PERIOD_MIN = 5  # Dexcom G6 native sampling
MALFORMED = {"007"}  # bad data from subject 007


@dataclass(frozen=True)
class Segment:
    """A run of consecutive, real, 5-min-apart Dexcom readings from one subject"""

    subject: str
    glucose: np.ndarray  # (L,) float, mg/dL, no NaN
    start_time: pd.Timestamp

    def __len__(self) -> int:
        return len(self.glucose)



# Phase detection

def detect_dexcom_phase(block: pd.DataFrame) -> int:
    """Return `minute % 5` of the REAL Dexcom readings in one contiguous block. Returns 0 if undetectable (a perfectly flat block)."""

    # Block is already a contiguous run of non-NaN Dexcom rows (from split_into_blocks)
    # Identify phase using second difference
    second_difference = block["Dexcom GL"].diff().diff()
    change_point_indices = second_difference[second_difference.abs() > 1e-6].index
    timestamps = block.loc[change_point_indices, "Timestamp"]
    phase = ((timestamps.dt.minute - 1) % 5).mode()[0]

    # Verification: rows NOT at (phase+1)%5 should have near-zero second difference
    non_knot_mask = block["Timestamp"].dt.minute % 5 != (phase + 1) % 5
    worst_residual = second_difference[non_knot_mask].abs().max()
    assert worst_residual < 1e-9, f"phase {phase} failed verification, worst residual: {worst_residual}"

    return phase




# Segmentation

def extract_segments(block: pd.DataFrame, sid: str, phase: int, gap_policy: str = "split") -> list[Segment]:
    """Real 5-min values -> list of contiguous, NaN-free Segments.

    A valid segment requires, for every consecutive pair of rows:
      - both Dexcom values present (not NaN), and
      - timestamps exactly 5 minutes apart.

    gap_policy:
      "split"       -- any gap ends the segment
      "interpolate" -- fill gaps <= 30 min (6 values) by linear interpolation, split on
                       longer ones. This is GlucoBench's rule (arXiv 2410.05780).
                       NEVER interpolate across a longer gap. NEVER let a window span
                       a segment boundary.

    TODO:
      - if gap_policy == "interpolate", handle short gaps BEFORE splitting
      - record how many windows each policy yields, that table can go in the paper.

      If ignoring makes it too small do interpolation, eg. forward fill or average tricks
      """

    # Filter to real 5-min values from detect_dexcom_phase)
    real_values = block[block["Timestamp"].dt.minute % 5 == phase]

    if gap_policy == "interpolate":
        # Fill gaps <= 30 min (6 values) by linear interpolation, split on longer ones
        # TODO
        pass
    elif gap_policy != "split":
        raise ValueError(f"unknown gap_policy: {gap_policy}")

    # Check which consecutive knots are exactly 5 min apart
    time_diff = real_values["Timestamp"].diff()
    is_consecutive = time_diff == pd.Timedelta(minutes=5)

    # Group contiguous runs, every gap increments the group number
    group = (~is_consecutive).cumsum()

    # Build a Segment from each group, dropping any too short to window
    segments = []
    for _, segment_group in real_values.groupby(group):
        if len(segment_group) < MIN_SEGMENT:
            continue
        glucose = segment_group["Dexcom GL"].to_numpy(dtype=float)
        segments.append(Segment(subject=sid, glucose=glucose, start_time=segment_group["Timestamp"].iloc[0]))

    return segments




def find_subject_csvs(data_dir: Path) -> dict[str, Path]:
    """Map subject id -> CSV path. Subject ids are NOT contiguous (024/025/037/040 absent)."""

    out = {}

    # Iterate through all paths found in dataset and map subject number to the CSV path
    for path in sorted(data_dir.glob("CGMacros-*/CGMacros-*.csv")):
        out[path.stem.replace("CGMacros-", "")] = path

    return out


def load_raw(path: Path) -> pd.DataFrame:
    """Load one subject CSV, normalizing the column schema."""

    # Normalize column schema
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df.columns = [col.strip() for col in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # drop artifact "Unnamed" columns

    return df

def split_into_blocks(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a subject's DataFrame into contiguous blocks of non-NaN Dexcom rows.

    Each block is a run of consecutive 1-minute-apart rows with Dexcom GL present.
    Subjects 018 and 026 have mid-recording sensor restarts that change the phase,
    so phase detection must happen per block, not per subject."""

    valid = df[df["Dexcom GL"].notna()]

    # Mark where a timestamp gap occurs (not exactly 1 min after the previous row)
    one_min = pd.Timedelta(minutes=1)
    gaps = valid["Timestamp"].diff() != one_min

    # Assign a group number to each contiguous run (increments at every gap)
    group = gaps.cumsum()

    # Collect each group as a separate block
    blocks = []
    for group_id in group.unique():
        blocks.append(valid[group == group_id])
    return blocks


def load_all_segments(data_dir: Path, gap_policy: str = "split") -> list[Segment]:
    """All 44 clean subjects. Subject 007 is excluded due to critical timestamp irregularity"""

    out: list[Segment] = []

    # Iterate through each subject, split data into blocks, and extract segments
    for sid, path in find_subject_csvs(data_dir).items():
        if sid in MALFORMED:
            continue

        df = load_raw(path)
        for block in split_into_blocks(df):
            if len(block) < MIN_SEGMENT:
                continue
            phase = detect_dexcom_phase(block)
            out.extend(extract_segments(block, sid, phase, gap_policy))

    return out



# Windowing

def make_windows(segment: Segment, history: int, horizon: int, multi_horizon: bool = False):
    """Slide a window across one segment's glucose array and split into inputs (X) and targets (y).

    Returns (X, y) where X is (N, history) and y is (N,) or (N, horizon) if multi_horizon.
    N = len(segment) - history - horizon + 1. Returns raw mg/dL — scale X later, not here."""

    if len(segment) < history + horizon:
        empty_x = np.empty((0, history))
        empty_y = np.empty((0, horizon)) if multi_horizon else np.empty((0,))
        return empty_x, empty_y

    # Slide a window of length (history + horizon) across the glucose array
    windows = sliding_window_view(segment.glucose, history + horizon)  # Creates 2D view, each row is a window

    # Left columns = history input, right side = prediction target(s)
    X = windows[:, :history]  # Get all history values from windows
    y = windows[:, history:] if multi_horizon else windows[:, -1]  # Get all horizon values (only last for single horizon)

    assert X.shape[0] == len(segment) - history - horizon + 1
    return X, y



# Scaling (leakage-critical)

def fit_scaler(train_segments: list[Segment]) -> tuple[float, float]:
    """Global mean/std over TRAINING SUBJECTS ONLY. Returns (mu, sigma).

    Per-subject normalization would leak: it requires the test subject's full
    recording, which isn't available in deployment."""

    # Stack all training glucose into one array
    all_glucose = np.concatenate([segment.glucose for segment in train_segments])
    mu = float(all_glucose.mean())
    sigma = float(all_glucose.std())

    assert sigma > 0, "zero std — all training glucose values are identical"
    return mu, sigma


def apply_scaler(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Z-score normalize: centers around 0, scales to unit variance.
    Use mu/sigma from fit_scaler (training subjects only)."""

    return (x - mu) / sigma


def invert_scaler(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Un-scale predictions back to mg/dL before computing RMSE."""

    return x * sigma + mu



# Cache

def save_segments(segments: list[Segment], path: Path, gap_policy: str = "split") -> None:
    """Cache to npz, keyed by subject, so training doesn't reparse CSVs."""

    # Flatten segments into parallel arrays that npz can store
    subjects = np.array([segment.subject for segment in segments])
    starts = np.array([str(segment.start_time) for segment in segments])
    lengths = np.array([len(segment) for segment in segments])
    glucose = np.concatenate([segment.glucose for segment in segments])

    # Label the file with the gap policy: segments_split.npz or segments_interpolate.npz
    labeled_path = path.with_name(f"{path.stem}_{gap_policy}{path.suffix}")

    np.savez_compressed(
        labeled_path,
        subjects=subjects,
        starts=starts,
        lengths=lengths,
        glucose=glucose,
    )

    # Append to manifest history so you can see all past runs
    git_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    entry = {
        "date": pd.Timestamp.now().isoformat(),
        "git_commit": git_commit,
        "gap_policy": gap_policy,
        "n_subjects": len(set(segment.subject for segment in segments)),
        "n_segments": len(segments),
        "n_readings": int(glucose.shape[0]),
    }

    manifest_path = path.parent / "segments_manifest.json"

    # Load existing history or start fresh
    if manifest_path.exists():
        history = json.loads(manifest_path.read_text())
    else:
        history = []

    history.append(entry)
    manifest_path.write_text(json.dumps(history, indent=2))


def load_segments(path: Path) -> list[Segment]:
    """Load segments from a cached npz file (created by save_segments)."""
    cached_data = np.load(path)
    subjects = cached_data["subjects"]
    starts = cached_data["starts"]
    lengths = cached_data["lengths"]
    glucose = cached_data["glucose"]

    segments = []
    offset = 0
    for i in range(len(subjects)):
        seg_glucose = glucose[offset:offset + lengths[i]]
        segments.append(Segment(
            subject=str(subjects[i]),
            glucose=seg_glucose.astype(float),
            start_time=pd.Timestamp(str(starts[i])),
        ))
        offset += lengths[i]
    return segments


if __name__ == "__main__":

    # Baseline reference (from baselines.py):
    # persistence 20.39, ridge AR(12) 19.39, linear extrap 29.74->120.91,
    # per-subject variance +/-4.52 (range 11.91-29.31)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/CGMacros"))
    parser.add_argument("--out", type=Path, default=Path("data/segments.npz"))
    parser.add_argument("--gap-policy", choices=["split", "interpolate"], default="split")
    args = parser.parse_args()

    segments = load_all_segments(args.data_dir, args.gap_policy)

    num_subjects = len({seg.subject for seg in segments})
    num_readings = sum(len(seg) for seg in segments)
    num_windows = sum(max(0, len(seg) - HISTORY - 6 + 1) for seg in segments)
    print(f"subjects {num_subjects}   readings {num_readings:,}   windows(H=12,PH=30) {num_windows:,}")

    # Regression test: expected counts from the full 44-subject pipeline
    assert num_subjects == 44, f"expected 44 clean subjects, got {num_subjects}"
    assert num_readings == 124_808, f"expected 124,808 readings, got {num_readings}"
    assert num_windows == 123_435, f"expected 123,435 PH30 windows, got {num_windows}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_segments(segments, args.out, args.gap_policy)

    # subjects 44   readings 124,771   windows(H=12,PH=30) 123,411