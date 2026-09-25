"""Baseline yardsticks for CGMacros glucose forecasting.

Persistence RMSE has no hyperparameters, so if it disagrees with the reference values
after preprocessing changes, something is wrong with the data.

Baselines, in increasing order of sophistication:
    constant     yhat = mean(y_train)              -- learned-nothing floor
    persistence  yhat = g(t)                       -- the main baseline
    linear       yhat = g(t) + (g(t)-g(t-1)) * k  -- naive trend extrapolation
    ridge AR(12) yhat = w . [g(t-11)..g(t)] + b   -- linear ceiling on this input

Usage:
    python src/baselines.py --data-dir /path/to/CGMacros
    python src/baselines.py --segments data/segments.npz     # your own preprocessing
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import DEXCOM_PERIOD_MIN, load_all_segments

warnings.filterwarnings("ignore", category=FutureWarning)

HISTORY = 12                       # 60 min of 5-min readings
HORIZONS_MIN = (30, 60, 120)


def build_segments(data_dir: Path) -> dict[str, list[np.ndarray]]:
    """Real 5-min Dexcom knots, split into contiguous runs. Subject 007 excluded."""
    segments = load_all_segments(data_dir)
    segs: dict[str, list[np.ndarray]] = {}
    for segment in segments:
        segs.setdefault(segment.subject, []).append(segment.glucose)
    return segs


def make_windows(segments: list[np.ndarray], k: int, H: int = HISTORY):
    """(N, H) history and (N,) target at +k steps. Zero-copy view, then one copy."""
    Xs, ys = [], []
    for seg in segments:
        n = len(seg) - H - k + 1
        if n <= 0:
            continue
        idx = np.arange(H)[None, :] + np.arange(n)[:, None]
        Xs.append(seg[idx])
        ys.append(seg[H + k - 1: H + k - 1 + n])
    if not Xs:
        return np.empty((0, H)), np.empty((0,))
    return np.concatenate(Xs), np.concatenate(ys)


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2)))


def mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err)))


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0):
    """Closed-form ridge with an unpenalized intercept. No sklearn needed."""
    A = np.c_[X, np.ones(len(X))]
    P = np.eye(A.shape[1]) * lam
    P[-1, -1] = 0.0                              # don't penalize the intercept
    return np.linalg.solve(A.T @ A + P, A.T @ y)


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.c_[X, np.ones(len(X))] @ w


def pooled_table(segs: dict[str, list[np.ndarray]]) -> None:
    all_segs = [seg for subj_segs in segs.values() for seg in subj_segs]
    print(f"\n{'=' * 76}\nPOOLED over all windows  (n_subjects={len(segs)}, H={HISTORY})\n{'=' * 76}")
    for ph in HORIZONS_MIN:
        k = ph // DEXCOM_PERIOD_MIN
        X, y = make_windows(all_segs, k)
        last, prev = X[:, -1], X[:, -2]
        rows = [
            ("constant  = mean(y)", y - y.mean()),
            ("persistence = g(t)", y - last),
            ("linear extrapolation", y - (last + (last - prev) * k)),
        ]
        print(f"\nPH = +{ph:3d} min   ({len(X):,} windows)")
        for name, err in rows:
            print(f"  {name:<24} RMSE {rmse(err):7.2f}   MAE {mae(err):7.2f}")


def per_subject_table(segs: dict[str, list[np.ndarray]]) -> None:
    """LOSO reports the MEAN OF PER-SUBJECT RMSEs, not the pooled RMSE. They differ."""
    print(f"\n{'=' * 76}\nPER-SUBJECT persistence  (this is what LOSO averages)\n{'=' * 76}")
    for ph in HORIZONS_MIN:
        k = ph // DEXCOM_PERIOD_MIN
        scores = []
        for subject_id, subject_segs in segs.items():
            X, y = make_windows(subject_segs, k)
            if len(X):
                scores.append(rmse(y - X[:, -1]))
        score_array = np.array(scores)
        print(f"  PH=+{ph:3d}   mean {score_array.mean():6.2f} +/- {score_array.std():5.2f}   "
              f"[min {score_array.min():5.2f}, max {score_array.max():5.2f}]  over {len(score_array)} subjects")
    print("\n  The +/- is large. Compare models with a PAIRED test (scipy.stats.wilcoxon),")
    print("  not by differencing two averages.")


def subject_holdout_table(segs: dict[str, list[np.ndarray]], reps: int = 5, n_test: int = 9) -> None:
    """Ridge needs an honest, subject-disjoint holdout. This is the linear ceiling."""
    print(f"\n{'=' * 76}\nSUBJECT-DISJOINT HOLDOUT  ({reps} reps x {n_test} test subjects)\n{'=' * 76}")
    rng = np.random.default_rng(7)
    subject_ids = sorted(segs)
    for ph in HORIZONS_MIN:
        k = ph // DEXCOM_PERIOD_MIN
        fold_results = []
        for _ in range(reps):
            test_subjects = set(rng.choice(subject_ids, n_test, replace=False))
            train_segs = [seg for sid in subject_ids if sid not in test_subjects for seg in segs[sid]]
            test_segs = [seg for sid in subject_ids if sid in test_subjects for seg in segs[sid]]
            Xtr, ytr = make_windows(train_segs, k)
            Xte, yte = make_windows(test_segs, k)
            mu, sd = Xtr.mean(), Xtr.std()                       # fit on TRAIN ONLY
            weights = ridge_fit((Xtr - mu) / sd, ytr)
            fold_results.append((rmse(yte - Xte[:, -1]),
                                 rmse(yte - ridge_predict((Xte - mu) / sd, weights))))
        results_array = np.array(fold_results)
        persistence_mean = results_array[:, 0].mean()
        ridge_mean = results_array[:, 1].mean()
        print(f"  PH=+{ph:3d}   persistence {persistence_mean:6.2f}   ridge-AR({HISTORY}) {ridge_mean:6.2f}   "
              f"skill {100 * (persistence_mean - ridge_mean) / persistence_mean:+5.1f}%")
    print("\n  Your LSTM target: beat ridge-AR(12). A 13-parameter linear model is not")
    print("  a low bar -- it is the bar. If the LSTM loses, it is broken, not 'underperforming'.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--data-dir", type=Path, help="raw CGMacros directory")
    g.add_argument("--segments", type=Path, help="npz from your own src/preprocess.py")
    args = ap.parse_args()

    if args.data_dir:
        segs = build_segments(args.data_dir)
    else:
        z = np.load(args.segments)
        subjects = z["subjects"]
        lengths = z["lengths"]
        glucose = z["glucose"]

        # Split the flat glucose array back into per-segment arrays
        arrays = np.split(glucose, np.cumsum(lengths[:-1]))

        # Group by subject
        segs: dict[str, list[np.ndarray]] = {}
        for sid, arr in zip(subjects, arrays):
            segs.setdefault(sid, []).append(arr)

    total_readings = sum(len(seg) for subj_segs in segs.values() for seg in subj_segs)
    print(f"subjects: {len(segs)}   real Dexcom readings in segments: {total_readings:,}")

    pooled_table(segs)
    per_subject_table(segs)
    subject_holdout_table(segs)


if __name__ == "__main__":
    main()
