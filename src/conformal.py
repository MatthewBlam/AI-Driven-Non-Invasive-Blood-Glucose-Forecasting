"""Split-conformal prediction intervals on the frozen forecasters.

Wraps the trained B2 checkpoints (static TMYIS, the headline model) in intervals
prediction +/- q_hat, where q_hat is the finite-sample-corrected quantile of the
model's absolute errors on a calibration set the model never benefited from.
Nothing here trains: every number is CPU inference over checkpoints that already
exist, so the whole study reruns in minutes.

The two claims under test, and where they are measured:
  - Is the coverage guarantee honest on average, and in the risky strata
    (post-meal, rising, high glucose)?           -> default run, --strata
  - Does it transfer to brand-new subjects, and does calibrating on a person's
    own first days repair it?                    -> --subject-half, --loso

Two invariants this file is built around:
  - Consecutive windows share 11 of 12 readings (stride 1), so a RANDOM
    calibration/evaluation split leaks near-duplicates and reports fake-perfect
    coverage. The honest split is temporal within each segment, with a purge gap
    of history-1+horizon windows so the two sides share zero readings. The random
    split is still computed -- as the measured warning, never for a claim.
  - Coverage claims use the per-seed spread over the 11 seeds as the noise band,
    never a binomial SE (autocorrelated windows make the binomial n a lie).

Outputs (results/):
    conformal_coverage.csv    one row per seed x split x alpha (default run)
    conformal_strata.csv      per-stratum coverage, global + Mondrian (--strata)
    conformal_newsubject.csv  per seed x orientation x subject     (--subject-half)
    conformal_loso.csv        one row per LOSO fold                (--loso)
    conformal_width.csv       B1 vs B2 width per horizon           (--b1-compare)
    conformal.md              the readable tables, rebuilt from whichever CSVs exist
    conformal_{coverage,strata,subjects,width}.png                 (--figures)

Usage (from the repo root):
    python src/conformal.py                    # self-checks + marginal coverage table
    python src/conformal.py --strata           # per-stratum coverage and Mondrian only
    python src/conformal.py --marginal --strata --subject-half --loso --b1-compare --figures
                                               # everything (the rebuild_all invocation)

Parts are independent flags; a bare invocation runs the marginal table. Every
invocation runs the self-check suite first; each check is shown able to fail.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import HISTORY
from dataset import split_subjects
from evaluate import (DEFAULT_SEEDS, RESULTS_DIR, iter_pooled_seeds, load_loso_subject,
                      meal_infix, run_tag)
from preprocess import load_segments

B2_INFIX = meal_infix("static", ("T", "M", "Y", "I", "S"))
DEFAULT_ALPHAS = (0.20, 0.10, 0.05)
HEADLINE_ALPHA = 0.10          # 90% is the reporting level of the glucose literature
DEFAULT_CALIB_FRAC = 0.5
LOSO_CALIB_FRAC = 0.25         # "the first quarter of the subject's own record"

# One invocation may run several parts that score the same checkpoints (the marginal
# table, the strata, and the subject split all read the same 11 runs). Predictions are
# deterministic, so cache them per (infix, ph, seed) for the life of the process.
_prediction_cache: dict[tuple, tuple] = {}


# ------------------------------------------------------------------ conformal core

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The split-conformal quantile: the k-th smallest calibration score, with
    k = ceil((n+1)(1-alpha)).

    The +1 buys the guarantee for the NEXT point rather than the points already
    seen; taking an order statistic (not an interpolated quantile) is what the
    theorem is proved for. Implemented as an explicit sort-and-index rather than
    the common np.quantile(..., method="higher") recipe: that recipe computes its
    position as level*(n-1) and lands one order statistic HIGHER than k whenever
    the position is fractional -- still valid (conservative), but not the
    estimator the theorem names, and it fails the n=99 worked example
    check_quantile pins (the 90th smallest of 1..99 is 90, the recipe says 91)."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    assert k <= n, \
        f"{n} calibration scores is too few to promise {1 - alpha:.0%} " \
        f"(needs the {k}-th smallest of {n}) -- silently clamping would fake the guarantee"
    return float(np.sort(scores)[k - 1])


def coverage(y: np.ndarray, preds: np.ndarray, mask: np.ndarray, q_hat: float) -> float:
    """Fraction of the masked windows whose actual value lands inside preds +/- q_hat."""
    return float(np.mean(np.abs(y[mask] - preds[mask]) <= q_hat))


# ------------------------------------------------------------------ split machinery

def purge_windows(horizon: int) -> int:
    """How many windows to drop at the calibration/evaluation boundary.

    A window starting at reading s touches inputs s..s+HISTORY-1 and the target
    s+HISTORY-1+horizon. The last calibration window therefore reaches
    HISTORY-1+horizon readings past its own start, and every window starting within
    that reach shares a reading with it. Note the reach is set by the TARGET, not
    the inputs: windows 12..11+horizon positions later share no input but ingest
    the calibration window's target -- check_purge's corruption catches exactly
    the off-by-a-region error of stopping at the first input-disjoint index."""
    return HISTORY - 1 + horizon


def test_segments_for_seed(segments, seed: int):
    """One seed's test segments in the exact order CGMacrosDataModule.setup builds
    them: subjects in split_subjects' shuffled order, each subject's segments in
    file order. This order is what makes window bookkeeping line up with the
    loader's arrays, so it is constructed the same way ablation_activity does."""
    subject_ids = sorted({segment.subject for segment in segments})
    _, _, test_subjects = split_subjects(subject_ids, seed=seed)

    segments_by_subject: dict[str, list] = {}
    for segment in segments:
        segments_by_subject.setdefault(segment.subject, []).append(segment)

    ordered = []
    for subject in test_subjects:
        ordered.extend(segments_by_subject.get(subject, []))
    return ordered


def window_counts(test_segments, horizon: int) -> list[int]:
    """Windows per segment: preprocess.make_windows' N = L - history - horizon + 1."""
    counts = []
    for segment in test_segments:
        counts.append(max(len(segment) - HISTORY - horizon + 1, 0))
    return counts


def temporal_masks(counts: list[int], calib_frac: float, purge: int,
                   reverse: bool = False):
    """(calibration, evaluation, purged) boolean masks over the concatenated windows.

    Per segment: the first calib_frac of windows calibrate, the next `purge`
    windows are dropped, the rest evaluate. reverse=True swaps the two roles (the
    temporal-drift control); the purged band stays purged either way. A segment
    shorter than the calibration span plus the purge contributes no evaluation
    windows -- correct behavior, not an edge case to patch."""
    calibration_parts, evaluation_parts, purged_parts = [], [], []
    for n in counts:
        calibration = np.zeros(n, dtype=bool)
        evaluation = np.zeros(n, dtype=bool)
        purged = np.zeros(n, dtype=bool)

        n_calibration = int(n * calib_frac)
        gap_end = min(n_calibration + purge, n)
        calibration[:n_calibration] = True
        purged[n_calibration:gap_end] = True
        evaluation[gap_end:] = True

        if reverse:
            calibration, evaluation = evaluation.copy(), calibration.copy()

        calibration_parts.append(calibration)
        evaluation_parts.append(evaluation)
        purged_parts.append(purged)

    return (np.concatenate(calibration_parts), np.concatenate(evaluation_parts),
            np.concatenate(purged_parts))


def random_masks(n_total: int, seed: int):
    """The WRONG split -- a uniformly random half/half with no purge. Computed so
    the leak can be measured and shown next to the honest line, never for a claim."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_total)
    calibration = np.zeros(n_total, dtype=bool)
    calibration[order[: n_total // 2]] = True
    return calibration, ~calibration


def subject_half_masks(subjects: np.ndarray, seed: int):
    """Both orientations of the new-subject split: calibrate on 4 of the seed's 8
    test subjects, evaluate on the other 4, then swap. The swap is free -- same
    predictions, different masks -- and doubles the evaluated subjects."""
    unique_subjects = sorted(set(subjects.tolist()))
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(unique_subjects))
    half = len(shuffled) // 2

    orientations = []
    for calibration_group, evaluation_group in ((shuffled[:half], shuffled[half:]),
                                                (shuffled[half:], shuffled[:half])):
        calibration = np.isin(subjects, calibration_group)
        evaluation = np.isin(subjects, evaluation_group)
        orientations.append((calibration, evaluation, evaluation_group))
    return orientations


# ------------------------------------------------------------------ self-checks

def check_quantile(trials: int = 2000, n_calibration: int = 19, n_eval: int = 1000) -> None:
    """Prove the quantile on data with a known answer, and prove the test can fail.

    n_calibration = 19 is chosen so the theory gives an EXACT target: with
    continuous scores the expected coverage of the k-th order statistic is
    k/(n+1) = ceil(20 * 0.9)/20 = 0.900 exactly, so the estimator is tested
    against an equality (within simulation tolerance), not just a bound. The
    corruption half matters as much as the pass half: if the NAIVE quantile
    (no +1, interpolated) also landed on 0.900, this test could not detect the
    difference it exists to detect."""
    worked_example = conformal_quantile(np.arange(1.0, 100.0), 0.10)
    assert worked_example == 90.0, f"n=99 worked example: got {worked_example}, want 90.0"

    rng = np.random.default_rng(0)
    corrected_coverages, naive_coverages = [], []
    for _ in range(trials):
        calibration_scores = np.abs(rng.standard_normal(n_calibration))
        eval_scores = np.abs(rng.standard_normal(n_eval))
        corrected_coverages.append(np.mean(eval_scores <= conformal_quantile(calibration_scores, 0.10)))
        naive_coverages.append(np.mean(eval_scores <= np.quantile(calibration_scores, 0.90)))

    mean_corrected = float(np.mean(corrected_coverages))
    mean_naive = float(np.mean(naive_coverages))
    expected = np.ceil((n_calibration + 1) * 0.9) / (n_calibration + 1)
    assert abs(mean_corrected - expected) < 0.005, \
        f"expected coverage {expected:.3f}, simulated {mean_corrected:.4f}"
    assert mean_naive < expected - 0.02, \
        f"the naive formula came out at {mean_naive:.4f} -- too close to " \
        f"{expected:.3f} for this test to detect the difference it exists to detect"

    try:
        conformal_quantile(np.ones(9), 0.05)
    except AssertionError:
        small_n_raised = True
    else:
        small_n_raised = False
    assert small_n_raised, "n=9 at alpha=0.05 must raise (needs the 10th smallest of 9)"

    print(f"  check_quantile OK       (corrected {mean_corrected:.4f} ~ {expected:.3f} "
          f"exact, naive {mean_naive:.4f} below, small-n raises)")


def _touched_readings(window_starts: np.ndarray, horizon: int) -> set:
    """Every reading index a set of windows touches: the input span plus the target."""
    touched = set()
    for start in window_starts:
        touched.update(range(start, start + HISTORY))
        touched.add(start + HISTORY - 1 + horizon)
    return touched


def check_purge(counts: list[int], horizon: int,
                calib_frac: float = DEFAULT_CALIB_FRAC) -> None:
    """Prove the two sides share zero readings at the derived purge, and that one
    window less provably leaks (a purge test that cannot fail is no test)."""
    for purge_offset, expect_clean in ((0, True), (-1, False)):
        purge = purge_windows(horizon) + purge_offset
        calibration, evaluation, _ = temporal_masks(counts, calib_frac, purge)
        assert calibration.sum() > 0 and evaluation.sum() > 0, \
            "a mask came back empty -- the disjointness check below would pass vacuously"

        shared_readings = 0
        position = 0
        for n in counts:
            calibration_starts = np.where(calibration[position:position + n])[0]
            evaluation_starts = np.where(evaluation[position:position + n])[0]
            shared_readings += len(_touched_readings(calibration_starts, horizon)
                                   & _touched_readings(evaluation_starts, horizon))
            position += n

        if expect_clean:
            assert shared_readings == 0, \
                f"purge={purge}: calibration and evaluation share {shared_readings} readings"
        else:
            assert shared_readings > 0, \
                f"purge={purge} should leak but found nothing -- the detector is broken"

    print(f"  check_purge OK          (purge={purge_windows(horizon)} clean, "
          f"purge-1 provably leaks)")


def run_checks(segments, ph: int, calib_frac: float) -> None:
    """The full self-check suite, run before any table on every invocation.

    check_purge gets the SAME calib_frac the tables will use -- a certification of
    masks the run never builds would be no certification at all."""
    print("self-checks:")
    check_quantile()
    counts = window_counts(test_segments_for_seed(segments, DEFAULT_SEEDS[0]), ph // 5)
    check_purge(counts, ph // 5, calib_frac)


# ------------------------------------------------------------------ loading

def load_pooled_run(segments, seed: int, ph: int, infix: str,
                    results_dir: Path = RESULTS_DIR):
    """(y, preds, last_input, subjects, window counts) for one pooled run, with the
    window bookkeeping asserted against the loader before anything is computed
    from it.

    Delegates to evaluate.iter_pooled_seeds rather than re-implementing the
    tag -> checkpoint -> config -> predictions pipeline -- that generator is the
    one place tag parsing lives, and a private copy here would drift from it."""
    cache_key = (str(results_dir), infix, ph, seed)
    if cache_key not in _prediction_cache:
        loaded = list(iter_pooled_seeds(segments, seeds=[seed], ph=ph, infix=infix,
                                        results_dir=results_dir))
        assert len(loaded) == 1, \
            f"no checkpoint under {results_dir / run_tag(seed, ph, infix)}/"
        _, _, y, preds, last_input, subjects = loaded[0]
        _prediction_cache[cache_key] = (y, preds, last_input, subjects)

    y, preds, last_input, subjects = _prediction_cache[cache_key]
    counts = window_counts(test_segments_for_seed(segments, seed), ph // 5)
    assert sum(counts) == len(y), \
        f"seed {seed}: window bookkeeping says {sum(counts)} windows, loader returned {len(y)}"
    return y, preds, last_input, subjects, counts


# ------------------------------------------------------------------ formatting

def fmt_mean_std(values, digits: int = 4) -> str:
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return f"{array[0]:.{digits}f}"
    return f"{array.mean():.{digits}f} +/- {array.std(ddof=1):.{digits}f}"


# ------------------------------------------------------------------ marginal coverage

def marginal_rows(segments, ph: int, seeds, alphas, calib_frac: float,
                  results_dir: Path) -> pd.DataFrame:
    """The default run: coverage and width per seed for the three splits.

    Forward is the deployment direction (calibrate on the days already seen) and
    the primary. Reverse is the temporal-drift control. Random is the leak,
    reported as a warning line only."""
    purge = purge_windows(ph // 5)
    rows = []
    for seed in seeds:
        y, preds, last_input, subjects, counts = load_pooled_run(
            segments, seed, ph, B2_INFIX, results_dir)
        scores = np.abs(y - preds)

        calibration_f, evaluation_f, purged = temporal_masks(counts, calib_frac, purge)
        assert calibration_f.sum() + evaluation_f.sum() + purged.sum() == len(y), \
            f"seed {seed}: the three masks do not partition the windows"
        calibration_r, evaluation_r, _ = temporal_masks(counts, calib_frac, purge, reverse=True)
        calibration_x, evaluation_x = random_masks(len(y), seed)

        # Each split carries ITS OWN purge count: the random split purges nothing --
        # that is what makes it the leak -- and its rows must say so.
        splits = [("forward", calibration_f, evaluation_f, int(purged.sum())),
                  ("reverse", calibration_r, evaluation_r, int(purged.sum())),
                  ("random", calibration_x, evaluation_x, 0)]
        for split_name, calibration, evaluation, n_purged in splits:
            for alpha in alphas:
                q_hat = conformal_quantile(scores[calibration], alpha)
                rows.append({"seed": seed, "ph": ph, "calib_frac": calib_frac,
                             "split": split_name, "alpha": alpha,
                             "coverage": coverage(y, preds, evaluation, q_hat),
                             "width": 2 * q_hat, "q_hat": q_hat,
                             "n_calib": int(calibration.sum()),
                             "n_eval": int(evaluation.sum()),
                             "n_purged": n_purged})
    return pd.DataFrame(rows)


def leak_audit(counts: list[int], seed: int) -> tuple[float, float]:
    """How close a random split's evaluation windows sit to calibration: the
    median nearest-calibration distance in window positions, and the fraction
    adjacent (distance <= 1 means 11 of 12 readings shared)."""
    n_total = sum(counts)
    calibration, evaluation = random_masks(n_total, seed)
    distances = []
    position = 0
    for n in counts:
        calibration_starts = np.where(calibration[position:position + n])[0]
        evaluation_starts = np.where(evaluation[position:position + n])[0]
        if len(calibration_starts) and len(evaluation_starts):
            insertion = np.searchsorted(calibration_starts, evaluation_starts)
            left_neighbor = np.where(
                insertion > 0,
                evaluation_starts - calibration_starts[np.maximum(insertion - 1, 0)],
                np.iinfo(int).max)
            right_neighbor = np.where(
                insertion < len(calibration_starts),
                calibration_starts[np.minimum(insertion, len(calibration_starts) - 1)]
                - evaluation_starts,
                np.iinfo(int).max)
            distances.append(np.minimum(left_neighbor, right_neighbor))
        position += n
    all_distances = np.concatenate(distances)
    return float(np.median(all_distances)), float(np.mean(all_distances <= 1))


def report_marginal(frame: pd.DataFrame, segments, ph: int) -> None:
    print("\n" + "=" * 78)
    print(f"  MARGINAL COVERAGE -- B2 static TMYIS, ph{ph}, "
          f"{frame['seed'].nunique()} seeds")
    print(f"  temporal split: calibrate early, purge {purge_windows(ph // 5)} windows, "
          f"evaluate late")
    print("=" * 78)

    forward = frame[frame["split"] == "forward"]
    print(f"\n  n_calib {int(forward['n_calib'].mean())} avg, "
          f"n_eval {int(forward['n_eval'].mean())} avg, "
          f"n_purged {int(forward['n_purged'].mean())} avg per seed")

    for split_name in ("forward", "reverse", "random"):
        subset = frame[frame["split"] == split_name]
        label = split_name + (" (THE LEAK -- never use)" if split_name == "random" else "")
        print(f"\n  {label}:")
        for alpha in sorted(subset["alpha"].unique(), reverse=True):
            at_alpha = subset[subset["alpha"] == alpha]
            print(f"    alpha {alpha:.2f} (nominal {1 - alpha:.0%}):  "
                  f"coverage {fmt_mean_std(at_alpha['coverage'])}   "
                  f"width {fmt_mean_std(at_alpha['width'], 2)} mg/dL")
        if split_name in ("forward", "reverse") and HEADLINE_ALPHA in subset["alpha"].values:
            headline = subset[subset["alpha"] == HEADLINE_ALPHA].sort_values("seed")
            print("    per-seed coverage at alpha 0.10: "
                  + " ".join(f"{value:.4f}" for value in headline["coverage"]))

    # The drift instrument needs the headline alpha; a custom --alpha list without
    # 0.10 gets the per-alpha tables above and skips this cleanly.
    if HEADLINE_ALPHA in frame["alpha"].values:
        forward_headline = forward[forward["alpha"] == HEADLINE_ALPHA].set_index("seed")["coverage"]
        reverse_headline = frame[(frame["split"] == "reverse")
                                 & (frame["alpha"] == HEADLINE_ALPHA)].set_index("seed")["coverage"]
        gap_pp = (forward_headline - reverse_headline) * 100
        spread = f" +/- {gap_pp.std(ddof=1):.2f}" if len(gap_pp) > 1 else ""
        print(f"\n  paired forward - reverse gap at alpha 0.10: "
              f"{gap_pp.mean():+.2f}{spread} pp, "
              f"forward higher on {int((gap_pp > 0).sum())}/{len(gap_pp)} seeds")

    counts = window_counts(test_segments_for_seed(segments, DEFAULT_SEEDS[0]), ph // 5)
    median_distance, adjacent_fraction = leak_audit(counts, DEFAULT_SEEDS[0])
    print(f"  leak audit (seed {DEFAULT_SEEDS[0]}, random split): median "
          f"nearest-calib distance {median_distance:.0f} window(s); "
          f"{adjacent_fraction:.1%} of eval windows adjacent to a calibration window")


# ------------------------------------------------------------------ strata + Mondrian

def prediction_time_state(segments, ph: int):
    """One-time prep for the per-window state loader: the raw meal table and the
    per-segment activity aggregates, keyed by id() of the same Segment objects."""
    from activity import aggregate_segments, load_activity
    from meals import load_meals

    raw_dir = Path("dataset/CGMacros")
    meals = load_meals(raw_dir)
    activity_frames = load_activity(raw_dir)
    aggregates = aggregate_segments(segments, activity_frames)
    aggregate_by_id = {id(segment): aggregate
                      for segment, aggregate in zip(segments, aggregates)}
    return meals, aggregate_by_id


def strata_rows(segments, ph: int, seeds, calib_frac: float,
                results_dir: Path) -> pd.DataFrame:
    """Per-stratum evaluation coverage under the GLOBAL q_hat, plus the Mondrian
    repair (per-group q_hat from the same calibration side) for two partitions.

    The pseudo-stratum -- a size-matched random draw from the evaluation windows --
    is the control: carving windows at random must sit at the marginal level, or
    the real strata's deviations are slicing noise, not signal."""
    from ablation_activity import load_window_state, stratum_masks

    meals, aggregate_by_id = prediction_time_state(segments, ph)
    purge = purge_windows(ph // 5)
    alpha = HEADLINE_ALPHA
    rows = []

    for seed in seeds:
        y, preds, last_input, subjects, counts = load_pooled_run(
            segments, seed, ph, B2_INFIX, results_dir)
        scores = np.abs(y - preds)
        calibration, evaluation, _ = temporal_masks(counts, calib_frac, purge)
        global_q = conformal_quantile(scores[calibration], alpha)

        dt_meal, dt_active, trend, last_reading = load_window_state(
            seed, segments, meals, aggregate_by_id, ph)
        assert len(dt_meal) == len(y), \
            f"seed {seed}: {len(dt_meal)} state rows for {len(y)} windows"
        # The glucose anchor: a stratification built one row off still produces a
        # plausible table, and this equality is the only thing that rules it out.
        assert np.allclose(last_reading, last_input, atol=1e-3), \
            f"seed {seed}: window state is not glucose-anchored -- rows are misaligned"

        named_masks = stratum_masks(dt_meal, dt_active, trend)
        named_masks.append(("current < 100", last_input < 100))
        named_masks.append(("current 100-180", (last_input >= 100) & (last_input <= 180)))
        named_masks.append(("current > 180", last_input > 180))

        # The Mondrian partitions and the pseudo-control's size are BUILT FROM the
        # named masks above, not re-derived from thresholds: stratum_masks owns the
        # cutoff definitions, and duplicating them here is how a future threshold
        # tune would silently split one table across two stratum definitions.
        mask_by_name = dict(named_masks)
        trend_names = ("rising > +1", "flat -1..+1", "falling < -1")
        meal_recent = mask_by_name["meal <=30 min"]

        n_pseudo = int(meal_recent[evaluation].sum())
        rng = np.random.default_rng(seed)
        pseudo = np.zeros(len(y), dtype=bool)
        pseudo[rng.choice(np.where(evaluation)[0], size=n_pseudo, replace=False)] = True
        named_masks.append(("pseudo (size-matched random)", pseudo))

        for stratum_name, stratum_mask in named_masks:
            eval_in_stratum = stratum_mask & evaluation
            if eval_in_stratum.sum() == 0:
                continue
            rows.append({"seed": seed, "ph": ph, "calib_frac": calib_frac,
                         "scheme": "global", "stratum": stratum_name,
                         "coverage": float(np.mean(scores[eval_in_stratum] <= global_q)),
                         "width": 2 * global_q,
                         "n_eval": int(eval_in_stratum.sum()),
                         "median_abs_resid": float(np.median(scores[eval_in_stratum]))})

        # Partition sanity: the trend triple must account for every evaluation
        # window, and its window-weighted coverage must rebuild the pooled number.
        trend_rows = [row for row in rows
                      if row["seed"] == seed and row["stratum"] in trend_names]
        assert sum(row["n_eval"] for row in trend_rows) == int(evaluation.sum()), \
            f"seed {seed}: the trend triple does not partition the evaluation windows"
        weighted = sum(row["coverage"] * row["n_eval"] for row in trend_rows) \
            / evaluation.sum()
        pooled = coverage(y, preds, evaluation, global_q)
        assert np.isclose(weighted, pooled), \
            f"seed {seed}: weighted stratum coverage {weighted:.6f} != pooled {pooled:.6f}"

        mondrian_partitions = {
            "mondrian_trend": {name: mask_by_name[name] for name in trend_names},
            "mondrian_meal": {"meal <=30 min": meal_recent,
                              "no meal <=30": ~meal_recent},
        }
        for scheme_name, groups in mondrian_partitions.items():
            for group_name, group_mask in groups.items():
                group_scores = scores[calibration & group_mask]
                assert len(group_scores) >= 19, \
                    f"seed {seed} {group_name}: {len(group_scores)} calibration scores " \
                    f"is below the alpha=0.05 floor"
                group_q = conformal_quantile(group_scores, alpha)
                eval_in_group = evaluation & group_mask
                if eval_in_group.sum() == 0:
                    # Same skip the global loop applies: a group that calibrates but
                    # never evaluates must not write a NaN row.
                    continue
                rows.append({"seed": seed, "ph": ph, "calib_frac": calib_frac,
                             "scheme": scheme_name, "stratum": group_name,
                             "coverage": float(np.mean(scores[eval_in_group] <= group_q)),
                             "width": 2 * group_q,
                             "n_eval": int(eval_in_group.sum()),
                             "median_abs_resid": float(np.median(scores[eval_in_group]))})
    return pd.DataFrame(rows)


def report_strata(frame: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print(f"  STRATA -- evaluation coverage under the GLOBAL q-hat, "
          f"alpha {HEADLINE_ALPHA} (nominal {1 - HEADLINE_ALPHA:.0%})")
    print("=" * 78)
    global_rows = frame[frame["scheme"] == "global"]
    print(f"\n  {'stratum':<30} {'n/seed':>7} {'coverage':>20} {'med |resid|':>12}")
    print("  " + "-" * 74)
    for stratum_name in global_rows["stratum"].unique():
        subset = global_rows[global_rows["stratum"] == stratum_name]
        print(f"  {stratum_name:<30} {int(subset['n_eval'].mean()):>7} "
              f"{fmt_mean_std(subset['coverage']):>20} "
              f"{subset['median_abs_resid'].mean():>12.2f}")

    print("\n  Mondrian (per-group q-hat from the calibration side of the same split):")
    for scheme_name in ("mondrian_trend", "mondrian_meal"):
        scheme_rows = frame[frame["scheme"] == scheme_name]
        for stratum_name in scheme_rows["stratum"].unique():
            subset = scheme_rows[scheme_rows["stratum"] == stratum_name]
            print(f"    [{scheme_name.split('_')[1]}] {stratum_name:<16} "
                  f"coverage {fmt_mean_std(subset['coverage'])}   "
                  f"width {fmt_mean_std(subset['width'], 2)} mg/dL")


# ------------------------------------------------------------------ new subjects

def subject_half_rows(segments, ph: int, seeds, results_dir: Path) -> pd.DataFrame:
    """Calibrate on half the seed's test subjects, evaluate on the other half,
    both orientations. Per-subject coverage is the finding; the pooled number is
    only the marginal-guarantee context it sits inside."""
    rows = []
    for seed in seeds:
        y, preds, last_input, subjects, counts = load_pooled_run(
            segments, seed, ph, B2_INFIX, results_dir)
        scores = np.abs(y - preds)
        for orientation, (calibration, evaluation, evaluation_group) in enumerate(
                subject_half_masks(subjects, seed)):
            q_hat = conformal_quantile(scores[calibration], HEADLINE_ALPHA)
            pooled_coverage = float(np.mean(scores[evaluation] <= q_hat))
            for subject in evaluation_group:
                subject_mask = subjects == subject
                rows.append({"seed": seed, "ph": ph, "orientation": orientation,
                             "subject": subject,
                             "coverage": float(np.mean(scores[subject_mask] <= q_hat)),
                             "pooled_coverage": pooled_coverage,
                             "width": 2 * q_hat,
                             "n_windows": int(subject_mask.sum())})
    return pd.DataFrame(rows)


def report_subject_half(frame: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("  NEW SUBJECTS -- calibrate on 4 test subjects, evaluate on the other 4")
    print("=" * 78)
    pooled_per_run = frame.groupby(["seed", "orientation"])["pooled_coverage"].first()
    per_subject = frame.groupby("subject")["coverage"].mean().sort_values()
    print(f"  pooled coverage over {len(pooled_per_run)} orientation-runs: "
          f"{fmt_mean_std(pooled_per_run)}")
    print(f"  per-subject coverage ({len(per_subject)} subjects): "
          f"min {per_subject.iloc[0]:.3f}  median {per_subject.median():.3f}  "
          f"max {per_subject.iloc[-1]:.3f}")
    below = per_subject[per_subject < 0.88]
    print(f"  subjects below 0.88: "
          + ", ".join(f"{subject}={value:.3f}" for subject, value in below.items()))


# ------------------------------------------------------------------ LOSO personal

def loso_rows(segments, ph: int, results_dir: Path) -> pd.DataFrame:
    """44 LOSO folds with PERSONAL calibration: the first quarter of the held-out
    subject's own record calibrates (purged as usual), the rest evaluates. Answers
    "is ~2.4 days of your own data enough to make the guarantee honest for you?" """
    # The B2 LOSO folds were trained at +30 only (guide 11 Part 7). At any other
    # ph the bookkeeping assert below would fire anyway -- fail here with the cause.
    assert ph == 30, \
        f"--loso needs ph 30 (the B2 LOSO folds exist only at +30), got ph {ph}"
    results_csv = results_dir / f"loso{B2_INFIX}_results.csv"
    logged = pd.read_csv(results_csv)
    subject_ids = logged["subject"].astype(int).map(lambda n: f"{n:03d}").tolist()

    segments_by_subject: dict[str, list] = {}
    for segment in segments:
        segments_by_subject.setdefault(segment.subject, []).append(segment)

    purge = purge_windows(ph // 5)
    rows = []
    for subject in subject_ids:
        y, preds, last_input, subjects = load_loso_subject(
            subject, segments, results_dir=results_dir, infix=B2_INFIX)
        counts = window_counts(segments_by_subject[subject], ph // 5)
        assert sum(counts) == len(y), \
            f"{subject}: window bookkeeping says {sum(counts)}, loader returned {len(y)}"

        calibration, evaluation, _ = temporal_masks(counts, LOSO_CALIB_FRAC, purge)
        scores = np.abs(y - preds)
        assert calibration.sum() >= 19, \
            f"{subject}: {int(calibration.sum())} calibration windows is below the " \
            f"alpha=0.05 floor"
        assert evaluation.sum() > 0, \
            f"{subject}: calibration plus purge consumed the whole record -- " \
            f"no evaluation windows, the fold's coverage would be NaN"
        q_hat = conformal_quantile(scores[calibration], HEADLINE_ALPHA)
        rows.append({"subject": subject, "ph": ph, "calib_frac": LOSO_CALIB_FRAC,
                     "coverage": float(np.mean(scores[evaluation] <= q_hat)),
                     "width": 2 * q_hat,
                     "n_calib": int(calibration.sum()),
                     "n_eval": int(evaluation.sum())})
    return pd.DataFrame(rows)


def report_loso(frame: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("  LOSO -- personal calibration: first quarter of the subject's own record")
    print("=" * 78)
    ordered = frame.sort_values("coverage")
    print(f"  {len(frame)} folds: coverage {fmt_mean_std(frame['coverage'])}  "
          f"min {ordered['coverage'].iloc[0]:.3f}  "
          f"median {frame['coverage'].median():.3f}  "
          f"max {ordered['coverage'].iloc[-1]:.3f}")
    print(f"  n_calib per subject: min {frame['n_calib'].min()}  "
          f"median {int(frame['n_calib'].median())}")
    below = ordered[ordered["coverage"] < 0.85]
    print(f"  subjects below 0.85: "
          + ", ".join(f"{row.subject}={row.coverage:.3f}" for row in below.itertuples()))


# ------------------------------------------------------------------ width economics

def width_rows(segments, seeds, calib_frac: float, results_dir: Path) -> pd.DataFrame:
    """B1 vs B2 interval width at matched honest coverage, per horizon. The pairing
    gift, asserted: at each (seed, horizon) the two models' targets must be
    byte-identical, or the comparison is void."""
    rows = []
    for ph in (30, 60, 120):
        purge = purge_windows(ph // 5)
        for model_label, infix in (("B1", ""), ("B2", B2_INFIX)):
            for seed in seeds:
                y, preds, last_input, subjects, counts = load_pooled_run(
                    segments, seed, ph, infix, results_dir)
                calibration, evaluation, _ = temporal_masks(counts, calib_frac, purge)
                q_hat = conformal_quantile(np.abs(y - preds)[calibration], HEADLINE_ALPHA)
                rows.append({"ph": ph, "model": model_label, "seed": seed,
                             "width": 2 * q_hat,
                             "coverage": coverage(y, preds, evaluation, q_hat)})
        for seed in seeds:
            y_b1 = _prediction_cache[(str(results_dir), "", ph, seed)][0]
            y_b2 = _prediction_cache[(str(results_dir), B2_INFIX, ph, seed)][0]
            assert np.array_equal(y_b1, y_b2), \
                f"ph{ph} seed {seed}: B1/B2 targets differ -- pairing broken"
    return pd.DataFrame(rows)


def report_width(frame: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print(f"  WIDTH ECONOMICS -- alpha {HEADLINE_ALPHA}, forward split, paired by seed")
    print("=" * 78)
    for ph in (30, 60, 120):
        b1 = frame[(frame["ph"] == ph) & (frame["model"] == "B1")].set_index("seed")
        b2 = frame[(frame["ph"] == ph) & (frame["model"] == "B2")].set_index("seed")
        delta = b1["width"] - b2["width"]
        print(f"  ph{ph:>3}:  B1 width {fmt_mean_std(b1['width'], 2)}   "
              f"B2 width {fmt_mean_std(b2['width'], 2)}   "
              f"B2 narrower by {fmt_mean_std(delta, 2)} mg/dL "
              f"({int((delta > 0).sum())}/{len(delta)} seeds)   "
              f"[coverage B2 {fmt_mean_std(b2['coverage'])}]")


# ------------------------------------------------------------------ markdown

def write_markdown(results_dir: Path) -> None:
    """Rebuild results/conformal.md from whichever conformal CSVs exist on disk.

    The CSVs are the single source (each one written from its part's one
    computation), so the markdown is a view that can never disagree with them --
    and a partial invocation cannot clobber sections it did not recompute."""
    lines = ["# Conformal prediction intervals", "",
             "Generated by `python src/conformal.py` (see `src/SCRIPTS.md`). "
             "Per-seed and per-subject data: the `conformal_*.csv` files beside this. "
             "All intervals are prediction +/- q_hat on the frozen B2 static TMYIS "
             "model; nominal level 90% unless a table says otherwise."]

    coverage_csv = results_dir / "conformal_coverage.csv"
    if coverage_csv.exists():
        frame = pd.read_csv(coverage_csv)
        # Columns follow whatever alphas the CSV holds (largest first), so a custom
        # --alpha run cannot leave NaN cells; the width column prefers the headline.
        alphas_present = sorted(frame["alpha"].unique(), reverse=True)
        width_alpha = HEADLINE_ALPHA if HEADLINE_ALPHA in alphas_present else alphas_present[0]
        header_cells = ["Split"]
        for alpha in alphas_present:
            header_cells.append(f"nominal {1 - alpha:.0%}")
        header_cells.append(f"width @{1 - width_alpha:.0%} (mg/dL)")
        lines += ["", "## Marginal coverage (temporal split, purged)", "",
                  "| " + " | ".join(header_cells) + " |",
                  "|" + "---|" * len(header_cells)]
        for split_name in ("forward", "reverse", "random"):
            subset = frame[frame["split"] == split_name]
            cells = [split_name + (" (THE LEAK)" if split_name == "random" else "")]
            for alpha in alphas_present:
                cells.append(fmt_mean_std(subset[subset["alpha"] == alpha]["coverage"]))
            cells.append(fmt_mean_std(subset[subset["alpha"] == width_alpha]["width"], 2))
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "_Forward = calibrate early (deployment direction). The random "
                  "split leaks on stride-1 windows and is shown only as the warning._"]

    strata_csv = results_dir / "conformal_strata.csv"
    if strata_csv.exists():
        frame = pd.read_csv(strata_csv)
        global_rows = frame[frame["scheme"] == "global"]
        lines += ["", "## Per-stratum coverage under the global q-hat (nominal 90%)", "",
                  "| Stratum | n/seed | coverage | med \\|resid\\| (mg/dL) |",
                  "|---|---|---|---|"]
        for stratum_name in global_rows["stratum"].unique():
            subset = global_rows[global_rows["stratum"] == stratum_name]
            lines.append(f"| {stratum_name} | {int(subset['n_eval'].mean())} | "
                         f"{fmt_mean_std(subset['coverage'])} | "
                         f"{subset['median_abs_resid'].mean():.2f} |")
        lines += ["", "| Mondrian group | coverage | width (mg/dL) |", "|---|---|---|"]
        for scheme_name in ("mondrian_trend", "mondrian_meal"):
            scheme_rows = frame[frame["scheme"] == scheme_name]
            for stratum_name in scheme_rows["stratum"].unique():
                subset = scheme_rows[scheme_rows["stratum"] == stratum_name]
                lines.append(f"| {stratum_name} | {fmt_mean_std(subset['coverage'])} | "
                             f"{fmt_mean_std(subset['width'], 2)} |")
        lines += ["", "_The pseudo-stratum is a size-matched random control: it must sit "
                  "at the marginal level, or the real strata's deviations would be "
                  "slicing noise. Mondrian = per-group q-hat from the same calibration "
                  "side._"]

    newsubject_csv = results_dir / "conformal_newsubject.csv"
    if newsubject_csv.exists():
        frame = pd.read_csv(newsubject_csv, dtype={"subject": str})
        pooled = frame.groupby(["seed", "orientation"])["pooled_coverage"].first()
        per_subject = frame.groupby("subject")["coverage"].mean()
        lines += ["", "## New subjects (subject-half calibration, nominal 90%)", "",
                  f"Pooled coverage {fmt_mean_std(pooled)} over {len(pooled)} "
                  f"orientation-runs; per-subject min {per_subject.min():.3f} / median "
                  f"{per_subject.median():.3f} / max {per_subject.max():.3f} "
                  f"({int((per_subject < 0.88).sum())} of {len(per_subject)} below 0.88). "
                  "Per-subject dots: see `conformal_subjects.png`. No per-subject CIs -- "
                  "within-subject autocorrelation makes them untrustworthy."]

    loso_csv = results_dir / "conformal_loso.csv"
    if loso_csv.exists():
        frame = pd.read_csv(loso_csv, dtype={"subject": str})
        below = frame[frame["coverage"] < 0.85].sort_values("coverage")
        below_text = ", ".join(f"{row.subject} ({row.coverage:.3f})"
                               for row in below.itertuples()) or "none"
        lines += ["", "## LOSO with personal calibration (nominal 90%)", "",
                  f"{len(frame)} folds, first quarter of each subject's own record as "
                  f"calibration (n_calib median {int(frame['n_calib'].median())} ~ 2.4 "
                  f"days): coverage {fmt_mean_std(frame['coverage'])}, min "
                  f"{frame['coverage'].min():.3f} / median "
                  f"{frame['coverage'].median():.3f} / max "
                  f"{frame['coverage'].max():.3f}. Below 0.85: {below_text}."]

    width_csv = results_dir / "conformal_width.csv"
    if width_csv.exists():
        frame = pd.read_csv(width_csv)
        lines += ["", "## Width economics (alpha 0.10, paired by seed)", "",
                  "| Horizon | B1 width | B2 width | B2 narrower by | seeds |",
                  "|---|---|---|---|---|"]
        for ph in (30, 60, 120):
            b1 = frame[(frame["ph"] == ph) & (frame["model"] == "B1")].set_index("seed")
            b2 = frame[(frame["ph"] == ph) & (frame["model"] == "B2")].set_index("seed")
            delta = b1["width"] - b2["width"]
            lines.append(f"| +{ph} | {fmt_mean_std(b1['width'], 2)} | "
                         f"{fmt_mean_std(b2['width'], 2)} | {fmt_mean_std(delta, 2)} | "
                         f"{int((delta > 0).sum())}/{len(delta)} |")

    lines += ["", "_Noise band: per-seed spread over the 11 seeds, never a binomial SE "
              "(stride-1 windows are autocorrelated). Coverage claims require "
              "|mean - nominal| > 2.0 pp AND >= 9/11 seeds in the same direction. State "
              "the calibration source next to every coverage number._", ""]
    out_path = results_dir / "conformal.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_path}")


# ------------------------------------------------------------------ figures

def make_figures(results_dir: Path) -> None:
    """The three exhibit figures plus the coverage panel, drawn from the CSVs so a
    figure can never disagree with its table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nominal = 1 - HEADLINE_ALPHA

    coverage_csv = results_dir / "conformal_coverage.csv"
    if coverage_csv.exists():
        frame = pd.read_csv(coverage_csv)
        headline = frame[frame["alpha"] == HEADLINE_ALPHA]
        fig, ax = plt.subplots(figsize=(6, 4))
        split_names = ("forward", "reverse", "random")
        for x_position, split_name in enumerate(split_names):
            values = headline[headline["split"] == split_name]["coverage"]
            ax.plot(np.full(len(values), x_position), values, "o", color="C0",
                    alpha=0.6, markersize=5)
            ax.plot(x_position, values.mean(), "_", color="k", markersize=22,
                    markeredgewidth=2)
        ax.axhline(nominal, color="k", linestyle="--", linewidth=1,
                   label=f"nominal {nominal:.0%}")
        ax.set_xticks(range(len(split_names)))
        ax.set_xticklabels(["forward\n(deployment)", "reverse\n(drift control)",
                            "random\n(THE LEAK)"])
        ax.set_ylabel("empirical coverage")
        ax.set_title("Marginal coverage at nominal 90% (one dot per seed)")
        ax.legend(loc="lower left", frameon=False)
        fig.tight_layout()
        out = results_dir / "conformal_coverage.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    strata_csv = results_dir / "conformal_strata.csv"
    if strata_csv.exists():
        frame = pd.read_csv(strata_csv)
        global_rows = frame[frame["scheme"] == "global"]
        stratum_names = list(global_rows["stratum"].unique())
        fig, ax = plt.subplots(figsize=(8, 6))
        for y_position, stratum_name in enumerate(stratum_names):
            values = global_rows[global_rows["stratum"] == stratum_name]["coverage"]
            is_control = stratum_name.startswith("pseudo")
            color = "C2" if is_control else "C0"
            ax.plot(values, np.full(len(values), y_position), "o", color=color,
                    alpha=0.55, markersize=4)
            ax.plot(values.mean(), y_position, "|", color="k", markersize=14,
                    markeredgewidth=2)
        mondrian = frame[frame["scheme"].str.startswith("mondrian")]
        for stratum_name in mondrian["stratum"].unique():
            if stratum_name in stratum_names:
                subset = mondrian[mondrian["stratum"] == stratum_name]
                ax.plot(subset["coverage"].mean(), stratum_names.index(stratum_name),
                        "D", color="C1", markersize=6, zorder=5)
        ax.plot([], [], "D", color="C1", markersize=6, label="Mondrian (per-group q-hat)")
        ax.plot([], [], "o", color="C2", alpha=0.55, label="pseudo-stratum control")
        ax.axvline(nominal, color="k", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(stratum_names)))
        ax.set_yticklabels(stratum_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"empirical coverage (nominal {nominal:.0%} dashed)")
        ax.set_title("Coverage by prediction-time stratum, global vs Mondrian q-hat")
        ax.legend(loc="lower left", frameon=False, fontsize=8)
        fig.tight_layout()
        out = results_dir / "conformal_strata.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    newsubject_csv = results_dir / "conformal_newsubject.csv"
    loso_csv = results_dir / "conformal_loso.csv"
    if newsubject_csv.exists() and loso_csv.exists():
        newsubject = pd.read_csv(newsubject_csv, dtype={"subject": str})
        loso = pd.read_csv(loso_csv, dtype={"subject": str})
        per_subject = newsubject.groupby("subject")["coverage"].mean()
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        panels = [(axes[0], per_subject.values,
                   f"cross-subject calibration\n({len(per_subject)} subjects)"),
                  (axes[1], loso["coverage"].values,
                   f"personal calibration, LOSO\n({len(loso)} subjects)")]
        for ax, values, title in panels:
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(values))
            ax.plot(jitter, values, "o", color="C0", alpha=0.6, markersize=5)
            ax.plot(0, values.mean(), "_", color="k", markersize=26, markeredgewidth=2)
            ax.axhline(nominal, color="k", linestyle="--", linewidth=1)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_xlim(-0.5, 0.5)
        axes[0].set_ylabel(f"per-subject coverage (nominal {nominal:.0%} dashed)")
        fig.suptitle("The guarantee per person: spread, and its repair", fontsize=11)
        fig.tight_layout()
        out = results_dir / "conformal_subjects.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    width_csv = results_dir / "conformal_width.csv"
    if width_csv.exists():
        frame = pd.read_csv(width_csv)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        horizons = (30, 60, 120)
        for model_label, color in (("B1", "C0"), ("B2", "C1")):
            means = [frame[(frame["ph"] == ph)
                           & (frame["model"] == model_label)]["width"].mean()
                     for ph in horizons]
            axes[0].plot(horizons, means, "o-", color=color, label=model_label)
            for ph in horizons:
                values = frame[(frame["ph"] == ph)
                               & (frame["model"] == model_label)]["width"]
                axes[0].plot(np.full(len(values), ph), values, "o", color=color,
                             alpha=0.25, markersize=4)
        axes[0].set_xlabel("horizon (min)")
        axes[0].set_ylabel("interval width at nominal 90% (mg/dL)")
        axes[0].set_xticks(horizons)
        axes[0].legend(frameon=False)
        axes[0].set_title("Width vs horizon")
        for x_position, ph in enumerate(horizons):
            b1 = frame[(frame["ph"] == ph) & (frame["model"] == "B1")].set_index("seed")
            b2 = frame[(frame["ph"] == ph) & (frame["model"] == "B2")].set_index("seed")
            delta = (b1["width"] - b2["width"]).values
            axes[1].plot(np.full(len(delta), x_position), delta, "o", color="C0",
                         alpha=0.6, markersize=5)
            axes[1].plot(x_position, delta.mean(), "_", color="k", markersize=22,
                         markeredgewidth=2)
        axes[1].axhline(0, color="k", linewidth=1)
        axes[1].set_xticks(range(len(horizons)))
        axes[1].set_xticklabels([f"+{ph}" for ph in horizons])
        axes[1].set_ylabel("B1 width - B2 width (mg/dL)")
        axes[1].set_title("What meal features buy (paired per seed)")
        fig.tight_layout()
        out = results_dir / "conformal_width.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


# ------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split-conformal intervals on the frozen B2 forecaster")
    parser.add_argument("--ph", type=int, default=30, choices=(30, 60, 120))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--alpha", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--calib-frac", type=float, default=DEFAULT_CALIB_FRAC)
    parser.add_argument("--marginal", action="store_true",
                        help="the marginal coverage table (forward/reverse/random); "
                             "runs by default when no part flag is given")
    parser.add_argument("--strata", action="store_true",
                        help="per-stratum coverage + the Mondrian repair (ph30)")
    parser.add_argument("--subject-half", action="store_true",
                        help="new-subject split: calibrate on 4 test subjects")
    parser.add_argument("--loso", action="store_true",
                        help="44 LOSO folds with personal calibration")
    parser.add_argument("--b1-compare", action="store_true",
                        help="B1 vs B2 interval width across all three horizons")
    parser.add_argument("--figures", action="store_true",
                        help="draw the PNGs from whichever CSVs exist")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.WARNING)
    segments = load_segments(args.data_dir / "segments_split.npz")

    run_checks(segments, args.ph, args.calib_frac)

    # The marginal study is a part like any other: a bare invocation runs it, but a
    # flagged run (e.g. --figures, which only redraws PNGs from the CSVs on disk)
    # must not pay 11 checkpoint loads for a table it never reads -- nor overwrite
    # conformal_coverage.csv as a side effect.
    other_parts_requested = (args.strata or args.subject_half or args.loso
                             or args.b1_compare or args.figures)
    if args.marginal or not other_parts_requested:
        frame = marginal_rows(segments, args.ph, args.seeds, tuple(args.alpha),
                              args.calib_frac, args.results_dir)
        report_marginal(frame, segments, args.ph)
        coverage_csv = args.results_dir / "conformal_coverage.csv"
        frame.to_csv(coverage_csv, index=False)
        print(f"\nwrote {coverage_csv}")

    if args.strata:
        strata_frame = strata_rows(segments, args.ph, args.seeds, args.calib_frac,
                                   args.results_dir)
        report_strata(strata_frame)
        strata_csv = args.results_dir / "conformal_strata.csv"
        strata_frame.to_csv(strata_csv, index=False)
        print(f"\nwrote {strata_csv}")

    if args.subject_half:
        subject_frame = subject_half_rows(segments, args.ph, args.seeds, args.results_dir)
        report_subject_half(subject_frame)
        newsubject_csv = args.results_dir / "conformal_newsubject.csv"
        subject_frame.to_csv(newsubject_csv, index=False)
        print(f"\nwrote {newsubject_csv}")

    if args.loso:
        loso_frame = loso_rows(segments, args.ph, args.results_dir)
        report_loso(loso_frame)
        loso_csv = args.results_dir / "conformal_loso.csv"
        loso_frame.to_csv(loso_csv, index=False)
        print(f"\nwrote {loso_csv}")

    if args.b1_compare:
        width_frame = width_rows(segments, args.seeds, args.calib_frac, args.results_dir)
        report_width(width_frame)
        width_csv = args.results_dir / "conformal_width.csv"
        width_frame.to_csv(width_csv, index=False)
        print(f"\nwrote {width_csv}")

    write_markdown(args.results_dir)

    if args.figures:
        make_figures(args.results_dir)


if __name__ == "__main__":
    main()
