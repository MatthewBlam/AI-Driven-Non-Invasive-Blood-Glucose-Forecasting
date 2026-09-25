"""Rebuild every table and figure in the paper with one command.

    python src/rebuild_all.py                 # everything, in dependency order (~18 min)
    python src/rebuild_all.py --dry-run       # show what would run, change nothing
    python src/rebuild_all.py --only figures  # one step (see --help for the step keys)
    python src/rebuild_all.py --skip metrics  # e.g. reuse the existing metrics CSVs
    python src/rebuild_all.py --no-smoke      # skip the 15s reproducibility pre-check

No training -- inference over existing checkpoints + pandas over CSVs. Safe to re-run.

One ordering constraint: figures.py --figure dispersion reads metrics_loso.csv, so the
metrics step runs first. Everything else is independent.

A smoke check runs first: seed 42 must reproduce RMSE 17.25 / persistence 19.11. If
the splits aren't rebuilding, this catches it in 15 seconds instead of after 8 minutes
of plausible-looking wrong numbers.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Seed 42's recorded numbers -- the canary for "are the splits still reproducing?"
SMOKE_RMSE, SMOKE_PERSISTENCE, SMOKE_TOL = 17.25, 19.11, 0.01


@dataclass
class Step:
    key: str
    script: str
    args: list[str]
    runtime: str
    what: str
    outputs: list[str] = field(default_factory=list)
    # Declared here, next to the step, so a new raw-reading step cannot silently
    # miss the prerequisite warning in check_prerequisites.
    needs_raw: bool = False


STEPS = [
    Step("metrics", "metrics_table.py", ["--loso"], "~10 min",
         "every number in the results table (RMSE/MAE/NRMSE/skill/Parkes/lag/dispersion), "
         "pooled over 11 seeds plus the B1 and B2 LOSO arms (44 folds each)",
         ["metrics_table.md", "metrics_table.csv", "metrics_loso.csv"],
         needs_raw=True),
    Step("figures", "figures.py", [], "~2 min",
         "all eleven figures",
         ["parkes_grid.png", "parkes_grid_type1.png", "pred_vs_actual.png",
          "pred_vs_actual_pooled.png", "rmse_vs_horizon.png", "loso_per_subject.png",
          "sweep_null.png", "data_integrity.png", "dispersion.png", "error_bins.png",
          "seed_pairs.png", "learning_curves.png"],
         needs_raw=True),
    Step("cohort", "cohort_table.py", [], "~5 s",
         "Table 1, participant characteristics",
         ["cohort_table.md", "cohort_per_subject.csv"],
         needs_raw=True),
    Step("footprint", "compute_footprint.py",
         ["--inference", "--csv", "results/compute_footprint.csv"], "~1 min",
         "training/inference cost and the paste-ready methods sentence",
         ["compute_footprint.csv"]),
    Step("conformal", "conformal.py",
         ["--marginal", "--strata", "--subject-half", "--loso", "--b1-compare",
          "--figures"], "~10 min",
         "split-conformal intervals on the frozen B2 model: marginal/stratified/"
         "per-subject coverage, the Mondrian repair, and B1-vs-B2 interval width",
         ["conformal.md", "conformal_coverage.csv", "conformal_strata.csv",
          "conformal_newsubject.csv", "conformal_loso.csv", "conformal_width.csv",
          "conformal_coverage.png", "conformal_strata.png", "conformal_subjects.png",
          "conformal_width.png"],
         needs_raw=True),
    Step("input_window", "plot_input_window.py", [], "~1 min",
         "poster figure: one real input hour, one row per baseline "
         "(glucose window + starred target, meal card, heart rate)",
         ["input_window.png"],
         needs_raw=True),
    Step("carb_response", "plot_carb_response.py", [], "~10 s",
         "poster figure: mean post-meal glucose response by carb load "
         "(descriptive, no model)",
         ["carb_response.png"],
         needs_raw=True),
    Step("poster", "poster_figures.py", [], "~3 min",
         "poster-standardized variants of the four poster figures "
         "(common 10-inch width, ~4:3, 300 dpi; same drawing code)",
         ["poster_input_window.png", "poster_loso_b2_per_subject.png",
          "poster_meal_strata.png", "poster_carb_response.png"],
         needs_raw=True),
]

# One line per artifact, for the closing index. The full explanations live in
# PROJECT_STATE.md under the figure and table guide -- keep these as labels, not
# explanations, so the two cannot drift into disagreement.
ARTIFACT_INDEX = [
    ("cohort_table.md", "Table 1: who the 44 subjects are, split by glycemic class"),
    ("metrics_table.md", "the results table: every model beside persistence on the same split"),
    ("compute_footprint.csv", "what the study cost to train (188 runs, 11.7 h, RTX 3080)"),
    ("loso_per_subject.png", "per-subject LOSO error; ticks below bars = model lost"),
    ("pred_vs_actual.png", "one calm subject vs one volatile one: the failure case"),
    ("error_bins.png", "WHERE the model helps: by glucose level and by trend"),
    ("dispersion.png", "the model's spread vs reality's: 44/44 subjects under-dispersed"),
    ("rmse_vs_horizon.png", "error at +30/+60/+120 vs persistence"),
    ("parkes_grid.png", "clinical risk view: would an error change a treatment decision?"),
    ("sweep_null.png", "no hyperparameter beats split noise"),
    ("learning_curves.png", "nothing is under-trained"),
    ("seed_pairs.png", "the model wins on 11/11 seeds at +30, paired within split"),
    ("data_integrity.png", "the published 1-min series is interpolated; how we recovered it"),
    ("attention_weights.png", "where attention looks: 62.6% on the last 2 of 12 readings"),
    ("conformal.md", "interval coverage: marginal, by stratum, per subject, and width"),
    ("conformal_strata.png", "where the 90% guarantee fails (post-meal/rising) and the Mondrian fix"),
    ("conformal_subjects.png", "per-person coverage spread, repaired by personal calibration"),
    ("conformal_width.png", "interval width vs horizon; what meal features buy in width"),
    ("input_window.png", "poster: what each baseline sees for the same real hour"),
    ("carb_response.png", "poster: post-meal glucose response by carb load, saturating above ~40 g"),
]


def smoke_check() -> bool:
    """Seed 42 must reproduce, or nothing downstream is trustworthy."""
    import numpy as np
    from evaluate import find_checkpoint, load_test_predictions, pooled_config
    from preprocess import load_segments

    ckpt = find_checkpoint("ph30_seed42", REPO_ROOT / "results")
    if ckpt is None:
        print("  no results/ph30_seed42 checkpoint -- cannot verify; continuing anyway")
        return True

    segments = load_segments(REPO_ROOT / "data" / "segments_split.npz")
    y, preds, last, _ = load_test_predictions(ckpt, segments, pooled_config(42))
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    persistence = float(np.sqrt(np.mean((last - y) ** 2)))
    ok = (abs(rmse - SMOKE_RMSE) < SMOKE_TOL
          and abs(persistence - SMOKE_PERSISTENCE) < SMOKE_TOL)
    print(f"  seed 42: RMSE {rmse:.2f} (expect {SMOKE_RMSE})  "
          f"persistence {persistence:.2f} (expect {SMOKE_PERSISTENCE})  "
          f"-> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("  the split is not reproducing -- fix that before rebuilding anything, or the\n"
              "  tables will look plausible and be wrong. Check data/segments_split.npz.")
    return ok


def check_prerequisites(steps) -> None:
    """Warn about missing inputs up front rather than 4 minutes in."""
    needs_raw = any(s.needs_raw for s in steps)
    checks = [
        (REPO_ROOT / "data" / "segments_split.npz", "the preprocessed cache", True),
        (REPO_ROOT / "results", "checkpoints and CSVs", True),
    ]
    if needs_raw:
        checks.append((REPO_ROOT / "dataset" / "CGMacros" / "bio.csv",
                       "raw dataset (Table 1, the data-integrity figure, "
                       "and the B2 rows' meal features)", False))
    for path, what, fatal in checks:
        if not path.exists():
            print(f"  {'MISSING' if fatal else 'missing'}: {path}  <- {what}")
            if fatal:
                sys.exit(f"cannot continue without {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild every table and figure")
    keys = [s.key for s in STEPS]
    parser.add_argument("--only", choices=keys, nargs="+", help="run just these steps")
    parser.add_argument("--skip", choices=keys, nargs="+", default=[], help="skip these steps")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--no-smoke", action="store_true", help="skip the reproducibility check")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue after a failing step instead of stopping")
    args = parser.parse_args()

    steps = [s for s in STEPS if (not args.only or s.key in args.only) and s.key not in args.skip]
    if not steps:
        sys.exit("nothing selected")

    # The dispersion figure needs the metrics step's CSV; say so rather than failing later.
    if any(s.key == "figures" for s in steps) and not any(s.key == "metrics" for s in steps):
        if not (REPO_ROOT / "results" / "metrics_loso.csv").exists():
            print("note: figures need results/metrics_loso.csv for the dispersion panel, and\n"
                  "      the metrics step is not selected -- that one figure will be skipped.")

    print(f"rebuild plan (repo root: {REPO_ROOT}):")
    for i, step in enumerate(steps, 1):
        argstr = " ".join(step.args)
        print(f"  {i}. {step.key:<10} python src/{step.script} {argstr:<45} {step.runtime}")
        print(f"     {step.what}")
    print("  no training happens in any step; all of it is inference + pandas\n")

    if args.dry_run:
        return

    if not args.no_smoke:
        print("smoke check (does seed 42 still reproduce?):")
        if not smoke_check() and not args.keep_going:
            sys.exit("aborting -- rerun with --no-smoke to build anyway")
        print()

    check_prerequisites(steps)

    results, total = [], time.perf_counter()
    for i, step in enumerate(steps, 1):
        print(f"\n{'=' * 78}\n[{i}/{len(steps)}] {step.key}: python src/{step.script} "
              f"{' '.join(step.args)}\n{'=' * 78}")
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / step.script), *step.args],
            cwd=REPO_ROOT)                       # relative data paths resolve from the root
        elapsed = time.perf_counter() - started
        results.append((step, completed.returncode, elapsed))
        if completed.returncode != 0:
            print(f"\n!! {step.key} exited {completed.returncode} after {elapsed:.0f}s")
            if not args.keep_going:
                sys.exit("stopping (use --keep-going to continue past failures)")

    print(f"\n{'=' * 78}\nSUMMARY ({time.perf_counter() - total:.0f}s total)\n{'=' * 78}")
    for step, code, elapsed in results:
        print(f"  {'ok ' if code == 0 else 'FAIL'} {step.key:<10} {elapsed:>6.0f}s   "
              f"{', '.join(step.outputs[:3])}{' ...' if len(step.outputs) > 3 else ''}")

    print("\nwhat you just built (full explanations in PROJECT_STATE.md):")
    for name, description in ARTIFACT_INDEX:
        path = REPO_ROOT / "results" / name
        mark = " " if path.exists() else "?"
        print(f" {mark} results/{name:<26} {description}")
    missing = [n for n, _ in ARTIFACT_INDEX if not (REPO_ROOT / "results" / n).exists()]
    if missing:
        print(f"\n  '?' = not on disk: {', '.join(missing)}")
        print("      (attention_weights.png comes from src/plot_attention_weights.py)")


if __name__ == "__main__":
    main()
