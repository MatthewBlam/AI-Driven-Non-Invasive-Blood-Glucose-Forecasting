"""Poster-standardized variants of the four poster figures.

The symposium poster (guide/15-poster.md) stacks figures in fixed-width columns,
so they need a COMMON width and a near-4:3 aspect -- the paper originals are
wider landscapes. This script re-renders them through the ORIGINAL drawing
code (nothing is redrawn here, so the two versions cannot disagree), changing
only geometry: 300 dpi, aspect close to 4:3 at a 10-inch width -- except the
two middle-column figures, the input window and the LOSO chart, which render
WIDE (about 2:1) as a matched pair so the column can stack the results
table, the findings and both figures. Drop-in widths are normalized in
PowerPoint; the ratio is what each band pins down.

    python src/poster_figures.py                     # all four (~2-3 min)
    python src/poster_figures.py --only loso carb    # skip the slow/fast ones

Outputs (results/):
    poster_input_window.png       via plot_input_window.py (subprocess -- its CLI
                                  owns the window selection and verification)
    poster_loso_b2_per_subject.png  via figures.figure_loso_b2 (CSV-driven, fast)
    poster_meal_strata.png        via ablation_meals: B1 + the static T,M,Y,I arm
                                  (the HEADLINE strata arm -- the 9.1% -> 22.5%
                                  numbers -- matching results/meal_strata.png),
                                  22 checkpoint evaluations, the slow part
    poster_carb_response.png      via plot_carb_response.py (subprocess -- its
                                  CLI owns the meal alignment and verification)

Each render is measured after writing: the actual pixel aspect must land close
to 4:3 (the target band below), so a layout regression cannot silently produce
a mis-shaped poster figure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

POSTER_DPI = 300
POSTER_WIDTH_IN = 10.0
# Acceptable FINAL (post-crop) width/height bands; bbox_inches="tight"
# shifts each figure a little (crop shaves whitespace, right-edge labels add
# width), so close and CONSISTENT within a column is the poster requirement.
# Set per column on poster-assembly asks (2026-09-24): the middle column's
# input window and LOSO chart render WIDE (about 2:1) as a matched pair --
# same post-crop width AND height -- so that column can stack the results
# table, the findings and both figures; the right column's strata and carb
# figures are a shorter-than-4:3 pair at about 1.7. RATIO_BAND (about 4:3)
# is the default for any future figure.
RATIO_BAND = (1.20, 1.45)
MID_BAND = (1.55, 1.80)
WIDE_BAND = (1.80, 2.20)

# Figure sizes chosen so the post-crop result lands inside its band. Measured,
# not derived.
# Wider canvas than the common width: without the verdict labels the tight
# crop ends at the right spine, so the extra inches land in the AXES and the
# post-crop ratio reaches ~2:1 (the poster drop-in width is normalized in
# PowerPoint; the ratio is what matters).
INPUT_WINDOW_SIZE = (11.2, 5.3)
LOSO_SIZE = (10.3, 5.4)
# The right-column pair went shorter 2026-09-24 (poster-assembly ask): same
# 10-inch width, ~1.7 ratio instead of 4:3.
STRATA_SIZE = (POSTER_WIDTH_IN, 5.9)
CARB_SIZE = (POSTER_WIDTH_IN, 5.9)

# The strata arm must be the one the recorded headline numbers came from:
# static T,M,Y,I (PROJECT_STATE Part 6 -- B1 9.1% -> B2 22.5% post-meal skill).
STRATA_SPEC = [("B1", None, None), ("T,M,Y,I", "static", ("T", "M", "Y", "I"))]


def measure(path: Path, band: tuple[float, float] = RATIO_BAND) -> tuple[int, int, float]:
    """(width_px, height_px, ratio) of a rendered PNG, printed and band-checked."""
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    ratio = width / height
    print(f"  {path.name}: {width} x {height} px  "
          f"({width / POSTER_DPI:.2f} x {height / POSTER_DPI:.2f} in, ratio {ratio:.3f})")
    assert band[0] <= ratio <= band[1], \
        f"{path.name} ratio {ratio:.3f} outside {band} -- adjust its figsize"
    return width, height, ratio


def poster_input_window(results_dir: Path) -> Path:
    """Re-render via the script's own CLI so selection + verification stay owned
    by plot_input_window.py (same pattern as rebuild_all.py)."""
    out = results_dir / "poster_input_window.png"
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "plot_input_window.py"),
         "--out", str(out), "--dpi", str(POSTER_DPI), "--no-verdicts",
         "--figsize", str(INPUT_WINDOW_SIZE[0]), str(INPUT_WINDOW_SIZE[1])],
        cwd=REPO_ROOT)
    assert completed.returncode == 0, "plot_input_window.py failed"
    return out


def poster_carb(results_dir: Path) -> Path:
    """Re-render via the script's own CLI (same pattern as poster_input_window):
    the meal alignment, exclusion accounting and verification stay owned by
    plot_carb_response.py."""
    out = results_dir / "poster_carb_response.png"
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "plot_carb_response.py"),
         "--out", str(out), "--dpi", str(POSTER_DPI),
         "--figsize", str(CARB_SIZE[0]), str(CARB_SIZE[1])],
        cwd=REPO_ROOT)
    assert completed.returncode == 0, "plot_carb_response.py failed"
    return out


def poster_loso(segments, results_dir: Path) -> Path:
    import figures

    out_name = "poster_loso_b2_per_subject.png"
    figures.figure_loso_b2(segments, Namespace(results_dir=results_dir),
                           figsize=LOSO_SIZE, out_name=out_name, dpi=POSTER_DPI)
    return results_dir / out_name


def poster_strata(segments, results_dir: Path, ph: int = 30) -> Path:
    from ablation_meals import check_pairing, collect_ladder, figure_meal_strata, \
        stratified_report

    print(f"collecting B1 + static T,M,Y,I across the seeds "
          f"(~22 checkpoint evaluations, the slow part)...")
    results = collect_ladder(STRATA_SPEC, segments, results_dir, ph)
    check_pairing(results)
    strata_data = stratified_report(results, segments)
    assert strata_data, "stratified_report returned nothing -- are the runs present?"

    out = results_dir / "poster_meal_strata.png"
    figure_meal_strata(strata_data, out, figsize=STRATA_SIZE, dpi=POSTER_DPI)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the four poster figures at a standardized geometry")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--only", nargs="+",
                        choices=["input_window", "loso", "strata", "carb"],
                        help="render a subset (default: all four)")
    args = parser.parse_args()
    selected = args.only or ["input_window", "loso", "strata", "carb"]

    outputs = []
    if "input_window" in selected:
        print(f"\n[poster_input_window]")
        outputs.append((poster_input_window(args.results_dir), WIDE_BAND))

    if "carb" in selected:
        print(f"\n[poster_carb_response]")
        outputs.append((poster_carb(args.results_dir), MID_BAND))

    if {"loso", "strata"} & set(selected):
        from preprocess import load_segments
        segments = load_segments(args.data_dir / "segments_split.npz")
        if "loso" in selected:
            print(f"\n[poster_loso_b2_per_subject]")
            outputs.append((poster_loso(segments, args.results_dir), WIDE_BAND))
        if "strata" in selected:
            print(f"\n[poster_meal_strata]")
            outputs.append((poster_strata(segments, args.results_dir), MID_BAND))

    print(f"\nstandardized geometry check (per-column bands: middle pair "
          f"~2:1, right pair ~1.7):")
    for path, band in outputs:
        measure(path, band)


if __name__ == "__main__":
    main()
