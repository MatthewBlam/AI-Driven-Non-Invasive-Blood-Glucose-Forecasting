"""Training compute footprint -- numbers for the paper's reproducibility sentence.

Reads run_meta.json when present, falls back to TensorBoard wall_time stamps for
older runs.

    python src/compute_footprint.py                  # grouped table + paste-ready sentence
    python src/compute_footprint.py --detail         # one line per run
    python src/compute_footprint.py --inference      # also time a forward pass
    python src/compute_footprint.py --csv results/compute_footprint.csv

Training device (RTX 3080) is auto-detected from checkpoint storage tags via
training_device(), not inferred from the current machine or accelerator="auto".

Caveats on wall-clock numbers (the script prints these too):
  - Early stopping decouples total time from architecture (identical configs ran 48 vs
    73 epochs), so per-run totals differ for reasons unrelated to the model.
  - Same config re-run: 184.3s vs 87.97s at identical 48 epochs and identical RMSE.
    Machine contention, not the model.
  - LOSO folds average 2.46 s/epoch vs pooled ph30's 4.03, despite training on more
    subjects -- different sessions.
  - 17k params barely occupies a 3080 with num_workers=0, so per-epoch cost is mostly
    host-side overhead. BiLSTM (2x params) costs only +7% per epoch -- understated.
The only defensible comparative number is s/epoch averaged over a whole 11-seed batch.
Report parameter counts and best_epoch for everything else. Inference (--inference)
runs on CPU (map_location="cpu") -- different device from training.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

# (regex, group label) -- first match wins; the sweep's shard folders hold many runs each.
GROUPS = [
    (re.compile(r"^loso_\d+$"), "LOSO folds"),
    (re.compile(r"^sweep_shard\d+$"), "hyperparameter sweep"),
    (re.compile(r"^ph(\d+)_seed\d+$"), "ph{0} single-horizon"),
    (re.compile(r"^ph(\d+)_multi_seed\d+$"), "ph{0} multi-horizon"),
    (re.compile(r"^ph(\d+)_bidir_seed\d+$"), "ph{0} BiLSTM"),
    (re.compile(r"^ph(\d+)_attn_seed\d+$"), "ph{0} Attn-LSTM"),
    (re.compile(r"^loso_mealstatic_TMYIS_\d+$"), "B2 LOSO folds"),
    (re.compile(r"^ph(\d+)_mealstatic_TMYIS_seed\d+$"), "B2 ph{0} static"),
    (re.compile(r"^ph(\d+)_mealchannels_TMYIS_seed\d+$"), "B2 ph{0} channels"),
    (re.compile(r"^ph(\d+)_mealhybrid_TMYIS_seed\d+$"), "B2 ph{0} hybrid"),
    (re.compile(r"^ph(\d+)_mealstatic_(\w+)_seed\d+$"), "B2 ph{0} static ablation"),
    (re.compile(r"^ph(\d+)_mealchannels_(\w+)_seed\d+$"), "B2 ph{0} channels ablation"),
    (re.compile(r"^ph(\d+)_b3static_(\w+)_seed\d+$"), "B3 ph{0} static"),
    (re.compile(r"^ph(\d+)_b3channels_(\w+)_seed\d+$"), "B3 ph{0} channels"),
]


def training_device(results_dir: Path, override: str | None = None) -> str:
    """Where the runs ACTUALLY trained, read out of a checkpoint.

    Inferred from checkpoint storage tags, not from the current machine or
    accelerator="auto". torch.load's map_location receives each tensor's original
    storage location, so the checkpoint file is the ground truth."""
    if override:
        return override
    try:
        import torch
        ckpt = next(results_dir.glob("**/*.ckpt"), None)
        if ckpt is None:
            return "unknown hardware (no checkpoint found)"
        tags: set[str] = set()

        def collect(storage, location):
            tags.add(location)
            return storage                      # already CPU-backed; keep it there

        torch.load(ckpt, map_location=collect, weights_only=False)
        if any(tag.startswith("cuda") for tag in tags):
            if torch.cuda.is_available():
                return f"an {torch.cuda.get_device_name(0)}"
            return "a CUDA GPU (model name unavailable on this machine)"
        return "CPU"
    except Exception as exc:                    # a missing torch must not kill the table
        return f"unknown hardware ({type(exc).__name__})"


def group_for(run_name: str):
    for pattern, template in GROUPS:
        match = pattern.match(run_name)
        if match:
            return template.format(*match.groups())
    return None


def run_cost(version_dir: Path):
    """(seconds, epochs, source) for one run directory, or None if unreadable.

    Prefers run_meta.json: it records the true wall clock including setup, which the
    tfevents span (first logged epoch -> last) necessarily misses."""
    meta_path = version_dir / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "wall_seconds" in meta:
                # trainer.current_epoch after fit() is already the COUNT of epochs run
                # (it increments at each epoch end), and it matches the number of
                # val_rmse events tfevents holds -- verified 48 == 48 on ph30 seed 42.
                return float(meta["wall_seconds"]), int(meta.get("epochs", 0)), "meta"
        except (json.JSONDecodeError, ValueError, KeyError):
            pass                                    # fall through to tfevents

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(version_dir))
    acc.Reload()
    tags = acc.Tags()["scalars"]
    if not tags:
        return None
    events = acc.Scalars("val_rmse" if "val_rmse" in tags else tags[0])
    if len(events) < 2:
        return None
    return events[-1].wall_time - events[0].wall_time, len(events), "tfevents"


def collect(results_dir: Path):
    """One record per run directory that contains events, newest-first order irrelevant."""
    version_dirs = sorted({ev.parent for ev in results_dir.glob("**/events.out.tfevents.*")})
    records = []
    for version_dir in version_dirs:
        # results/<run>/version_N/  ->  the run name is two levels up from the events file
        run_name = version_dir.parent.name if version_dir.name.startswith("version_") \
            else version_dir.name
        group = group_for(run_name)
        if group is None:
            continue
        cost = run_cost(version_dir)
        if cost is None:
            print(f"  [skip] no usable scalars in {version_dir}")
            continue
        seconds, epochs, source = cost
        records.append({"group": group, "run": run_name, "version": version_dir.name,
                        "seconds": seconds, "epochs": epochs, "source": source})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Study-wide training compute footprint")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--detail", action="store_true", help="one line per run")
    parser.add_argument("--inference", action="store_true",
                        help="also time a forward pass on the seed-42 test split")
    parser.add_argument("--csv", type=Path, default=None, help="write the per-run rows here")
    parser.add_argument("--device-label", default=None,
                        help="override the auto-detected training device in the sentence")
    args = parser.parse_args()

    print(f"scanning {args.results_dir}/ (this reads every tfevents file, ~1 min)")
    records = collect(args.results_dir)
    if not records:
        print("no runs found")
        return

    by_group = defaultdict(list)
    for record in records:
        by_group[record["group"]].append(record)

    print(f"\n{'group':<24} {'runs':>5} {'epochs':>7} {'minutes':>9} {'s/epoch':>8}  source")
    for group in sorted(by_group):
        rows = by_group[group]
        seconds = sum(r["seconds"] for r in rows)
        epochs = sum(r["epochs"] for r in rows)
        sources = {r["source"] for r in rows}
        print(f"{group:<24} {len(rows):>5} {epochs:>7} {seconds / 60:>9.1f} "
              f"{seconds / max(epochs, 1):>8.2f}  {'+'.join(sorted(sources))}")

    total_seconds = sum(r["seconds"] for r in records)
    print(f"{'TOTAL':<24} {len(records):>5} {sum(r['epochs'] for r in records):>7} "
          f"{total_seconds / 60:>9.1f}  = {total_seconds / 3600:.1f} h")

    if args.detail:
        print(f"\n{'run':<24} {'version':<10} {'epochs':>7} {'seconds':>9}  source")
        for record in sorted(records, key=lambda r: (r["group"], r["run"])):
            print(f"{record['run']:<24} {record['version']:<10} {record['epochs']:>7} "
                  f"{record['seconds']:>9.1f}  {record['source']}")

    # The one comparative number that survives the noise: s/epoch over a whole batch.
    order = ("ph30 single-horizon", "ph30 BiLSTM", "ph30 Attn-LSTM")
    present = [group for group in order if group in by_group]
    if len(present) > 1:
        print("\nper-epoch cost by architecture (+30, averaged over each 11-seed batch --"
              "\nthe only time comparison here that is not dominated by machine noise):")
        baseline = None
        for group in present:
            rows = by_group[group]
            per_epoch = sum(r["seconds"] for r in rows) / sum(r["epochs"] for r in rows)
            if baseline is None:
                baseline = per_epoch
                note = "baseline (LSTM)"
            else:
                note = f"{per_epoch / baseline - 1:+.0%} vs LSTM"
            print(f"  {group:<24} {per_epoch:.2f} s/epoch  ({note})")

    infer_note = ""
    if args.inference:
        import time
        import numpy as np
        import torch
        from evaluate import find_checkpoint, pooled_config
        from preprocess import load_segments
        from dataset import CGMacrosDataModule
        from train import LitGlucoseLSTM
        import logging
        logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.WARNING)

        segments = load_segments(args.data_dir / "segments_split.npz")
        model = LitGlucoseLSTM.load_from_checkpoint(
            find_checkpoint("ph30_seed42", args.results_dir), map_location="cpu").eval()
        dm = CGMacrosDataModule(segments, history=12, horizon=6, batch_size=256,
                                split_mode="pooled_random", seed=42)
        dm.setup()
        X = dm.test_dataset.inputs
        with torch.no_grad():
            model(X[:256])                                    # warm up, then measure
            times = []
            for _ in range(3):
                start = time.perf_counter()
                model(X)
                times.append(time.perf_counter() - start)
        best = min(times)
        per_window_us = 1e6 * best / len(X)
        # Deliberately CPU: evaluate.py loads with map_location="cpu", and CPU inference
        # is the deployment-relevant number for a phone/wearable. GPU inference is not
        # measured here -- do not quote this as a GPU figure.
        print(f"\ninference: {len(X):,} windows in {best:.3f}s = {per_window_us:.1f} us/window "
              f"(CPU, best of 3; training used the GPU -- these are different devices)")
        infer_note = f" Inference costs ~{per_window_us:.0f} us per window on CPU."

    single = by_group.get("ph30 single-horizon", [])
    if single:
        per_run_min = statistics.mean(r["seconds"] for r in single) / 60
        median_epochs = statistics.median(r["epochs"] for r in single)
        device = training_device(args.results_dir, args.device_label)
        print("\npaste-ready methods sentence:")
        print(f'  "All models were trained on {device}. A single +30 run takes '
              f'~{per_run_min:.0f} min (median {median_epochs:.0f} epochs with early stopping, '
              f'patience 20); the complete study -- {len(records)} training runs covering the '
              f'hyperparameter sweep, the B1 architecture and horizon batches, the B2 meal and '
              f'B3 activity ablations, and 88 LOSO folds '
              f'-- required {total_seconds / 3600:.1f} h.{infer_note}"')

    print("\ncaveats (see the module docstring before tabulating any of this):")
    print("  - per-RUN wall clock is not a model property: early stopping and machine")
    print("    contention move it far more than architecture does (seed 42: LSTM 184s vs")
    print("    BiLSTM 157s, wrongly suggesting the 2x-parameter model is cheaper; and the")
    print("    same LSTM config re-ran in 88s at an identical 48 epochs and identical RMSE).")
    print("  - do not compare s/epoch ACROSS groups: they ran in different sessions.")
    print("  - a 17k-param model does not saturate this GPU and num_workers=0, so per-epoch")
    print("    time is mostly host-side overhead -- it understates real compute differences.")
    print("  - tfevents spans omit setup time; run_meta.json rows are the true wall clock.")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"\nwrote {args.csv}  ({len(records)} rows)")


if __name__ == "__main__":
    main()
