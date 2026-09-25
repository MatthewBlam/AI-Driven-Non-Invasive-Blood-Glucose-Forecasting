"""Shared test-set prediction extractor.

Every evaluation metric and figure needs the same three arrays on a checkpoint's OWN
test split: the true future glucose, the model's prediction, and what persistence
would have said.  That extraction lives here once, so each metric on top of it is a
couple of lines.  It is what diagnose_prediction_lag() does, minus the correlations.

Note: predictions and targets are already raw mg/dL, but the model's inputs are
z-scored -- so only `last_input` needs un-scaling on the way out.

Two loaders, same four return values:
    load_test_predictions(ckpt, segments, config)   pooled split (seed decides it)
    load_loso_subject(subject, segments, config)    one LOSO fold (its own checkpoint)

Consumers: metrics_table.py, figures.py.

Run directly to smoke-check that a run reproduces its recorded numbers:
    python src/evaluate.py                          # ph30 seed 42 -> RMSE 17.25 / persist 19.11
    python src/evaluate.py --ph 60 --seed 49
    python src/evaluate.py --tag ph30_attn_seed42   # ph and seed read out of the name
    python src/evaluate.py --loso 039               # a LOSO fold -> RMSE 29.29
"""

from __future__ import annotations

import argparse
import re
from argparse import Namespace
from functools import lru_cache
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch

from dataset import B3_MEAL_BLOCKS, B3_MEAL_ENCODING, CGMacrosDataModule
from model import count_params
from preprocess import load_segments
from train import LitGlucoseLSTM

RESULTS_DIR = Path("results")
DEFAULT_SEEDS = tuple(range(42, 53))       # the 11 pooled seeds used for cross-seed evaluation


def _extract(model, dm):
    """Pull the arrays off a set-up datamodule's test split, all raw mg/dL.

    Shared by both loaders below so the pooled and LOSO paths cannot drift apart
    (the same reasoning as train.py sharing compute_persistence_rmse).

    Returns five things; the public loaders expose the first four. The fifth is the
    whole un-scaled input window, needed by anything that conditions on the recent
    trend. By construction history_mgdl[:, -1] IS last_input.

    B2: inputs are (N, history, 1 + n_channels) and channel 0 is always the z-scored
    glucose, so every mg/dL line below is unchanged. `statics` is (N, 0) under B1."""
    X = dm.test_dataset.inputs                         # (N, history, 1+n_ch), z-scored
    static = dm.test_dataset.statics                   # (N, K_s); (N, 0) under B1
    with torch.no_grad():
        preds = model(X, static).numpy().squeeze()     # (N,) or (N, H), mg/dL
    y_actual = dm.test_dataset.targets.numpy().squeeze()                   # (N,) or (N, H)
    history_mgdl = X[:, :, 0].numpy() * dm.training_std + dm.training_mean  # (N, history)
    last_input = history_mgdl[:, -1]                                       # (N,)
    subjects = np.array(dm.test_dataset.subject_ids)                       # (N,)
    return y_actual, preds, last_input, subjects, history_mgdl


def _meal_kwargs(config) -> dict:
    """The meal arguments a datamodule needs, or {} under B1.

    A B2 checkpoint was trained against features built from ITS OWN split's cleaning
    constants, so rebuilding the split means rebuilding those too -- which is exactly what
    CGMacrosDataModule.setup() does from the training subject list. Reading the raw meals
    here keeps that the only place cleaning is ever fitted."""
    blocks = getattr(config, "meal_blocks", None)
    if not blocks:
        return {}
    return dict(meal_blocks=tuple(blocks),
                meal_encoding=getattr(config, "meal_encoding", "static"),
                meals=_load_meals_cached(Path(getattr(config, "raw_dir", "dataset/CGMacros"))))


def _activity_kwargs(config) -> dict:
    """The activity arguments a datamodule needs, or {} under B1/B2 -- the B3 counterpart
    of _meal_kwargs, for the same reason: ActivityStats is only ever fitted inside
    setup(), from the rebuilt split's own training subjects."""
    blocks = getattr(config, "activity_blocks", None)
    if not blocks:
        return {}
    return dict(activity_blocks=tuple(blocks),
                activity_encoding=getattr(config, "activity_encoding", "static"),
                activity=_load_activity_cached(Path(getattr(config, "raw_dir", "dataset/CGMacros"))))


@lru_cache(maxsize=2)
def _load_meals_cached(raw_dir: Path):
    """44 CSVs re-read once per seed would cost ~15 s across an 11-seed loop for a table
    that never changes. The frame is treated as read-only downstream (apply_macro_cleaning
    copies), so one shared instance is safe."""
    from meals import load_meals
    return load_meals(raw_dir)


@lru_cache(maxsize=2)
def _load_activity_cached(raw_dir: Path):
    """Same argument as _load_meals_cached: the raw minute frames never change across an
    evaluation loop, and everything downstream reads them without mutating."""
    from activity import load_activity
    return load_activity(raw_dir)


def _pooled_model_and_dm(checkpoint_path, segments, config):
    """Load a pooled checkpoint and rebuild its exact test split."""
    L.seed_everything(config.seed, workers=True)

    model = LitGlucoseLSTM.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()

    dm = CGMacrosDataModule(
        segments,
        history=config.history,
        horizon=config.ph // 5,
        batch_size=config.batch_size,
        split_mode="pooled_random",
        multi_horizon=config.multi_horizon,
        seed=config.seed,
        **_meal_kwargs(config),
        **_activity_kwargs(config),
    )
    dm.setup()
    return model, dm


def load_test_predictions(checkpoint_path, segments, config):
    """Return (y_actual, preds, last_input, subjects), all raw mg/dL, on the
    checkpoint's own test split.

    Single-horizon: y_actual and preds are (N,). Multi-horizon: (N, H), one column
    per horizon. last_input and subjects are always (N,)."""
    return _extract(*_pooled_model_and_dm(checkpoint_path, segments, config))[:4]


def load_test_predictions_with_history(checkpoint_path, segments, config):
    """load_test_predictions plus the un-scaled input window (N, history) in mg/dL.

    Separate function rather than a flag so the documented 4-tuple contract that every
    other evaluation consumer relies on cannot shift under them."""
    return _extract(*_pooled_model_and_dm(checkpoint_path, segments, config))


def load_loso_subject(subject: str, segments, config=None, results_dir: Path = RESULTS_DIR,
                      infix: str = ""):
    """Same four arrays, for ONE LOSO fold, from that fold's own checkpoint.

    The fold's val_subjects are read back out of the run's results CSV so the scaler
    is fit on exactly the subjects this model trained on -- rebuild the split any
    other way and the z-scoring silently shifts.

    `infix` selects which LOSO run: "" is B1 (loso_{subject}, loso_results.csv);
    "_mealstatic_TMYIS" is the B2 run train.py writes (loso_mealstatic_TMYIS_*);
    "_b3static_HRFAE" a B3 run. When no config is given, the feature settings are
    parsed from the infix -- the parse_feature_tag discipline, so a B2/B3 fold cannot
    be scored without its features.

    Needed because the pooled seed-42 test set is only 8 subjects and neither 039
    nor 046 (the two LOSO losses) is in it, so the paper's "bad day" panel has to
    come from their own folds."""
    if config is None:
        meal_encoding, meal_blocks, activity_encoding, activity_blocks = parse_feature_tag(infix)
        config = pooled_config(42, meal_blocks=meal_blocks,
                               meal_encoding=meal_encoding or "static",
                               activity_blocks=activity_blocks,
                               activity_encoding=activity_encoding or "static")

    results_csv = results_dir / f"loso{infix}_results.csv"
    rows = pd.read_csv(results_csv)
    # CSV has no types: "001" is read back as the integer 1, so re-pad to the
    # zero-padded string the segments use.
    rows["sid"] = rows["subject"].astype(int).map(lambda n: f"{n:03d}")
    match = rows[rows["sid"] == subject]
    if match.empty:
        raise ValueError(f"subject {subject!r} has no row in {results_csv}")
    val_subjects = match.iloc[0]["val_subjects"].split(";")

    fold_tag = f"loso{infix}_{subject}"
    ckpt = find_checkpoint(fold_tag, results_dir)
    if ckpt is None:
        raise FileNotFoundError(
            f"no checkpoint under {results_dir}/{fold_tag}/ -- run "
            f"`python src/train.py --data-dir data --mode loso` first")

    L.seed_everything(config.seed, workers=True)
    model = LitGlucoseLSTM.load_from_checkpoint(ckpt, map_location="cpu")
    model.eval()

    dm = CGMacrosDataModule(
        segments,
        history=config.history,
        horizon=config.ph // 5,
        batch_size=config.batch_size,
        split_mode="loso",
        test_subjects=[subject],
        val_subjects=val_subjects,
        multi_horizon=config.multi_horizon,
        seed=config.seed,
        **_meal_kwargs(config),
        **_activity_kwargs(config),
    )
    dm.setup()
    return _extract(model, dm)[:4]


# ---------------------------------------------------------------- run bookkeeping

def find_checkpoint(tag: str, results_dir: Path = RESULTS_DIR):
    """Lightning nests checkpoints as results/<tag>/version_0/checkpoints/*.ckpt --
    glob for it, the epoch number in the filename varies per run. None if absent."""
    return next((results_dir / tag).glob("**/*.ckpt"), None)


def run_tag(seed: int, ph: int = 30, infix: str = "") -> str:
    """train.py's auto-naming:
    ph{PH}[_multi|_bidir|_attn][_meal{ENC}_{BLOCKS}|_b3{ENC}_{BLOCKS}]_seed{N}.

    B2 infixes look like "_mealhybrid_TMYIS", B3 like "_b3static_HRFAE" -- see
    `meal_infix` / `activity_infix` for the other direction."""
    return f"ph{ph}{infix}_seed{seed}"


def meal_infix(encoding: str, blocks) -> str:
    """The B2 half of a run tag, so callers build names instead of hand-typing them."""
    return f"_meal{encoding}_{''.join(blocks)}"


def activity_infix(encoding: str, blocks) -> str:
    """The B3 half of a run tag, e.g. "_b3static_HRFAE". The name does not repeat the
    meal settings: "b3" is DEFINED as meal static TMYIS underneath, and
    parse_feature_tag turns that definition back into config when scoring."""
    return f"_b3{encoding}_{''.join(blocks)}"


def parse_meal_tag(tag: str):
    """(encoding, blocks) out of a run folder name, or (None, None) for a B1 run.

    A B2 checkpoint cannot be scored without knowing which features it was trained on, and
    the folder name is the only record that travels with the checkpoint. Parsing it back out
    is the same discipline as re-reading ph and seed from the tag: it stops a run being
    evaluated against a feature set it never saw."""
    match = re.search(r"_meal(static|channels|hybrid)_([A-Z]+)(?=_|$)", tag)
    return (match.group(1), tuple(match.group(2))) if match else (None, None)


def parse_b3_tag(tag: str):
    """(activity_encoding, activity_blocks) out of a run folder name, or (None, None).

    "hybrid" is accepted by the pattern so that if such a folder ever exists, the refusal
    happens in dataset.activity_encoding_columns with the gating message, not as a silent
    B1 mis-score here."""
    match = re.search(r"_b3(static|channels|hybrid)_([A-Z]+)(?=_|$)", tag)
    return (match.group(1), tuple(match.group(2))) if match else (None, None)


def parse_feature_tag(tag: str):
    """Every feature setting a run folder name encodes:
    (meal_encoding, meal_blocks, activity_encoding, activity_blocks).

    A b3 tag carries no explicit meal part -- "b3" is DEFINED as meal static TMYIS
    underneath (dataset.B3_MEAL_BLOCKS), and this is where that implication becomes
    config. Callers that rebuild a split from a tag or infix should use this rather
    than the two low-level parsers, so the implication can never be forgotten."""
    meal_encoding, meal_blocks = parse_meal_tag(tag)
    activity_encoding, activity_blocks = parse_b3_tag(tag)
    if activity_blocks:
        assert meal_blocks is None, \
            f"tag {tag!r} has both a _meal and a _b3 part -- train.py never writes that"
        meal_encoding, meal_blocks = B3_MEAL_ENCODING, B3_MEAL_BLOCKS
    return meal_encoding, meal_blocks, activity_encoding, activity_blocks


def pooled_config(seed: int, ph: int = 30, history: int = 12,
                  batch_size: int = 256, multi_horizon: bool = False,
                  meal_blocks=None, meal_encoding: str = "static",
                  activity_blocks=None, activity_encoding: str = "static",
                  raw_dir: Path = Path("dataset/CGMacros")) -> Namespace:
    """The frozen config as a Namespace -- these values ARE train.py's argparse
    defaults (config.yaml documents them but train.py does not read it).

    `meal_blocks=None` is B1 and the meal modules are never imported; same for
    `activity_blocks=None` and the activity modules."""
    return Namespace(seed=seed, history=history, ph=ph,
                     batch_size=batch_size, multi_horizon=multi_horizon,
                     meal_blocks=meal_blocks, meal_encoding=meal_encoding,
                     activity_blocks=activity_blocks, activity_encoding=activity_encoding,
                     raw_dir=raw_dir)


def checkpoint_params(checkpoint_path) -> int:
    """Parameter count for the architecture stored in a checkpoint's hparams."""
    model = LitGlucoseLSTM.load_from_checkpoint(checkpoint_path, map_location="cpu")
    return count_params(model.network)


def iter_pooled_seeds(segments, seeds=DEFAULT_SEEDS, ph: int = 30, infix: str = "",
                      results_dir: Path = RESULTS_DIR):
    """Yield (seed, ckpt, y, preds, last_input, subjects) for every seed whose run
    exists -- reused by all cross-seed evaluation loops (metrics, figures, ablation).

    A missing run is announced, never silently dropped: a table that quietly
    averaged 9 seeds while claiming 11 is the failure mode this guards."""
    for seed in seeds:
        tag = run_tag(seed, ph, infix)
        ckpt = find_checkpoint(tag, results_dir)
        if ckpt is None:
            print(f"  [skip] no checkpoint under {results_dir / tag}/")
            continue
        meal_encoding, meal_blocks, activity_encoding, activity_blocks = parse_feature_tag(tag)
        cfg = pooled_config(seed, ph=ph, multi_horizon="_multi" in infix,
                            meal_blocks=meal_blocks, meal_encoding=meal_encoding or "static",
                            activity_blocks=activity_blocks,
                            activity_encoding=activity_encoding or "static")
        y, preds, last_input, subjects = load_test_predictions(ckpt, segments, cfg)
        yield seed, ckpt, y, preds, last_input, subjects


def iter_pooled_seeds_with_history(segments, seeds=DEFAULT_SEEDS, ph: int = 30,
                                   infix: str = "", results_dir: Path = RESULTS_DIR):
    """iter_pooled_seeds, plus the un-scaled input window per seed."""
    for seed in seeds:
        tag = run_tag(seed, ph, infix)
        ckpt = find_checkpoint(tag, results_dir)
        if ckpt is None:
            print(f"  [skip] no checkpoint under {results_dir / tag}/")
            continue
        meal_encoding, meal_blocks, activity_encoding, activity_blocks = parse_feature_tag(tag)
        cfg = pooled_config(seed, ph=ph, multi_horizon="_multi" in infix,
                            meal_blocks=meal_blocks, meal_encoding=meal_encoding or "static",
                            activity_blocks=activity_blocks,
                            activity_encoding=activity_encoding or "static")
        yield (seed, ckpt) + load_test_predictions_with_history(ckpt, segments, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-check one run's test-set predictions")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--ph", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default=None,
                        help="run folder under results/ (default: ph{PH}_seed{SEED})")
    parser.add_argument("--loso", default=None, metavar="SUBJECT",
                        help="check a LOSO fold instead, e.g. --loso 039")
    parser.add_argument("--infix", type=str, default="",
                        help="LOSO run infix, e.g. _mealstatic_TMYIS for the B2 folds or "
                             "_b3static_HRFAE for B3 (only used with --loso)")
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/CGMacros"),
                        help="raw CGMacros CSVs, needed only to score a B2 (meal) run")
    args = parser.parse_args()

    ph, seed = args.ph, args.seed
    tag = args.tag or run_tag(seed, ph)

    # The split is decided by the seed the run was TRAINED with, so read ph and seed
    # back out of the folder name -- otherwise `--tag ph60_seed49` alone would score
    # that model on the seed-42 split, i.e. on subjects it trained on.
    if args.tag:
        match = re.match(r"ph(\d+)", tag)
        if match:
            ph = int(match.group(1))
        match = re.search(r"seed(\d+)$", tag)
        if match:
            seed = int(match.group(1))

    segments = load_segments(args.data_dir / "segments_split.npz")
    # A B2/B3 tag also carries its encoding and block list; read them back out for the
    # same reason ph and seed are read back out -- so a run is always scored on what it
    # saw (a b3 tag additionally implies meal static TMYIS, per parse_feature_tag).
    meal_encoding, meal_blocks, activity_encoding, activity_blocks = parse_feature_tag(tag)
    config = pooled_config(seed, ph=ph, history=args.history,
                           batch_size=args.batch_size, multi_horizon="_multi" in tag,
                           meal_blocks=meal_blocks, meal_encoding=meal_encoding or "static",
                           activity_blocks=activity_blocks,
                           activity_encoding=activity_encoding or "static",
                           raw_dir=args.raw_dir)

    if args.loso:
        source = f"LOSO fold {args.loso}"
        if args.infix:
            source += f" ({args.infix})"
        ckpt = find_checkpoint(f"loso{args.infix}_{args.loso}", args.results_dir)
        # The fold's feature settings come from the INFIX, not from the pooled tag --
        # scoring a B2/B3 fold with the tag-derived (B1) config is the exact mistake
        # parse_feature_tag exists to prevent.
        meal_encoding, meal_blocks, activity_encoding, activity_blocks = \
            parse_feature_tag(args.infix)
        loso_config = pooled_config(seed, history=args.history,
                                    batch_size=args.batch_size, meal_blocks=meal_blocks,
                                    meal_encoding=meal_encoding or "static",
                                    activity_blocks=activity_blocks,
                                    activity_encoding=activity_encoding or "static",
                                    raw_dir=args.raw_dir)
        y_actual, preds, last_input, subjects = load_loso_subject(
            args.loso, segments, loso_config, args.results_dir, infix=args.infix)
    else:
        source = f"seed {seed}, ph +{ph}"
        ckpt = find_checkpoint(tag, args.results_dir)
        if ckpt is None:
            print(f"no checkpoint under {args.results_dir / tag}/ -- train it first:\n"
                  f"  python src/train.py --data-dir {args.data_dir} --ph {ph} --seed {seed}")
            return
        y_actual, preds, last_input, subjects = load_test_predictions(ckpt, segments, config)

    # Multi-horizon targets are (N, H); broadcast the single last reading across columns.
    last = last_input if preds.ndim == 1 else last_input[:, None]
    rmse = float(np.sqrt(np.mean((preds - y_actual) ** 2)))
    persist = float(np.sqrt(np.mean((last - y_actual) ** 2)))

    print(f"\ncheckpoint  : {ckpt}")
    print(f"split       : {source}, {len(y_actual):,} test windows, "
          f"{len(set(subjects.tolist()))} held-out subjects")
    print(f"shapes      : y {y_actual.shape}  preds {preds.shape}  last_input {last_input.shape}")
    print(f"RMSE        : {rmse:.2f} mg/dL")
    print(f"persistence : {persist:.2f} mg/dL")
    print(f"skill       : {(persist - rmse) / persist:.1%}")
    # A range near 0 +/- 1 here means the z-scoring was never undone.
    print(f"last_input  : {last_input.min():.0f}-{last_input.max():.0f} mg/dL (un-scaled)")

    if preds.ndim == 2:
        print(f"\n  ^ multi-horizon: that RMSE is POOLED over {preds.shape[1]} horizons and is NOT\n"
              f"    comparable to a single-horizon score -- use src/per_horizon_eval.py instead.")
    if tag == "ph30_seed42" and not args.loso:
        print("\n  expected (PROJECT_STATE): RMSE 17.25 | persistence 19.11 | "
              "skill 9.7% | last_input 40-314")


if __name__ == "__main__":
    main()
