"""
Segments -> Dataset/DataModule.

split  ->  fit scaler on train  ->  window each split independently

Meal features (B2) extend that pipeline WITHOUT changing the window set: the (L, K) matrix
`meal_features.build_features` produces is parallel to `segment.glucose`, so it is windowed
by the same arithmetic and every window keeps its identity. Two slices come out of it, and
the encoding decides which are used:

    channels   feat[i : i+history, channel_cols]   -> (history, n_ch), concatenated onto glucose
    static     feat[i + history - 1, static_cols]  -> (K_s,), concatenated onto h_n

Pairing guarantee: meal features do not add or drop a single window, and `split_subjects`
depends only on the subject list, so B1 and B2 at the same seed see the byte-identical test
set in the identical order.  `python src/dataset.py` asserts it.

Meal cleaning is fitted INSIDE `setup()`, from the split's own training subjects, for the
same reason `fit_scaler` is: the caps/fills are constants learned from data, so fitting them
anywhere else risks fitting them on the wrong split.

Activity features (B3) ride the same mechanism: their (L, K) matrices are column-
concatenated onto the meal matrices (meal columns first), and the combined column indices
are routed to `channel_cols` / `static_cols` by the ACTIVITY encoding. `ActivityStats` is
the third instance of the fit-inside-setup contract (scaler, macro cleaning, now this).
A "b3" run always means meal static TMYIS underneath (`B3_MEAL_BLOCKS`) -- activity is
added on top of B2's winner, never instead of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import lightning as L

from preprocess import Segment, apply_scaler, fit_scaler, make_windows

# Encoding C ("hybrid"): which feature blocks the LSTM sees as per-timestep channels, the
# rest going to the static head.  Block S (carb_bin) is the meal-arrival event.
#
# NOTE, measured 2026-08-07 over all 123,435 windows: block S is NOT the only time-varying
# block. `decay` is non-constant across the 12 steps in 100% of non-fasting windows,
# `dt_norm` and the impulse columns in 100%, and carb_bin in only 22%. So the original
# rationale ("carbs arrive at a time, dt is a property of the present") does not survive
# measurement, even though the encoding it defines is still a sensible middle point. Read a
# hybrid result as "does the arrival event alone need sequential treatment?", NOT as
# "sequential vs static". Changing this tuple changes what `hybrid` means -- give any such
# run a distinct --tag, because the folder name does not encode it.
SEQUENTIAL_BLOCKS = ("S",)

MEAL_ENCODINGS = ("static", "channels", "hybrid")

# B3. "b3" in a folder name (ph30_b3static_HRFAE_seed42) does NOT repeat the meal settings;
# it is DEFINED as these two constants underneath. train.py enforces the pair whenever
# --activity-blocks is set, and evaluate.parse_feature_tag re-implies it when scoring, so a
# b3 checkpoint can never be rebuilt over a different meal configuration than it trained on.
B3_MEAL_BLOCKS = ("T", "M", "Y", "I", "S")
B3_MEAL_ENCODING = "static"

# Encoding "channels", B3 edition: which activity FEATURES ride as per-timestep channels.
# Only the raw per-reading signals do. Everything else stays static even in this arm:
# the trailing summaries (hr_mean30_z, hr_rel30, hr_missfrac60, cal30_z) reach back before
# the window's first reading, and the A block clips at 300 min -- a 60-min input window
# cannot represent "last active 4 h ago"; that information physically does not fit in 12
# steps. hr_slope15 stays static too: its 15-min lookback is inside the window, so the LSTM
# can recover it from the hr_z channel. Meal features stay static (B2's winner) in every
# arm. Same caveat as SEQUENTIAL_BLOCKS: changing this tuple changes what "channels" means
# -- give any such run a distinct --tag, because the folder name does not encode it.
ACTIVITY_CHANNEL_FEATURES = ("hr_z", "hr_rel", "hr_missing", "cal_z")

# "hybrid" is deliberately absent: guide 12 Part 4 gates it on an arm showing a paired
# positive first, and what it should mean would depend on which arm that is.
ACTIVITY_ENCODINGS = ("static", "channels")


def encoding_columns(blocks: tuple[str, ...], encoding: str,
                     seq_blocks: tuple[str, ...] = SEQUENTIAL_BLOCKS):
    """Split the K feature columns into (channel_cols, static_cols, names) for one encoding.

    All three encodings see the SAME K features -- only the temporal structure differs. That
    is what makes the ablation interpretable: a difference between arms cannot be blamed on
    one arm having more information.
    """
    from meal_features import block_names   # local: keeps B1 imports free of the meal code

    names = block_names(blocks)
    every = list(range(len(names)))

    if encoding == "static":
        return [], every, names
    if encoding == "channels":
        return every, [], names
    if encoding == "hybrid":
        sequential = set(block_names(tuple(b for b in blocks if b in seq_blocks)))
        channel_cols = [k for k, name in enumerate(names) if name in sequential]
        static_cols = [k for k, name in enumerate(names) if name not in sequential]
        assert channel_cols, (
            f"encoding 'hybrid' needs at least one of blocks {seq_blocks} in --meal-blocks, "
            f"got {blocks} -- otherwise it silently degenerates into 'static'")
        return channel_cols, static_cols, names
    raise ValueError(f"unknown meal encoding {encoding!r}; known: {MEAL_ENCODINGS}")


def activity_encoding_columns(blocks: tuple[str, ...], encoding: str):
    """Split the activity feature columns into (channel_cols, static_cols, names).

    Same contract as `encoding_columns`: every encoding sees the SAME features, only the
    temporal structure differs, so a difference between arms cannot be blamed on one arm
    having more information. Pure in (blocks, encoding) -- train.py sizes the model with it
    before any datamodule exists, and setup() derives the same numbers independently;
    GlucoseLSTM.forward re-checks both widths on every batch.
    """
    from activity_features import block_names   # local: keeps B1/B2 imports free of activity code

    names = block_names(tuple(blocks))
    every = list(range(len(names)))

    if encoding == "static":
        return [], every, names
    if encoding == "channels":
        dense = set(ACTIVITY_CHANNEL_FEATURES)
        channel_cols = [k for k, name in enumerate(names) if name in dense]
        static_cols = [k for k, name in enumerate(names) if name not in dense]
        assert channel_cols, (
            f"encoding 'channels' with blocks {tuple(blocks)} contains none of the dense "
            f"features {ACTIVITY_CHANNEL_FEATURES} -- it would silently degenerate into "
            f"'static'")
        return channel_cols, static_cols, names
    raise ValueError(f"unknown activity encoding {encoding!r}; known: {ACTIVITY_ENCODINGS} "
                     f"('hybrid' is gated until an arm earns it -- guide 12 Part 4)")


def concat_feature_matrices(meal_matrices: list[np.ndarray] | None,
                            activity_matrices: list[np.ndarray] | None) -> list[np.ndarray] | None:
    """Column-concatenate two per-segment feature-matrix lists; either may be None.

    Meal columns first, activity columns after -- setup() offsets the activity column
    indices by the meal width to match. The self-checks below rebuild matrices through
    this same function, so the two layouts cannot drift apart."""
    if meal_matrices is None:
        return activity_matrices
    if activity_matrices is None:
        return meal_matrices
    assert len(meal_matrices) == len(activity_matrices), \
        f"{len(meal_matrices)} meal matrices vs {len(activity_matrices)} activity matrices"
    return [np.concatenate([meal, activity], axis=1)
            for meal, activity in zip(meal_matrices, activity_matrices)]


class WindowDataset(Dataset):
    """Segments to (input, static, target) triples.

    `static` is always present -- zero-width (N, 0) tensor under B1, so the batch
    shape is uniform regardless of whether meal features are enabled."""

    def __init__(self, segments: list[Segment], history: int, horizon: int, mean: float, std: float,
                 multi_horizon: bool = False,
                 feature_matrices: list[np.ndarray] | None = None,
                 channel_cols: list[int] = (), static_cols: list[int] = ()):
        if feature_matrices is not None:
            assert len(feature_matrices) == len(segments), \
                f"{len(feature_matrices)} feature matrices for {len(segments)} segments"

        all_inputs, all_statics, all_targets, all_subject_ids = [], [], [], []

        for index, segment in enumerate(segments):
            inputs, targets = make_windows(segment, history, horizon, multi_horizon)
            inputs = apply_scaler(np.ascontiguousarray(inputs), mean, std)
            inputs = inputs[:, :, None]                     # (n, history) -> (n, history, 1)
            n_windows = len(inputs)

            if feature_matrices is not None:
                features = feature_matrices[index]
                assert len(features) == len(segment.glucose), (
                    f"feature matrix has {len(features)} rows for a {len(segment.glucose)}-"
                    "reading segment -- it must be parallel to segment.glucose")

                # Window the features with the SAME arithmetic make_windows uses, so the
                # rows stay aligned window-for-window. Window w covers readings
                # [w, w+history), i.e. its last input reading is w + history - 1.
                starts = np.arange(n_windows)
                if len(channel_cols):
                    step = starts[:, None] + np.arange(history)[None, :]      # (n, history)
                    # NOT a tile of one static row: slicing the matrix per timestep is what
                    # gives encoding B a trajectory instead of a repeated constant.
                    channels = features[:, channel_cols][step]                # (n, history, n_ch)
                    inputs = np.concatenate([inputs, channels], axis=-1)
                statics = (features[starts + history - 1][:, static_cols] if len(static_cols)
                           else np.zeros((n_windows, 0)))
            else:
                statics = np.zeros((n_windows, 0))

            all_inputs.append(inputs)
            all_statics.append(statics)
            all_targets.append(targets)
            all_subject_ids.extend([segment.subject] * n_windows)

        all_inputs = np.concatenate(all_inputs)
        all_statics = np.concatenate(all_statics)
        all_targets = np.concatenate(all_targets)

        # Meal features are NOT z-scored. They are already O(1) via fixed divisors and are
        # mostly exact zeros; z-scoring would map "no meal" onto a large negative number and
        # destroy the sparsity the network relies on. Activity features (B3) arrive already
        # scaled by frozen train stats inside activity_features.build_features -- nothing
        # here scales any feature column.
        self.inputs = torch.tensor(all_inputs, dtype=torch.float32)
        self.statics = torch.tensor(all_statics, dtype=torch.float32)
        self.targets = torch.tensor(all_targets, dtype=torch.float32)
        if self.targets.dim() == 1:
            self.targets = self.targets.unsqueeze(-1)

        self.subject_ids = all_subject_ids


    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return self.inputs[index], self.statics[index], self.targets[index]


def split_subjects(all_subjects: list[str], seed: int = 42, fractions=(0.70, 0.15, 0.15)) -> tuple[list[str], list[str], list[str]]:
    """Randomly split subject IDs into train/val/test groups. No subject appears in two groups to prevent data leakage."""

    # Use the same seed to make results deterministic across different models (for future comparison)
    rng = np.random.default_rng(seed)
    sorted_subjects = sorted(set(all_subjects))
    rng.shuffle(sorted_subjects)

    total = len(sorted_subjects)
    num_train = int(total * fractions[0])
    num_val = int(total * fractions[1])

    train_subjects = sorted_subjects[:num_train]
    val_subjects = sorted_subjects[num_train:num_train + num_val]
    test_subjects = sorted_subjects[num_train + num_val:]

    return train_subjects, val_subjects, test_subjects


class CGMacrosDataModule(L.LightningDataModule):
    """Owns the entire data pipeline: splitting, scaling, and serving batches. Lightning calls these automatically at the right time."""

    def __init__(self, segments: list[Segment], history: int = 12, horizon: int = 6,
                 batch_size: int = 256, split_mode: str = "pooled_random",
                 test_subjects: list[str] | None = None,
                 val_subjects: list[str] | None = None,
                 multi_horizon: bool = False, seed: int = 42,
                 meal_blocks: tuple[str, ...] | None = None,
                 meal_encoding: str = "static",
                 meals=None,
                 activity_blocks: tuple[str, ...] | None = None,
                 activity_encoding: str = "static",
                 activity=None):
        """`meals` is the RAW table from meals.load_meals -- uncleaned on purpose, because
        the caps and fills are fitted in setup() from this split's training subjects.
        Passing it in (rather than loading it here) means run_loso reads 44 CSVs once, not
        44 times. `meal_blocks=None` is B1: no meal code runs at all.

        `activity` is the dict of RAW minute frames from activity.load_activity -- raw for
        the same reason `meals` is: ActivityStats is fitted in setup() from this split's
        training subjects. `activity_blocks=None` is B1/B2: no activity code runs at all."""
        super().__init__()
        # `meals` is a DataFrame and `activity` a dict of them -- keep both out of hparams,
        # which get serialized into every checkpoint. Same reason `segments` is ignored.
        self.save_hyperparameters(ignore=["segments", "meals", "activity"])
        self.segments = segments
        self.meals = meals
        self.activity = activity
        self.macro_cleaning = None      # set in setup() when meal_blocks is on
        self.activity_stats = None      # set in setup() when activity_blocks is on
        self.meal_names: list[str] = []
        self.activity_names: list[str] = []
        self.channel_cols: list[int] = []
        self.static_cols: list[int] = []

    def setup(self, stage: str | None = None) -> None:
        """Prepare train/val/test datasets. Lightning calls this before training starts.

        The order to prevent data leakage:
          1. Group segments by subject
          2. Split subjects into train/val/test groups
          3. Fit the scaler on TRAINING subjects only
          4. Build WindowDataset for each split using that same scaler"""

        segments_by_subject: dict[str, list[Segment]] = {}
        for segment in self.segments:
            segments_by_subject.setdefault(segment.subject, []).append(segment)

        all_subject_ids = sorted(segments_by_subject.keys())

        if self.hparams.split_mode == "loso":
            test_subject_set = set(self.hparams.test_subjects)
            val_subject_set = set(self.hparams.val_subjects)
            train_subject_list = [s for s in all_subject_ids if s not in test_subject_set and s not in val_subject_set]
        else:  # "pooled_random"
            train_subject_list, val_subject_set, test_subject_set = split_subjects(all_subject_ids, seed=self.hparams.seed)

        # For each subject in train, get all their segments, and flatten into one list
        train_segments = [seg for subj in train_subject_list for seg in segments_by_subject[subj]]
        val_segments = [seg for subj in val_subject_set for seg in segments_by_subject[subj]]
        test_segments = [seg for subj in test_subject_set for seg in segments_by_subject[subj]]

        # For input z-score normalization
        training_mean, training_std = fit_scaler(train_segments)
        self.training_mean = training_mean
        self.training_std = training_std

        split_segment_lists = [("train", train_segments), ("val", val_segments),
                               ("test", test_segments)]

        # Meal features (B2/B3). Fitted here, from train_subject_list, for exactly the
        # reason fit_scaler is -- the caps and fills are learned constants, and fitting them
        # outside this method is how they end up describing the wrong split.
        meal_matrices = {"train": None, "val": None, "test": None}
        meal_channel_cols: list[int] = []
        meal_static_cols: list[int] = []
        if self.hparams.meal_blocks:
            from meal_features import build_all_features
            from meals import apply_macro_cleaning, fit_macro_cleaning, scale_macros

            assert self.meals is not None, \
                "meal_blocks is set but meals=None -- pass meals=load_meals(raw_dir)"
            meal_block_tuple = tuple(self.hparams.meal_blocks)

            self.macro_cleaning = fit_macro_cleaning(self.meals, train_subject_list)
            ready = scale_macros(apply_macro_cleaning(self.meals, self.macro_cleaning))

            meal_channel_cols, meal_static_cols, self.meal_names = encoding_columns(
                meal_block_tuple, self.hparams.meal_encoding)

            for name, split_segments in split_segment_lists:
                meal_matrices[name] = build_all_features(split_segments, ready, meal_block_tuple)

        # Activity features (B3). The third instance of the fit-inside-setup contract:
        # scaler, macro cleaning, now ActivityStats.
        activity_matrices = {"train": None, "val": None, "test": None}
        activity_channel_cols: list[int] = []
        activity_static_cols: list[int] = []
        if self.hparams.activity_blocks:
            from activity import aggregate_segments, fit_activity_stats
            from activity_features import build_all_features as build_all_activity_features

            assert self.activity is not None, \
                "activity_blocks is set but activity=None -- pass activity=load_activity(raw_dir)"
            activity_block_tuple = tuple(self.hparams.activity_blocks)

            activity_channel_cols, activity_static_cols, self.activity_names = \
                activity_encoding_columns(activity_block_tuple, self.hparams.activity_encoding)

            aggregates = {name: aggregate_segments(split_segments, self.activity)
                          for name, split_segments in split_segment_lists}
            self.activity_stats = fit_activity_stats(aggregates["train"])
            for name in aggregates:
                activity_matrices[name] = build_all_activity_features(
                    aggregates[name], self.activity_stats, activity_block_tuple)

        # One combined matrix per segment: meal columns first, activity columns after, so
        # the activity indices shift by the meal width. Under B1 both halves are None and
        # both column lists stay empty; under B2-only or activity-only the offset is a
        # no-op. The combined lists are what input_size / static_dim read.
        meal_width = len(self.meal_names)
        self.channel_cols = list(meal_channel_cols) + \
            [meal_width + column for column in activity_channel_cols]
        self.static_cols = list(meal_static_cols) + \
            [meal_width + column for column in activity_static_cols]
        features = {name: concat_feature_matrices(meal_matrices[name], activity_matrices[name])
                    for name in ("train", "val", "test")}

        shared_params = dict(history=self.hparams.history, horizon=self.hparams.horizon,
                             mean=training_mean, std=training_std,
                             multi_horizon=self.hparams.multi_horizon,
                             channel_cols=self.channel_cols, static_cols=self.static_cols)
        self.train_dataset = WindowDataset(train_segments, feature_matrices=features["train"], **shared_params)
        self.val_dataset = WindowDataset(val_segments, feature_matrices=features["val"], **shared_params)
        self.test_dataset = WindowDataset(test_segments, feature_matrices=features["test"], **shared_params)

    # The two numbers the model has to match. Only valid after setup(); train.py builds the
    # model first and so derives the same pair from `encoding_columns` and
    # `activity_encoding_columns` directly (both pure functions of blocks + encoding, no
    # split needed). GlucoseLSTM.forward asserts both widths on every batch, so a
    # disagreement between the two routes fails on batch 1 rather than training a quietly
    # mis-wired model.
    @property
    def input_size(self) -> int:
        """Channels per timestep: 1 (glucose) + however many feature channels (meal and/or
        activity) the encodings route per-timestep."""
        return 1 + len(self.channel_cols)

    @property
    def static_dim(self) -> int:
        """Width of the per-window static vector (meal statics then activity statics).
        0 under B1."""
        return len(self.static_cols)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0, pin_memory=True)


# ---------------------------------------------------------------------------------------
# Self-checks for the feature windowing pipeline. Run:
#     python src/dataset.py                 # B2 (three encodings) + B3 (two), seed 42
#     python src/dataset.py --seed 49 --encoding channels --skip-b3
#     python src/dataset.py --skip-meals    # B3 checks only
#
# meal_features.build_features already guarantees that row i of the (L, K) feature matrix
# uses only meals at or before reading i (per-row causality). Windowing opens a SEPARATE
# hole: which rows a window gets. An off-by-one there (static from w+history instead of
# w+history-1, or channels from [w+1, w+1+history)) passes every per-row test, trains fine,
# and quietly shows each window 5 minutes of its own future. Hence
# `check_window_alignment` and `check_window_leak` below.
# ---------------------------------------------------------------------------------------

def _windows_per_segment(segments, history: int, horizon: int) -> list[int]:
    """Window count per segment, by the same arithmetic make_windows uses."""
    return [max(0, len(seg.glucose) - history - horizon + 1) for seg in segments]


def check_window_alignment(segments, dataset, features, channel_cols, static_cols,
                           history: int, horizon: int, mean: float, std: float,
                           label: str = "") -> int:
    """For a given window at position `win`, its static row must be feature row
    win+history-1, and its channels must be feature rows [win, win+history) -- anchored on
    GLUCOSE, which the meal code never touches.

    The glucose anchor is the point: comparing the feature slices to themselves would be
    tautological. Comparing them to the reading whose glucose value ends the window ties the
    meal state to the same instant the model is forecasting from.
    """
    checked = offset = 0
    for segment, feature_matrix in zip(segments, features):
        n_windows = max(0, len(segment.glucose) - history - horizon + 1)
        for win in ({0, n_windows // 2, n_windows - 1} if n_windows else ()):
            row, last_reading = offset + win, win + history - 1
            glucose = dataset.inputs[row, -1, 0].item() * std + mean
            assert abs(glucose - segment.glucose[last_reading]) < 1e-3, (
                f"{label}: window {win} ends on glucose {glucose:.2f} but reading "
                f"{last_reading} is {segment.glucose[last_reading]:.2f}")
            if len(static_cols):
                assert np.allclose(dataset.statics[row].numpy(),
                                   feature_matrix[last_reading, static_cols], atol=1e-6), \
                    f"{label}: static vector is not feature row at win+history-1"
            if len(channel_cols):
                assert np.allclose(dataset.inputs[row, :, 1:].numpy(),
                                   feature_matrix[win:win + history][:, channel_cols], atol=1e-6), \
                    f"{label}: channels are not feat[win : win+history]"
            checked += 1
        offset += n_windows
    assert offset == len(dataset), f"{offset} windows walked vs {len(dataset)} in the dataset"
    return checked


def check_window_leak(segments, ready_meals, blocks, channel_cols, static_cols,
                      history: int, horizon: int, mean: float, std: float,
                      label: str = "") -> tuple[int, int]:
    """Delete every meal after a cutoff; every window predicting at or before it must be
    bit-identical. Returns (n_unchanged, n_changed).

    The cleaning constants are frozen by the caller, so this isolates the leakage question
    from the (separate) question of what refitting the fills would do.
    """
    from meal_features import build_all_features
    from meals import segment_times

    counts = _windows_per_segment(segments, history, horizon)
    prediction_times = np.concatenate(
        [segment_times(seg)[history - 1: history - 1 + n] for seg, n in zip(segments, counts)])

    # Stay in datetime64 rather than round-tripping through int64. `segment_times` returns
    # datetime64[**us**], so `.astype("int64")` yields MICROseconds; re-labelling those
    # integers `datetime64[ns]` reinterprets them as nanoseconds -- a 1000x error that still
    # produces a plausible-looking timestamp (1970-01-20). That emptied `safe` and made this
    # whole test pass while checking nothing. Indexing the sorted array never sees a unit.
    cutoff = np.sort(prediction_times)[len(prediction_times) // 2]

    def build(meals):
        return WindowDataset(segments, history, horizon, mean, std,
                             feature_matrices=build_all_features(segments, meals, blocks),
                             channel_cols=channel_cols, static_cols=static_cols)

    full = build(ready_meals)
    truncated = build(ready_meals[ready_meals["timestamp"] <= pd.Timestamp(cutoff)])
    assert len(prediction_times) == len(full)

    safe = prediction_times <= cutoff
    # torch.equal on two EMPTY tensors is True, so a bad cutoff makes this pass vacuously.
    assert safe.sum() > 1000, f"{label}: only {int(safe.sum())} windows before the cutoff"
    assert torch.equal(full.inputs[safe], truncated.inputs[safe]), \
        f"{label}: FUTURE LEAK -- channels changed when later meals were deleted"
    assert torch.equal(full.statics[safe], truncated.statics[safe]), \
        f"{label}: FUTURE LEAK -- static vector changed when later meals were deleted"

    later = ~safe
    assert (not torch.equal(full.inputs[later], truncated.inputs[later])
            or not torch.equal(full.statics[later], truncated.statics[later])), \
        f"{label}: deleting the later meals changed nothing -- this test cannot fail, so " \
        "it proves nothing"
    return int(safe.sum()), int(later.sum())

def check_pairing(segments, meals, seed: int = 42, ph: int = 30,
                  blocks: tuple[str, ...] = ("T", "M", "Y", "I", "S"),
                  encoding: str = "static") -> None:
    """B2 must see the byte-identical split B1 does -- same windows, same order, same glucose.

    This is the B2 analogue of compare_architectures.py's persistence-consistency assert.
    Without it a silently dropped or reordered window would still train fine and still
    produce a plausible RMSE, while quietly destroying every paired B1-vs-B2 comparison.
    """
    from model import GlucoseLSTM, count_params

    shared = dict(history=12, horizon=ph // 5, batch_size=256, split_mode="pooled_random",
                  seed=seed)
    baseline_dm = CGMacrosDataModule(segments, **shared)
    baseline_dm.setup()
    meals_dm = CGMacrosDataModule(segments, meal_blocks=blocks, meal_encoding=encoding,
                                  meals=meals, **shared)
    meals_dm.setup()

    for split in ("train", "val", "test"):
        baseline_ds = getattr(baseline_dm, f"{split}_dataset")
        meals_ds = getattr(meals_dm, f"{split}_dataset")
        assert len(baseline_ds) == len(meals_ds), \
            f"{split}: {len(baseline_ds)} B1 windows vs {len(meals_ds)} B2 windows"
        assert torch.equal(baseline_ds.targets, meals_ds.targets), f"{split}: targets differ"
        assert torch.equal(baseline_ds.inputs, meals_ds.inputs[..., 0:1]), \
            f"{split}: glucose channel differs"
        assert baseline_ds.subject_ids == meals_ds.subject_ids, f"{split}: window order differs"
    assert (baseline_dm.training_mean, baseline_dm.training_std) == \
           (meals_dm.training_mean, meals_dm.training_std), \
        "the z-scaler moved -- meal features must not touch the glucose statistics"

    test_dataset = meals_dm.test_dataset
    model = GlucoseLSTM(input_size=meals_dm.input_size, static_dim=meals_dm.static_dim)
    with torch.no_grad():                       # shapes must survive a real forward pass
        out = model(test_dataset.inputs[:8], test_dataset.statics[:8])

    print(f"  {encoding:<9} K={len(meals_dm.meal_names):<3} input_size={meals_dm.input_size:<3} "
          f"static_dim={meals_dm.static_dim:<3} params={count_params(model):>6,}  "
          f"inputs {tuple(test_dataset.inputs.shape)} statics {tuple(test_dataset.statics.shape)} "
          f"-> {tuple(out.shape)}")

    # Window-alignment and leak checks, run on the test split (the one every reported
    # number comes from). Cleaning is reused from the datamodule so the leak test isolates
    # leakage rather than mixing in the effect of refitting the fills.
    from meal_features import build_all_features
    from meals import apply_macro_cleaning, scale_macros

    by_subject: dict[str, list[Segment]] = {}
    for segment in segments:
        by_subject.setdefault(segment.subject, []).append(segment)
    _, _, test_subjects = split_subjects(sorted(by_subject), seed=seed)
    test_segments = [seg for subj in test_subjects for seg in by_subject[subj]]

    cleaned_scaled_meals = scale_macros(apply_macro_cleaning(meals, meals_dm.macro_cleaning))
    features = build_all_features(test_segments, cleaned_scaled_meals, blocks)

    aligned = check_window_alignment(
        test_segments, test_dataset, features, meals_dm.channel_cols, meals_dm.static_cols,
        history=12, horizon=ph // 5, mean=meals_dm.training_mean, std=meals_dm.training_std,
        label=encoding)
    unchanged, changed = check_window_leak(
        test_segments, cleaned_scaled_meals, blocks, meals_dm.channel_cols, meals_dm.static_cols,
        history=12, horizon=ph // 5, mean=meals_dm.training_mean, std=meals_dm.training_std,
        label=encoding)

    print(f"            PAIRING OK | ALIGNMENT OK ({aligned} windows glucose-anchored) | "
          f"NO WINDOW LEAK ({unchanged:,} bit-identical, {changed:,} correctly changed)")


def check_activity_encoding_columns() -> None:
    """The pure-function route train.py sizes the model with must slice sensibly and must
    refuse the degenerate and gated cases loudly rather than guessing."""
    from activity_features import DEFAULT_BLOCKS, block_names

    names = block_names(DEFAULT_BLOCKS)

    channel_cols, static_cols, returned_names = activity_encoding_columns(DEFAULT_BLOCKS, "static")
    assert channel_cols == [] and static_cols == list(range(len(names)))
    assert returned_names == names

    channel_cols, static_cols, _ = activity_encoding_columns(DEFAULT_BLOCKS, "channels")
    assert [names[k] for k in channel_cols] == list(ACTIVITY_CHANNEL_FEATURES), \
        f"dense channels came out as {[names[k] for k in channel_cols]}"
    assert sorted(channel_cols + static_cols) == list(range(len(names))), \
        "channels + statics must partition the feature columns"

    # The A block alone has no dense feature -> 'channels' must refuse, not silently
    # degenerate into 'static'.
    refused = False
    try:
        activity_encoding_columns(("A",), "channels")
    except AssertionError:
        refused = True
    assert refused, "channels encoding with no dense feature was not refused"

    # 'hybrid' is gated (guide 12 Part 4) -> must raise, not guess a definition.
    refused = False
    try:
        activity_encoding_columns(DEFAULT_BLOCKS, "hybrid")
    except ValueError:
        refused = True
    assert refused, "the gated hybrid activity encoding was not refused"


def check_b3_window_leak(test_segments, meal_matrices, activity, activity_stats,
                         activity_blocks, channel_cols, static_cols,
                         history: int, horizon: int, mean: float, std: float,
                         label: str = "") -> tuple[int, int]:
    """Perturb raw HR after a cutoff; every window predicting at or before it must be
    bit-identical. Returns (n_unchanged, n_changed).

    activity_features.py already proves row i of the feature MATRIX is causal. This proves
    the WINDOWING on top of it is too -- an off-by-one there (static from w+history instead
    of w+history-1) passes every per-row test and quietly shows each window 5 minutes of
    its own future. ActivityStats and the meal matrices are frozen across both builds, so
    the only thing that may respond to the perturbation is the activity feature pipeline."""
    from activity import HR_COLUMN, aggregate_segments
    from activity_features import build_all_features as build_all_activity_features
    from meals import segment_times

    counts = _windows_per_segment(test_segments, history, horizon)
    prediction_times = np.concatenate(
        [segment_times(seg)[history - 1: history - 1 + n]
         for seg, n in zip(test_segments, counts)])
    # Indexing the sorted datetime64 array directly -- see check_window_leak for the
    # unit-conversion trap this sidesteps.
    cutoff = np.sort(prediction_times)[len(prediction_times) // 2]

    def build(frames) -> WindowDataset:
        aggregates = aggregate_segments(test_segments, frames)
        activity_matrices = build_all_activity_features(aggregates, activity_stats,
                                                        tuple(activity_blocks))
        return WindowDataset(test_segments, history, horizon, mean, std,
                             feature_matrices=concat_feature_matrices(meal_matrices,
                                                                      activity_matrices),
                             channel_cols=channel_cols, static_cols=static_cols)

    # +50 bpm on every measured minute after the cutoff (NaN + 50 stays NaN, so
    # missingness patterns are untouched -- only measured HR moves).
    perturbed_frames = {}
    for subject, frame in activity.items():
        frame = frame.copy()
        after_cutoff = frame.index.to_numpy() > cutoff
        frame.loc[after_cutoff, HR_COLUMN] = frame.loc[after_cutoff, HR_COLUMN] + 50.0
        perturbed_frames[subject] = frame

    full = build(activity)
    perturbed = build(perturbed_frames)
    assert len(prediction_times) == len(full)

    safe = prediction_times <= cutoff
    # torch.equal on two EMPTY tensors is True, so a bad cutoff makes this pass vacuously.
    assert safe.sum() > 1000, f"{label}: only {int(safe.sum())} windows before the cutoff"
    assert torch.equal(full.inputs[safe], perturbed.inputs[safe]), \
        f"{label}: FUTURE LEAK -- channels changed when later HR was perturbed"
    assert torch.equal(full.statics[safe], perturbed.statics[safe]), \
        f"{label}: FUTURE LEAK -- static vector changed when later HR was perturbed"

    later = ~safe
    assert (not torch.equal(full.inputs[later], perturbed.inputs[later])
            or not torch.equal(full.statics[later], perturbed.statics[later])), \
        f"{label}: perturbing the later HR changed nothing -- this test cannot fail, so " \
        "it proves nothing"
    return int(safe.sum()), int(later.sum())


def check_b3(segments, meals, activity, seed: int = 42, ph: int = 30,
             activity_blocks: tuple[str, ...] | None = None,
             activity_encoding: str = "static") -> None:
    """B3 must see the byte-identical split B2 does -- same windows, same order, same
    targets, same glucose channel, same scaler, and B2's own static columns unchanged at
    the front of B3's statics.

    Compared against B2 rather than B1 because b3 is DEFINED on top of B2's winner (meal
    static TMYIS): if B3's extra columns moved anything B2 sees, every paired B2-vs-B3
    delta in Parts 5-6 would be contaminated."""
    from model import GlucoseLSTM, count_params
    from activity_features import DEFAULT_BLOCKS

    if activity_blocks is None:
        activity_blocks = DEFAULT_BLOCKS

    shared = dict(history=12, horizon=ph // 5, batch_size=256, split_mode="pooled_random",
                  seed=seed, meal_blocks=B3_MEAL_BLOCKS, meal_encoding=B3_MEAL_ENCODING,
                  meals=meals)
    b2_dm = CGMacrosDataModule(segments, **shared)
    b2_dm.setup()
    b3_dm = CGMacrosDataModule(segments, activity_blocks=activity_blocks,
                               activity_encoding=activity_encoding, activity=activity,
                               **shared)
    b3_dm.setup()

    n_meal = len(b3_dm.meal_names)
    for split in ("train", "val", "test"):
        b2_ds = getattr(b2_dm, f"{split}_dataset")
        b3_ds = getattr(b3_dm, f"{split}_dataset")
        assert len(b2_ds) == len(b3_ds), \
            f"{split}: {len(b2_ds)} B2 windows vs {len(b3_ds)} B3 windows"
        assert torch.equal(b2_ds.targets, b3_ds.targets), f"{split}: targets differ"
        assert torch.equal(b2_ds.inputs[..., 0:1], b3_ds.inputs[..., 0:1]), \
            f"{split}: glucose channel differs"
        assert b2_ds.subject_ids == b3_ds.subject_ids, f"{split}: window order differs"
        assert torch.equal(b2_ds.statics, b3_ds.statics[:, :n_meal]), \
            f"{split}: B2's meal statics are not intact at the front of B3's statics"
        if activity_encoding == "static":
            assert torch.equal(b2_ds.inputs, b3_ds.inputs), \
                f"{split}: static encoding must leave the input tensor untouched"
    assert (b2_dm.training_mean, b2_dm.training_std) == \
           (b3_dm.training_mean, b3_dm.training_std), \
        "the z-scaler moved -- activity features must not touch the glucose statistics"

    test_dataset = b3_dm.test_dataset
    model = GlucoseLSTM(input_size=b3_dm.input_size, static_dim=b3_dm.static_dim)
    with torch.no_grad():                       # shapes must survive a real forward pass
        out = model(test_dataset.inputs[:8], test_dataset.statics[:8])

    print(f"  b3 {activity_encoding:<9} blocks={''.join(activity_blocks):<6} "
          f"input_size={b3_dm.input_size:<3} static_dim={b3_dm.static_dim:<3} "
          f"params={count_params(model):>6,}  "
          f"inputs {tuple(test_dataset.inputs.shape)} statics {tuple(test_dataset.statics.shape)} "
          f"-> {tuple(out.shape)}")

    # Alignment and leak, on the test split, rebuilding the combined matrices from the
    # datamodule's OWN fitted objects so the checks isolate windowing from fitting.
    from meal_features import build_all_features as build_all_meal_features
    from meals import apply_macro_cleaning, scale_macros
    from activity import aggregate_segments
    from activity_features import build_all_features as build_all_activity_features

    by_subject: dict[str, list[Segment]] = {}
    for segment in segments:
        by_subject.setdefault(segment.subject, []).append(segment)
    _, _, test_subjects = split_subjects(sorted(by_subject), seed=seed)
    test_segments = [seg for subj in test_subjects for seg in by_subject[subj]]

    ready_meals = scale_macros(apply_macro_cleaning(meals, b3_dm.macro_cleaning))
    meal_matrices = build_all_meal_features(test_segments, ready_meals, B3_MEAL_BLOCKS)
    test_aggregates = aggregate_segments(test_segments, activity)
    activity_matrices = build_all_activity_features(test_aggregates, b3_dm.activity_stats,
                                                    tuple(activity_blocks))
    combined = concat_feature_matrices(meal_matrices, activity_matrices)

    aligned = check_window_alignment(
        test_segments, test_dataset, combined, b3_dm.channel_cols, b3_dm.static_cols,
        history=12, horizon=ph // 5, mean=b3_dm.training_mean, std=b3_dm.training_std,
        label=f"b3 {activity_encoding}")
    unchanged, changed = check_b3_window_leak(
        test_segments, meal_matrices, activity, b3_dm.activity_stats, activity_blocks,
        b3_dm.channel_cols, b3_dm.static_cols,
        history=12, horizon=ph // 5, mean=b3_dm.training_mean, std=b3_dm.training_std,
        label=f"b3 {activity_encoding}")

    print(f"            PAIRING OK | ALIGNMENT OK ({aligned} windows glucose-anchored) | "
          f"NO WINDOW LEAK ({unchanged:,} bit-identical, {changed:,} correctly changed)")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from preprocess import load_segments

    parser = argparse.ArgumentParser(
        description="Pairing and window self-checks: B1 vs B2 (meals), then B2 vs B3 (activity).")
    parser.add_argument("--segments", type=Path, default=Path("data/segments_split.npz"))
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/CGMacros"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ph", type=int, default=30, choices=[30, 60, 120])
    parser.add_argument("--blocks", default="T,M,Y,I,S")
    parser.add_argument("--encoding", default=None, choices=MEAL_ENCODINGS,
                        help="meal encoding to check (default: all three)")
    parser.add_argument("--activity-encoding", default=None, choices=ACTIVITY_ENCODINGS,
                        help="activity encoding to check (default: both)")
    parser.add_argument("--skip-meals", action="store_true", help="skip the B1-vs-B2 checks")
    parser.add_argument("--skip-b3", action="store_true", help="skip the B2-vs-B3 checks")
    args = parser.parse_args()

    from meals import load_meals

    segments = load_segments(args.segments)
    meals = load_meals(args.raw_dir)
    blocks = tuple(b.strip() for b in args.blocks.split(",") if b.strip())

    if not args.skip_meals:
        print(f"B1 vs B2 pairing, seed {args.seed}, ph +{args.ph}, blocks {''.join(blocks)}")
        print(f"({len(segments)} segments, {len(meals):,} meals)\n")
        for encoding in ([args.encoding] if args.encoding else MEAL_ENCODINGS):
            check_pairing(segments, meals, seed=args.seed, ph=args.ph,
                          blocks=blocks, encoding=encoding)

    if not args.skip_b3:
        from activity import load_activity
        from activity_features import DEFAULT_BLOCKS

        check_activity_encoding_columns()
        activity = load_activity(args.raw_dir)
        print(f"\nB2 vs B3 pairing, seed {args.seed}, ph +{args.ph}, "
              f"activity blocks {''.join(DEFAULT_BLOCKS)} on meal static "
              f"{''.join(B3_MEAL_BLOCKS)} (encoding-column checks passed)\n")
        for activity_encoding in ([args.activity_encoding] if args.activity_encoding
                                  else ACTIVITY_ENCODINGS):
            check_b3(segments, meals, activity, seed=args.seed, ph=args.ph,
                     activity_encoding=activity_encoding)

    print("\nSame windows, same order, same targets, same glucose channel, same scaler.")
