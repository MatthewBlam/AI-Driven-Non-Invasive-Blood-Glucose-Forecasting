"""Lightning training loop for CGMacros glucose forecasting.

Usage:
    python src/train.py --data-dir data --mode pooled --ph 30

Meal-feature flags (no-op when omitted -- glucose-only baseline is the default):

    --meal-blocks T,M,Y,I,S            which feature blocks to build
    --meal-encoding {static,channels,hybrid}

Activity flags (B3), no-op when omitted:

    --activity-blocks H,R,F,A,E        which activity blocks (commas optional: HRFAE)
    --activity-encoding {static,channels}

Runs auto-name as results/ph30_meal{ENCODING}_{BLOCKS}_seed{N}/ (B2) or
results/ph30_b3{ENCODING}_{BLOCKS}_seed{N}/ (B3), extending the existing
_multi/_bidir/_attn convention. A b3 name does not repeat the meal settings: "b3" is
DEFINED as meal static TMYIS underneath (dataset.B3_MEAL_BLOCKS), enforced below, and
run_meta.json records both halves. Without --meal-blocks nothing in the meal pipeline runs
and the behaviour is byte-identical to the glucose-only baseline (verified: seed 42
reproduces 17.254837036132812).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import B3_MEAL_BLOCKS, B3_MEAL_ENCODING, CGMacrosDataModule
from model import GlucoseLSTM, AttentionLSTM, count_params
from preprocess import load_segments


class LitGlucoseLSTM(L.LightningModule):
    """Lightning wrapper for GlucoseLSTM -- handles training, validation, and test steps."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 1, output_dimension: int = 1,
                 bidirectional: bool = False, learning_rate: float = 1e-3, attention: bool = False,
                 input_size: int = 1, static_dim: int = 0):
        super().__init__()
        self.save_hyperparameters()
        if attention:
            assert not bidirectional, "AttentionLSTM has no bidirectional mode"
            assert input_size == 1 and static_dim == 0, \
                "AttentionLSTM supports glucose-only input -- meal features require GlucoseLSTM"
            self.network = AttentionLSTM(hidden_size, num_layers, output_dimension)
        else:
            self.network = GlucoseLSTM(hidden_size, num_layers, output_dimension, bidirectional,
                                       input_size=input_size, static_dim=static_dim)

    def forward(self, inputs, static=None):
        return self.network(inputs, static)

    def _step(self, batch, stage: str):
        """Shared forward + loss for train/val/test.

        Logs MSE, not RMSE -- Lightning averages per-batch values, and averaging
        RMSEs is not the same as RMSE of all samples."""

        glucose_windows, meal_state, actual_glucose = batch
        predicted_glucose = self(glucose_windows, meal_state)
        mse_loss = F.mse_loss(predicted_glucose, actual_glucose)  # MSE: mean((predicted - actual)^2)

        # on_epoch=True, on_step=False: log once per epoch (averaged), not every batch
        self.log(f"{stage}_mse", mse_loss, prog_bar=False, on_epoch=True, on_step=False)
        return mse_loss

    def training_step(self, batch, batch_index):
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_index):
        return self._step(batch, stage="val")

    def test_step(self, batch, batch_index):
        return self._step(batch, stage="test")

    def on_validation_epoch_end(self):
        val_mse = self.trainer.callback_metrics.get("val_mse")
        if val_mse is not None:
            self.log("val_rmse", val_mse.sqrt(), prog_bar=True)

    def on_test_epoch_end(self):
        test_mse = self.trainer.callback_metrics.get("test_mse")
        if test_mse is not None:
            self.log("test_rmse", test_mse.sqrt(), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)



# Trainer factory

def make_trainer(log_directory: Path, experiment_name: str, max_epochs: int = 100) -> L.Trainer:
    """Trainer with early stopping on val_rmse (patience 20), best-checkpoint saving,
    CSV logging, and gradient clipping."""
    return L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=1.0,
        deterministic=True,
        log_every_n_steps=25,
        logger=TensorBoardLogger(str(log_directory), name=experiment_name),
        callbacks=[
            EarlyStopping(monitor="val_rmse", mode="min", patience=20),
            ModelCheckpoint(monitor="val_rmse", mode="min", save_top_k=1),
        ],
        enable_progress_bar=True,
    )


def compute_persistence_rmse(data_module) -> tuple[float, int]:
    """Persistence baseline ("glucose stays at the last reading") on the test split.

    Returns (rmse in mg/dL, number of test windows). Call after the datamodule
    has been set up.

    Multi-horizon safe: targets are (N, horizon) when multi_horizon is on, so the
    last input reading is broadcast across every horizon column -- persistence
    predicts the same value at every horizon. Without the reshape, (N, horizon)
    minus (N,) raises a broadcast error.

    Meal-safe: inputs are (N, history, 1 + n_meal_channels), and channel 0 is always the
    z-scored glucose. Index it explicitly -- `.squeeze(-1)` was a silent no-op once the last
    axis stopped being 1, which would have made `X[:, -1]` a whole channel row and turned
    persistence into nonsense on exactly the runs it exists to be compared against."""

    X_test = data_module.test_dataset.inputs[:, :, 0].numpy()      # (N, history), glucose only
    y_test = data_module.test_dataset.targets.squeeze(-1).numpy()  # (N,) or (N, horizon)
    last_glucose = X_test[:, -1] * data_module.training_std + data_module.training_mean

    if y_test.ndim == 2:
        last_glucose = last_glucose[:, None]

    return float(np.sqrt(np.mean((y_test - last_glucose) ** 2))), len(y_test)


# Meal feature wiring. All of it is a no-op when --meal-blocks is absent.

def meal_setup(config) -> tuple[tuple[str, ...] | None, int, int, object]:
    """Turn the two meal flags into (blocks, input_size, static_dim, raw_meals).

    Returns (None, 1, 0, None) when no meal blocks are requested, so both run functions can call this unconditionally
    and the meal modules are never even imported. The dimensions come from
    `dataset.encoding_columns`, which is a pure function of blocks + encoding -- it needs
    no split, so the model can be built before the datamodule has run setup().
    GlucoseLSTM re-checks both widths on every batch, so the two routes to the same
    numbers cannot silently disagree."""
    blocks = getattr(config, "meal_blocks", None)
    if not blocks:
        return None, 1, 0, None

    from dataset import encoding_columns
    from meals import load_meals

    blocks = tuple(blocks)
    channel_cols, static_cols, names = encoding_columns(blocks, config.meal_encoding)
    meals = load_meals(config.raw_dir)
    print(f"meals: {len(meals):,} rows, blocks {''.join(blocks)} -> {len(names)} features, "
          f"encoding {config.meal_encoding} ({len(channel_cols)} channels, "
          f"{len(static_cols)} static)")
    return blocks, 1 + len(channel_cols), len(static_cols), meals


def activity_setup(config) -> tuple[tuple[str, ...] | None, int, int, object]:
    """Turn the two activity flags into (blocks, n_channels, n_static, raw_frames).

    Returns (None, 0, 0, None) when no activity blocks are requested, so B1/B2 paths never
    import the activity modules and stay byte-identical -- the same contract as meal_setup.
    The widths are ADDED to meal_setup's (input_size = glucose + meal channels + activity
    channels), and both come from pure functions of blocks + encoding, so the model can be
    built before the datamodule has run setup(). GlucoseLSTM re-checks both widths on
    every batch."""
    blocks = getattr(config, "activity_blocks", None)
    if not blocks:
        return None, 0, 0, None

    from dataset import activity_encoding_columns
    from activity import load_activity

    blocks = tuple(blocks)
    channel_cols, static_cols, names = activity_encoding_columns(blocks, config.activity_encoding)
    frames = load_activity(config.raw_dir)
    total_minutes = sum(len(frame) for frame in frames.values())
    print(f"activity: {len(frames)} subjects, {total_minutes:,} minute rows, "
          f"blocks {''.join(blocks)} -> {len(names)} features, "
          f"encoding {config.activity_encoding} ({len(channel_cols)} channels, "
          f"{len(static_cols)} static)")
    return blocks, len(channel_cols), len(static_cols), frames


def run_infix(config) -> str:
    """The feature part of a run/folder name: "" (B1), "_meal{ENC}_{BLOCKS}" (B2), or
    "_b3{ENC}_{BLOCKS}" (B3). A b3 name does NOT repeat the meal settings -- "b3" is
    defined as meal static TMYIS underneath, which main() enforces before any run starts.
    evaluate.py builds and parses the same strings (run_tag / parse_feature_tag)."""
    if getattr(config, "activity_blocks", None):
        return f"_b3{config.activity_encoding}_{''.join(config.activity_blocks)}"
    if config.meal_blocks:
        return f"_meal{config.meal_encoding}_{''.join(config.meal_blocks)}"
    return ""


# Run a pooled experiment

def run_pooled(segments, config, experiment_name: str = "pooled") -> dict:
    """Single train/val/test split. Fast for debugging the pipeline.

    experiment_name: TensorBoard log folder under results/. Parallel callers
    (e.g. sweep shards) must pass distinct names -- two processes sharing one
    folder can race for the same version_N and overwrite checkpoints."""

    L.seed_everything(config.seed, workers=True)

    meal_blocks, input_size, static_dim, meals = meal_setup(config)
    activity_blocks, n_activity_channels, n_activity_static, activity_frames = activity_setup(config)
    input_size = input_size + n_activity_channels
    static_dim = static_dim + n_activity_static

    output_dimension = config.ph // 5 if config.multi_horizon else 1
    model = LitGlucoseLSTM(
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_dimension=output_dimension,
        bidirectional=config.bidirectional,
        learning_rate=config.lr,
        attention=getattr(config, "attention", False),
        input_size=input_size,
        static_dim=static_dim,
    )
    print(f"params: {count_params(model.network)}")

    data_module = CGMacrosDataModule(
        segments,
        history=config.history,
        horizon=config.ph // 5,
        batch_size=config.batch_size,
        split_mode="pooled_random",
        multi_horizon=config.multi_horizon,
        seed=config.seed,
        meal_blocks=meal_blocks,
        meal_encoding=getattr(config, "meal_encoding", "static"),
        meals=meals,
        activity_blocks=activity_blocks,
        activity_encoding=getattr(config, "activity_encoding", "static"),
        activity=activity_frames,
    )

    trainer = make_trainer(Path("results"), experiment_name, max_epochs=100)

    start_time = time.perf_counter()
    trainer.fit(model, datamodule=data_module)  # Run entire training loop
    training_wall_clock = time.perf_counter() - start_time

    # Compute baselines on the exact same test windows the LSTM will be tested on
    persistence_rmse, n_test_windows = compute_persistence_rmse(data_module)

    test_results = trainer.test(ckpt_path="best", datamodule=data_module)[0]
    # Best val_rmse seen during training (what ModelCheckpoint monitored) --
    # trainer.test only returns test metrics, but the sweep selects on validation
    test_results["val_rmse"] = float(trainer.checkpoint_callback.best_model_score)
    test_results["wall_seconds"] = round(training_wall_clock, 2)
    test_results["epochs"] = trainer.current_epoch
    test_results["params"] = count_params(model.network)
    test_results["persistence_rmse"] = persistence_rmse
    test_results["test_windows"] = n_test_windows
    # Makes a feature run folder self-describing -- the ablation table is read months
    # later, and "which blocks was this?" must not depend on remembering the command line.
    # For a b3 run the meal keys record the implied meal static TMYIS explicitly, so the
    # b3 definition travels with the checkpoint rather than living only in dataset.py.
    test_results["meal_blocks"] = "".join(meal_blocks) if meal_blocks else None
    test_results["meal_encoding"] = getattr(config, "meal_encoding", None) if meal_blocks else None
    test_results["activity_blocks"] = "".join(activity_blocks) if activity_blocks else None
    test_results["activity_encoding"] = getattr(config, "activity_encoding", None) if activity_blocks else None

    # Persist the run's metadata beside its checkpoint. wall_seconds is otherwise only
    # printed and then lost -- recovering it later means parsing wall_time stamps out of
    # the tfevents file, which misses setup time and cannot be trusted per-run anyway.
    # Written per version_N so reruns never overwrite each other's numbers.
    meta_path = Path(trainer.logger.log_dir) / "run_meta.json"
    meta_path.write_text(json.dumps(test_results, indent=2, default=float), encoding="utf-8")

    return test_results


def run_loso(segments, config) -> list[dict]:
    """Leave-One-Subject-Out: 44 folds. Fresh model + trainer per fold."""

    all_subjects = sorted({seg.subject for seg in segments})
    fold_results = []

    # Read the meal CSVs and activity frames ONCE, not once per fold. Each fold still fits
    # its own caps, fills, and ActivityStats, inside its own datamodule.setup(), from that
    # fold's training subjects.
    meal_blocks, input_size, static_dim, meals = meal_setup(config)
    activity_blocks, n_activity_channels, n_activity_static, activity_frames = activity_setup(config)
    input_size = input_size + n_activity_channels
    static_dim = static_dim + n_activity_static
    infix = run_infix(config)

    for i, test_subject in enumerate(all_subjects):
        print(f"\n{'='*50}")
        print(f"LOSO fold {i+1}/{len(all_subjects)}: testing on {test_subject}")
        print(f"{'='*50}")

        L.seed_everything(config.seed, workers=True)

        # Hold out ~4 subjects for validation (from the remaining 43)
        remaining = [s for s in all_subjects if s != test_subject]
        rng = np.random.default_rng(config.seed)
        rng.shuffle(remaining)
        n_val = max(1, int(len(remaining) * 0.1))   # ~4 subjects
        val_subjects = remaining[:n_val]

        output_dimension = config.ph // 5 if config.multi_horizon else 1
        model = LitGlucoseLSTM(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_dimension=output_dimension,
            bidirectional=config.bidirectional,
            learning_rate=config.lr,
            attention=getattr(config, "attention", False),
            input_size=input_size,
            static_dim=static_dim,
        )

        data_module = CGMacrosDataModule(
            segments,
            history=config.history,
            horizon=config.ph // 5,
            batch_size=config.batch_size,
            split_mode="loso",
            test_subjects=[test_subject],
            val_subjects=val_subjects,
            multi_horizon=config.multi_horizon,
            seed=config.seed,
            meal_blocks=meal_blocks,
            meal_encoding=getattr(config, "meal_encoding", "static"),
            meals=meals,
            activity_blocks=activity_blocks,
            activity_encoding=getattr(config, "activity_encoding", "static"),
            activity=activity_frames,
        )

        folder = f"loso{infix}_{test_subject}"
        trainer = make_trainer(
            Path("results"), folder, max_epochs=100
        )

        trainer.fit(model, datamodule=data_module)

        # Persistence baseline for this fold. Reuse the same helper run_pooled
        # uses -- it handles the multi-horizon target shape, and sharing it means
        # the two code paths can't drift apart. trainer.fit already called
        # data_module.setup(), so the datasets exist.
        persistence_rmse, n_test_windows = compute_persistence_rmse(data_module)

        test_result = trainer.test(ckpt_path="best", datamodule=data_module)[0]
        test_rmse = test_result.get("test_rmse", 0)
        skill = (persistence_rmse - test_rmse) / persistence_rmse

        fold_results.append({
            "subject": test_subject,
            "test_rmse": test_rmse,
            "persistence_rmse": persistence_rmse,
            "skill": skill,                          # saved so the CSV is self-contained
            "test_windows": n_test_windows,
            "epochs": trainer.current_epoch,
            "val_subjects": ";".join(val_subjects),  # exact fold makeup, for reproducibility
        })

        print(f"  RMSE: {test_rmse:.2f}  persistence: {persistence_rmse:.2f}  skill: {skill:.1%}")

    return fold_results


def diagnose_prediction_lag(checkpoint_path: str, segments, config) -> dict:
    """Check whether the model is genuinely forecasting or just echoing the last input.

    Compares predictions against two references:
      - last_input (unscaled): what persistence would predict
      - actual_target: the true future glucose value

    If the model is just echoing, predictions ~= last_input and RMSE ~= persistence.
    If it's genuinely forecasting, predictions deviate from last_input toward actual_target.
    """
    L.seed_everything(config.seed, workers=True)

    meal_blocks, _, _, meals = meal_setup(config)
    activity_blocks, _, _, activity_frames = activity_setup(config)

    output_dimension = config.ph // 5 if config.multi_horizon else 1
    model = LitGlucoseLSTM.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()

    data_module = CGMacrosDataModule(
        segments,
        history=config.history,
        horizon=config.ph // 5,
        batch_size=config.batch_size,
        split_mode="pooled_random",
        multi_horizon=config.multi_horizon,
        seed=config.seed,
        meal_blocks=meal_blocks,
        meal_encoding=getattr(config, "meal_encoding", "static"),
        meals=meals,
        activity_blocks=activity_blocks,
        activity_encoding=getattr(config, "activity_encoding", "static"),
        activity=activity_frames,
    )
    data_module.setup()

    X_test = data_module.test_dataset.inputs     # (N, 12, 1+n_ch), z-scored glucose in ch 0
    static_test = data_module.test_dataset.statics   # (N, static_dim); (N, 0) without meals
    y_test = data_module.test_dataset.targets    # (N, 1), raw mg/dL

    with torch.no_grad():
        preds = model(X_test, static_test).numpy().squeeze()  # (N,), raw mg/dL

    y_actual = y_test.numpy().squeeze()          # (N,), raw mg/dL
    mean, std = data_module.training_mean, data_module.training_std
    last_input = (X_test[:, -1, 0].numpy() * std + mean)  # (N,), unscaled mg/dL

    r_pred_actual = float(np.corrcoef(preds, y_actual)[0, 1])
    r_pred_last   = float(np.corrcoef(preds, last_input)[0, 1])
    r_actual_last = float(np.corrcoef(y_actual, last_input)[0, 1])

    rmse_model       = float(np.sqrt(np.mean((preds - y_actual) ** 2)))
    rmse_persistence = float(np.sqrt(np.mean((last_input - y_actual) ** 2)))
    rmse_pred_vs_last = float(np.sqrt(np.mean((preds - last_input) ** 2)))

    adjustment = preds - last_input
    mean_adj = float(np.mean(adjustment))
    std_adj  = float(np.std(adjustment))
    mae_adj  = float(np.mean(np.abs(adjustment)))

    results = {
        "r_pred_actual": r_pred_actual,
        "r_pred_last": r_pred_last,
        "r_actual_last": r_actual_last,
        "rmse_model": rmse_model,
        "rmse_persistence": rmse_persistence,
        "rmse_pred_vs_last": rmse_pred_vs_last,
        "mean_adjustment": mean_adj,
        "std_adjustment": std_adj,
        "mae_adjustment": mae_adj,
        "n_test": len(preds),
    }

    print("\n" + "=" * 55)
    print("  PREDICTION LAG DIAGNOSTIC")
    print("=" * 55)

    print("\nCorrelations:")
    print(f"  r(pred, actual_target)   = {r_pred_actual:.4f}")
    print(f"  r(pred, last_input)      = {r_pred_last:.4f}")
    print(f"  r(actual, last_input)    = {r_actual_last:.4f}  <- glucose autocorrelation")

    print("\nRMSEs:")
    print(f"  RMSE(pred, actual)       = {rmse_model:.2f} mg/dL  <- model")
    print(f"  RMSE(last_input, actual) = {rmse_persistence:.2f} mg/dL  <- persistence")
    print(f"  RMSE(pred, last_input)   = {rmse_pred_vs_last:.2f} mg/dL  <- deviation from echoing")

    print("\nAdjustment from last input (pred - last_input):")
    print(f"  mean  = {mean_adj:+.2f} mg/dL")
    print(f"  std   = {std_adj:.2f} mg/dL")
    print(f"  MAE   = {mae_adj:.2f} mg/dL")

    print("\nVerdict: ", end="")
    if rmse_pred_vs_last < 2.0 and rmse_model >= rmse_persistence * 0.98:
        print("MODEL IS ECHOING the last input. Predictions ~= persistence.")
    elif r_pred_actual > r_actual_last and rmse_model < rmse_persistence:
        skill = (rmse_persistence - rmse_model) / rmse_persistence * 100
        print(f"Model is FORECASTING. Tracks future glucose better than autocorrelation alone.")
        print(f"         Skill vs persistence: {skill:.1f}%  (adjustments avg {mae_adj:.1f} mg/dL from last input)")
    else:
        print("BORDERLINE -- model deviates from last input but doesn't clearly beat persistence.")

    print(f"\n  (n = {len(preds):,} test windows)")
    print("=" * 55)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/CGMacros"))
    parser.add_argument("--mode", choices=["pooled", "loso"], default="pooled")
    parser.add_argument("--ph", type=int, default=30, choices=[30, 60, 120])
    parser.add_argument("--multi-horizon", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnose", type=str, default=None, metavar="CKPT", help="Path to checkpoint; runs prediction lag diagnostic instead of training")
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--tag", type=str, default=None, help="checkpoint folder name under results/ (default: auto from config)")
    parser.add_argument("--attention", action="store_true")
    # Meal feature flags. --data-dir points at the segment cache; the meal CSVs live in
    # the raw dataset tree, so they need their own flag (same split cohort_table.py has).
    parser.add_argument("--meal-blocks", type=str, default=None, metavar="T,M,Y,I,S",
                        help="comma-separated meal feature blocks (default: none = glucose-only)")
    parser.add_argument("--meal-encoding", choices=["static", "channels", "hybrid"],
                        default="static", help="how meal features reach the model")
    # Activity feature flags (B3). "hybrid" is not offered: guide 12 Part 4 gates it on an
    # arm showing a paired positive first, and dataset.activity_encoding_columns refuses it.
    parser.add_argument("--activity-blocks", type=str, default=None, metavar="H,R,F,A,E",
                        help="activity feature blocks, commas optional (default: none). "
                             "Implies meal static T,M,Y,I,S underneath -- the b3 definition.")
    parser.add_argument("--activity-encoding", choices=["static", "channels"],
                        default="static", help="how activity features reach the model")
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/CGMacros"),
                        help="raw CGMacros CSVs, for the meal logs and activity minutes "
                             "(only needed with --meal-blocks / --activity-blocks)")
    config = parser.parse_args()

    config.meal_blocks = tuple(
        block.strip().upper() for block in config.meal_blocks.split(",") if block.strip()
    ) if config.meal_blocks else None
    config.activity_blocks = tuple(
        character for character in config.activity_blocks.upper() if character not in ", "
    ) if config.activity_blocks else None

    # "b3" is DEFINED as activity on top of B2's winner (meal static TMYIS). The folder
    # name does not repeat the meal settings, so a nonstandard combination is refused
    # rather than recorded under a name that lies about it.
    if config.activity_blocks:
        if config.meal_blocks is None:
            config.meal_blocks = B3_MEAL_BLOCKS
            config.meal_encoding = B3_MEAL_ENCODING
            print(f"--activity-blocks implies meal {B3_MEAL_ENCODING} "
                  f"{''.join(B3_MEAL_BLOCKS)} underneath (the b3 definition); enabling it")
        elif (config.meal_blocks != B3_MEAL_BLOCKS
              or config.meal_encoding != B3_MEAL_ENCODING):
            parser.error(
                f"--activity-blocks requires meal {B3_MEAL_ENCODING} "
                f"{''.join(B3_MEAL_BLOCKS)} (the b3 definition); got --meal-blocks "
                f"{''.join(config.meal_blocks)} --meal-encoding {config.meal_encoding}")

    torch.set_float32_matmul_precision("high")

    segments = load_segments(config.data_dir / "segments_split.npz")

    if config.diagnose:
        diagnose_prediction_lag(config.diagnose, segments, config)
        return

    if config.mode == "loso":
        fold_results = run_loso(segments, config)

        model_rmses = np.array([r["test_rmse"] for r in fold_results])
        persist_rmses = np.array([r["persistence_rmse"] for r in fold_results])
        skills = (persist_rmses - model_rmses) / persist_rmses

        print(f"\n{'=' * 55}")
        print(f"LOSO SUMMARY ({len(fold_results)} folds, macro-averaged)")
        print(f"{'=' * 55}")
        print(f"Model RMSE:       {model_rmses.mean():.2f} +/- {model_rmses.std(ddof=1):.2f} mg/dL")
        print(f"Persistence RMSE: {persist_rmses.mean():.2f} +/- {persist_rmses.std(ddof=1):.2f} mg/dL")
        print(f"Skill:            {skills.mean():.1%} +/- {skills.std(ddof=1):.1%}")

        import csv
        infix = run_infix(config)
        out_path = Path(f"results/loso{infix}_results.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fold_results[0].keys())
            writer.writeheader()
            writer.writerows(fold_results)  # <-- all 44 rows
        print(f"\nPer-subject results saved to {out_path}")

        # Also persist the headline number so it isn't only in the terminal.
        summary_path = Path(f"results/loso{infix}_summary.txt")
        summary_path.write_text(
            f"LOSO ({len(fold_results)} folds, macro-averaged)\n"
            f"Model RMSE:       {model_rmses.mean():.2f} +/- {model_rmses.std(ddof=1):.2f} mg/dL\n"
            f"Persistence RMSE: {persist_rmses.mean():.2f} +/- {persist_rmses.std(ddof=1):.2f} mg/dL\n"
            f"Skill:            {skills.mean():.1%} +/- {skills.std(ddof=1):.1%}\n"
        )
        print(f"Summary saved to {summary_path}")
        return

    # Descriptive checkpoint folder -> results/<name>/version_N/. LOSO already
    # names folders per subject; this gives pooled runs the same self-documenting
    # names instead of piling every run into one opaque version_N sequence.
    #
    # The feature part comes from run_infix -- "_mealstatic_TMYIS" (B2) or
    # "_b3static_HRFAE" (B3, meal part implied) -- so folder naming has exactly one
    # author in this file. evaluate.run_tag() builds the same strings, and
    # aggregate_horizons.py's regex r"^ph(\d+)_seed(\d+)$" is anchored at both ends,
    # so feature runs cannot pollute the glucose-only horizon table.
    base_parts = (
        [f"ph{config.ph}"]
        + (["multi"] if config.multi_horizon else [])
        + (["bidir"] if config.bidirectional else [])
        + (["attn"] if getattr(config, "attention", False) else [])
    )
    run_name = config.tag or ("_".join(base_parts) + run_infix(config) + f"_seed{config.seed}")
    test_results = run_pooled(segments, config, experiment_name=run_name)
    print(f"checkpoints: results/{run_name}/")

    model_rmse = test_results.get("test_rmse", 0)
    persistence_rmse = test_results["persistence_rmse"]
    skill = (persistence_rmse - model_rmse) / persistence_rmse
    print(f"\ntest RMSE: {model_rmse:.2f} mg/dL")
    print(f"persistence RMSE (same split): {persistence_rmse:.2f} mg/dL")
    print(f"skill vs persistence: {skill:.3f}")
    print(f"test windows: {test_results['test_windows']:,}")
    print(f"wall clock: {test_results['wall_seconds']}s, epochs: {test_results['epochs']}, params: {test_results['params']}")


if __name__ == "__main__":
    main()
