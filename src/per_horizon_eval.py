"""Per-horizon RMSE breakdown for a multi-horizon checkpoint.

A `--multi-horizon` run logs ONE test_rmse pooled over all horizons (+5 .. +ph). The
easy near-horizons drag that average far below a single-horizon score, so it is NOT
comparable to the +ph number -- comparing them "discovers" a huge gain that is pure
bookkeeping. This recomputes RMSE per horizon column, and compares the +ph column
against the single-horizon model at the same seed (auto-loaded if present) -- the
honest "does sharing one representation across horizons help?" number.

Reads the descriptively-named folders train.py writes:
    results/ph{PH}_multi_seed{SEED}/   (the multi-horizon run)
    results/ph{PH}_seed{SEED}/         (the single-horizon reference, optional)

Usage:
    python src/per_horizon_eval.py                    # ph30 multi-horizon, seed 42
    python src/per_horizon_eval.py --ph 30 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _predict(ckpt: Path, ph: int, seed: int, segments, multi_horizon: bool):
    """Return (preds, targets, last_input) in mg/dL on the checkpoint's test split.
    preds/targets are (N, H) for multi-horizon, (N, 1) otherwise; last is (N,)."""
    import lightning as L
    import torch
    from dataset import CGMacrosDataModule
    from train import LitGlucoseLSTM

    L.seed_everything(seed, workers=True)
    model = LitGlucoseLSTM.load_from_checkpoint(ckpt, map_location="cpu")
    model.eval()
    dm = CGMacrosDataModule(segments, history=12, horizon=ph // 5, batch_size=256,
                            split_mode="pooled_random", multi_horizon=multi_horizon, seed=seed)
    dm.setup()
    X = dm.test_dataset.inputs
    with torch.no_grad():
        preds = model(X).numpy()
    y = dm.test_dataset.targets.numpy()
    last = X[:, -1, 0].numpy() * dm.training_std + dm.training_mean
    return preds, y, last


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-horizon RMSE breakdown for a multi-horizon run")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--ph", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from preprocess import load_segments
    segments = load_segments(args.data_dir / "segments_split.npz")

    multi_dir = args.results_dir / f"ph{args.ph}_multi_seed{args.seed}"
    multi_ckpts = sorted(multi_dir.glob("**/*.ckpt"))
    if not multi_ckpts:
        print(f"no multi-horizon checkpoint in {multi_dir}/ -- run\n"
              f"  python src/train.py --data-dir data --ph {args.ph} --multi-horizon --seed {args.seed}")
        return

    preds, y, last = _predict(multi_ckpts[0], args.ph, args.seed, segments, multi_horizon=True)
    n_horizons = preds.shape[1]

    # Single-horizon reference at +ph, auto-loaded if its folder exists (same split).
    single_ckpts = sorted((args.results_dir / f"ph{args.ph}_seed{args.seed}").glob("**/*.ckpt"))
    single_rmse = None
    if single_ckpts:
        single_preds, single_targets, _ = _predict(single_ckpts[0], args.ph, args.seed, segments, multi_horizon=False)
        single_rmse = float(np.sqrt(np.mean((single_preds.squeeze() - single_targets.squeeze()) ** 2)))

    print(f"\nmulti-horizon: {multi_ckpts[0].name}  (ph{args.ph}, seed {args.seed}, {n_horizons} horizons)")
    print(f"{'horizon':>8} {'LSTM':>8} {'persist':>9} {'skill':>7}   {'vs single-horizon':>22}")
    for j in range(n_horizons):
        rmse_j = np.sqrt(np.mean((preds[:, j] - y[:, j]) ** 2))
        persist_j = np.sqrt(np.mean((last - y[:, j]) ** 2))
        skill_j = (persist_j - rmse_j) / persist_j
        tag = ""
        if (j + 1) * 5 == args.ph and single_rmse is not None:
            tag = f"single {single_rmse:.2f} (delta {rmse_j - single_rmse:+.2f})"
        print(f"{(j+1)*5:>6}m  {rmse_j:>8.2f} {persist_j:>9.2f} {skill_j:>6.1%}   {tag:>22}")

    # The pooled number the training run logged -- shown to make the trap concrete.
    pooled_rmse = np.sqrt(np.mean((preds - y) ** 2))
    pooled_persist = np.sqrt(np.mean((last[:, None] - y) ** 2))
    print(f"\npooled over all {n_horizons} horizons: LSTM {pooled_rmse:.2f}  persist {pooled_persist:.2f}  "
          f"skill {(pooled_persist - pooled_rmse) / pooled_persist:.1%}")
    print("  ^ pooled over all horizons -- not comparable to a single-horizon score.")


if __name__ == "__main__":
    main()
