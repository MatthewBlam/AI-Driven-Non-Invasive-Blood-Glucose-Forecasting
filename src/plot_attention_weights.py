"""Plot where the Attention-LSTM looks, averaged over seeds.

Loads every ph{PH}_attn_seed{N} checkpoint, extracts attention weights on that
seed's own test split, and plots the mean +/- std across seeds. Multi-seed because
a single seed's attention shape could be a lucky draw (seed 42 overstated skill
at +60/+120 in earlier runs); 11 models trained on 11 different splits agreeing
is what makes the finding credible.

Usage (from the project root):
    python src/plot_attention_weights.py
    python src/plot_attention_weights.py --seeds 42 43 44 --out results/attn3.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset import CGMacrosDataModule
from preprocess import load_segments
from train import LitGlucoseLSTM


def seed_attention_weights(segments, seed: int, ph: int, history: int) -> np.ndarray:
    """Average attention weight per input timestep for one seed's model."""

    # Lightning nests checkpoints as results/<run>/version_0/checkpoints/<epoch=..>.ckpt
    # -- glob for it, the epoch number varies per run
    folder = Path(f"results/ph{ph}_attn_seed{seed}")
    ckpt = next(folder.glob("**/*.ckpt"), None)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint under {folder}/ -- did seed {seed} finish training?")

    # attention=True is stored in hparams, so load_from_checkpoint rebuilds
    # the AttentionLSTM architecture from the checkpoint
    model = LitGlucoseLSTM.load_from_checkpoint(ckpt, map_location="cpu")
    model.eval()
    if not hasattr(model.network, "get_attention_weights"):
        raise TypeError(f"{ckpt} is not an attention model -- retrain seed {seed} with --attention")

    # seed must match the checkpoint's -- it determines the split, so a mismatch
    # would extract weights on windows this model trained on.
    data_module = CGMacrosDataModule(
        segments, history=history, horizon=ph // 5,
        split_mode="pooled_random", seed=seed,
    )
    data_module.setup()          # must call this before touching test_dataset

    weights = model.network.get_attention_weights(data_module.test_dataset.inputs)
    return weights.mean(dim=0).numpy()               # (history,)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Attention-LSTM weights across seeds")
    parser.add_argument("--segments", type=Path, default=Path("data/segments_split.npz"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 53)))
    parser.add_argument("--ph", type=int, default=30)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("results/attention_weights.png"))
    parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    args = parser.parse_args()

    # Load ONCE -- load_segments re-reads the .npz on every call, and doing that
    # inside the loop would re-parse the same file once per seed for nothing.
    segments = load_segments(args.segments)

    per_seed = []
    for seed in args.seeds:
        per_seed.append(seed_attention_weights(segments, seed, args.ph, args.history))
        print(f"  seed {seed}: done")

    per_seed = np.stack(per_seed)                    # (n_seeds, history)
    mean_weights = per_seed.mean(axis=0)
    std_weights = per_seed.std(axis=0, ddof=1)       # sample std, matches the rest of the project

    timesteps = np.arange(1, args.history + 1)
    uniform = 1 / args.history

    plt.figure(figsize=(8, 4))
    plt.bar(timesteps, mean_weights, color="tab:blue", alpha=0.45,
            label=f"mean over {len(per_seed)} seeds")

    # Per-seed dots instead of error bars -- the distribution is bimodal (most seeds
    # cluster low, a couple collapse to ~1.0), so +/- std would be misleading.
    # Deterministic jitter so the figure is reproducible.
    jitter = np.linspace(-0.22, 0.22, len(per_seed))
    for offset, seed_weights in zip(jitter, per_seed):
        plt.scatter(timesteps + offset, seed_weights, s=9, color="tab:blue",
                    edgecolors="white", linewidths=0.3, zorder=3)
    plt.scatter([], [], s=9, color="tab:blue", edgecolors="white", linewidths=0.3,
                label=f"individual seeds (n={len(per_seed)})")

    plt.axhline(uniform, ls="--", c="gray", label=f"uniform ({uniform:.3f})")
    plt.xlabel(f"Input timestep (1 = oldest, {args.history} = most recent)")
    plt.ylabel("Average attention weight")
    plt.title("Attention-LSTM: Where does the model look?")
    plt.xticks(list(timesteps))
    plt.legend()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nsaved {args.out}")

    # Caption numbers.
    print(f"\n{'step':>5} {'mean':>8} {'std':>8} {'x uniform':>10}")
    for t, m, s in zip(timesteps, mean_weights, std_weights):
        print(f"{t:>5} {m:>8.4f} {s:>8.4f} {m / uniform:>9.2f}x")
    print(f"\nweight on last 2 steps: {mean_weights[-2:].sum():.1%}   "
          f"first 6 steps: {mean_weights[:6].sum():.1%}")

    # Per-seed consistency. The headline claim is about DIRECTION (does every seed
    # favor the most recent reading?), which the mean alone can't tell you.
    last_step_weights = per_seed[:, -1]
    n_argmax_last = int((per_seed.argmax(axis=1) == args.history - 1).sum())
    print(f"\nargmax == timestep {args.history} in {n_argmax_last}/{len(per_seed)} seeds"
          f"   (last-2 > first-6 in "
          f"{int((per_seed[:, -2:].sum(1) > per_seed[:, :6].sum(1)).sum())}/{len(per_seed)})")
    print(f"weight on timestep {args.history} ranges "
          f"{last_step_weights.min():.3f} to {last_step_weights.max():.3f}")

    # A seed that puts ~everything on the last reading has degenerated into exact
    # persistence. Worth naming: it makes the spread bimodal, so mean +/- std
    # misdescribes it (hence the per-seed dots on the plot).
    collapsed = [s for s, weight in zip(args.seeds, last_step_weights) if weight > 0.9]
    if collapsed:
        print(f"COLLAPSED to ~pure persistence (w[{args.history}] > 0.9): "
              f"seeds {collapsed}  -- {len(collapsed)}/{len(per_seed)}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
