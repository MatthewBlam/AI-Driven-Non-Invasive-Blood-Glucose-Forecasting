"""
GlucoseLSTM: takes 12 glucose readings, predicts future glucose.

Input:  (batch, 12, 1)  -- batch of windows, 12 readings each, 1 glucose value per reading
Output: (batch, 1)      -- one prediction per window (or more if output_dim > 1)

Two optional arguments extend the architecture for meal-feature encodings, both
defaulting to glucose-only behaviour so every existing checkpoint still loads and
still reports 17,217 parameters:

    input_size   1 + n_channels   meal features fed per timestep, alongside glucose
    static_dim   width of the "state now" vector concatenated to final hidden state before the head

The three encodings are just three settings of that pair:

    A static    input_size=1,    static_dim=21   +21 params    LSTM path identical to glucose-only
    B channels  input_size=22,   static_dim=0    +5,376        every feature a trajectory
    C hybrid    input_size=2,    static_dim=20   +276          arrival event sequential
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GlucoseLSTM(nn.Module):
    """Takes 12 glucose readings, predicts future glucose."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 1, output_dimension: int = 1,
                 bidirectional: bool = False, dropout: float = 0.0,
                 input_size: int = 1, static_dim: int = 0):
        """
        :param hidden_size: LSTM size
        :param num_layers: LSTM layers
        :param output_dimension: Number of predictions (1 -> t+30 ; 6 -> t+5, t+10, ..., t+30)
        :param bidirectional: Bool toggle
        :param dropout: Only applies if num_layers > 1
        :param input_size: Channels per timestep. 1 = glucose only; >1 adds meal-feature channels
        :param static_dim: Width of the per-window static vector concatenated to final hidden state (0 = none)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.static_dim = static_dim

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        num_directions = 2 if bidirectional else 1
        self.head = nn.Linear(num_directions * hidden_size + static_dim, output_dimension)

    def forward(self, x: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        """Takes a batch of glucose windows (+ optional static meal state) and returns predictions."""

        # Ensure correct tensor shape. Checked against the LSTM's own input_size rather than a
        # hardcoded 1, so glucose-only (1) and every meal encoding (2 or 22) are all guarded.
        assert x.dim() == 3 and x.size(-1) == self.lstm.input_size, \
            f"expected (B, H, {self.lstm.input_size}), got {tuple(x.shape)}"

        # all_timestep_outputs: hidden state at every timestep (used by attention variant)
        # final_hidden: hidden state at last timestep only (sequence summary)
        # final_cell: cell state at last timestep (not used here)
        all_timestep_outputs, (final_hidden, final_cell) = self.lstm(x)

        if self.bidirectional:
            summary = torch.cat([final_hidden[-2], final_hidden[-1]], dim=-1)  # (batch, 2*hidden)
        else:
            summary = final_hidden[-1]

        if self.static_dim:
            assert static is not None, "model built with static_dim > 0 but got static=None"
            assert static.size(-1) == self.static_dim, \
                f"expected static (B, {self.static_dim}), got {tuple(static.shape)}"
            summary = torch.cat([summary, static], dim=-1)  # (batch, hidden + static_dim)

        return self.head(summary)


class AttentionLSTM(nn.Module):
    """LSTM with learned attention over timestep outputs"""

    def __init__(self, hidden_size: int = 64, num_layers: int = 1, output_dimension: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention: learns a score for each timestep
        self.attention = nn.Linear(hidden_size, 1)
        self.head = nn.Linear(hidden_size, output_dimension)

    def forward(self, x: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        assert x.dim() == 3 and x.size(-1) == 1, f"expected (B, T, 1), got {tuple(x.shape)}"
        # Design decision: AttentionLSTM accepts `static` only so the training harness
        # (LitGlucoseLSTM) can call every network with a uniform signature.
        # Meal-feature encodings are not wired into this architecture -- the architecture
        # comparison already concluded with a three-way tie, so re-opening it here would
        # double the run count to re-answer a settled question.  Fail loudly, not silently.
        assert static is None or static.numel() == 0, \
            "AttentionLSTM has no static head -- use GlucoseLSTM for meal encodings"

        timestep_outputs, _ = self.lstm(x)  # (batch, 12, hidden)

        # Attention scores: one per timestep
        scores = self.attention(timestep_outputs).squeeze(-1)  # (batch, 12)
        weights = torch.softmax(scores, dim=1)  # (batch, 12), sums to 1

        # Weighted sum of hidden states
        context = torch.bmm(                      # (batch, 1, hidden)
            weights.unsqueeze(1),                 # (batch, 1, 12)
            timestep_outputs,                     # (batch, 12, hidden)
        ).squeeze(1)                              # (batch, hidden)

        return self.head(context)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention weights for visualization"""
        with torch.no_grad():
            timestep_outputs, _ = self.lstm(x)
            scores = self.attention(timestep_outputs).squeeze(-1)
            return torch.softmax(scores, dim=1)


def count_params(model: nn.Module) -> int:
    """Count trainable parameters. Report this next to RMSE."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick shape test, run with: python src/model.py
    # If the shapes are wrong, nothing later works
    model = GlucoseLSTM(hidden_size=64, output_dimension=1)
    print("params:", count_params(model))  # expect 17217

    fake_batch = torch.randn(8, 12, 1)  # 8 fake windows
    print("output shape:", model(fake_batch).shape)  # expect (8, 1)

    # Attention
    attn = AttentionLSTM(hidden_size=64, output_dimension=1)
    print("attn params:", count_params(attn))  # expect 17282
    print("attn output shape:", attn(fake_batch).shape)  # expect (8, 1)

    attention_weights = attn.get_attention_weights(fake_batch)
    print("weights shape:", attention_weights.shape)  # expect (8, 12)
    print("weights sum to 1:", attention_weights.sum(dim=1))  # expect all 1.0

    # Meal-feature encodings: verify parameter counts before running -- a wrong
    # parameter count is the cheapest way to catch a mis-wired encoding.
    print("\nB2 encodings (K = 21 meal features, 1 of them block S):")
    for label, input_channels, static_width in [("A static  ", 1, 21), ("B channels", 22, 0), ("C hybrid  ", 2, 20)]:
        network = GlucoseLSTM(hidden_size=64, output_dimension=1,
                              input_size=input_channels, static_dim=static_width)
        batch = torch.randn(8, 12, input_channels)
        static_features = torch.randn(8, static_width) if static_width else None
        prediction = network(batch, static_features)
        print(f"  {label} input_size={input_channels:>2} static_dim={static_width:>2} -> "
              f"params {count_params(network):>6,}  out {tuple(prediction.shape)}")
    # expect 17,238 / 22,593 / 17,493

    # Param arithmetic:
    # LSTM: 4 gates * (1*64 + 64*64 + 2*64) = 17,152; head: 64+1 = 65; total: 17,217
    # Each extra input channel costs 4*64 = 256 params; each static feature costs 1.
    # Static encoding: +21 params (+0.1%); channels encoding: +5,376 (+31%).
