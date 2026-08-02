"""Three metric-learning models of increasing expressiveness.

All three map standardised audio features into a space where distance is measured, and
all three normalise their output to a fixed global scale. That normalisation is not
cosmetic: the NCA objective can be trivially maximised by inflating the scale until the
neighbour softmax becomes one-hot, so leaving scale free makes the problem degenerate.
Only relative geometry is learnable.
"""

import torch
import torch.nn as nn


def _normalise_scale(z):
    """Fix global scale so the neighbour softmax temperature means something."""
    rms = torch.sqrt((z ** 2).sum(dim=1).mean()).clamp_min(1e-8)
    return z / rms


class DiagonalMetric(nn.Module):
    """One learnable weight per feature -- the interpretable model.

    Parameterised in log space so weights stay positive, and the log-weights are
    re-centred to mean zero on every forward pass. That fixes the geometric mean at 1,
    leaving only the ratios between features free, which is exactly the quantity that
    is meaningful ("Energy matters 3x more than Liveness").
    """

    kind = "diagonal"

    def __init__(self, weight_index, n_weights=11):
        super().__init__()
        self.register_buffer("weight_index", torch.as_tensor(weight_index, dtype=torch.long))
        self.log_w = nn.Parameter(torch.zeros(n_weights, dtype=torch.float64))

    def weights(self):
        centred = self.log_w - self.log_w.mean()
        return torch.exp(centred)

    def forward(self, x):
        return _normalise_scale(x * self.weights()[self.weight_index])

    def regulariser(self):
        # Pull toward uniform weighting. This is the dial that controls how far the
        # map is allowed to drift from "position means sound".
        return (self.log_w - self.log_w.mean()).pow(2).mean()


class LinearMetric(nn.Module):
    """Full Mahalanobis metric: learns rotations and feature interactions.

    Strictly more expressive than diagonal, but the output axes no longer correspond
    to named features, so the learned transform cannot be read as per-feature weights.
    """

    kind = "linear"

    def __init__(self, d_in):
        super().__init__()
        self.L = nn.Parameter(torch.eye(d_in, dtype=torch.float64))

    def forward(self, x):
        return _normalise_scale(x @ self.L)

    def regulariser(self):
        eye = torch.eye(self.L.shape[0], dtype=self.L.dtype, device=self.L.device)
        return (self.L - eye).pow(2).mean()


class MLPMetric(nn.Module):
    """Nonlinear encoder -- the expressiveness ceiling, and the collapse risk.

    Free to fold the space however it likes to separate genres, which is why it needs
    watching: the highest genre score here may correspond to a map that has stopped
    representing sound at all.
    """

    kind = "mlp"

    def __init__(self, d_in, hidden=64, d_out=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out),
        ).double()

    def forward(self, x):
        return _normalise_scale(self.net(x))

    def regulariser(self):
        return sum(p.pow(2).mean() for p in self.parameters()) / len(list(self.parameters()))


def build(kind, weight_index, d_in):
    if kind == "diagonal":
        return DiagonalMetric(weight_index, n_weights=int(weight_index.max()) + 1)
    if kind == "linear":
        return LinearMetric(d_in)
    if kind == "mlp":
        return MLPMetric(d_in)
    raise ValueError(f"unknown model kind: {kind}")
