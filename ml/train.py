"""Multi-label NCA objective and training loop.

The objective is neighbourhood-based rather than pairwise-margin based, because UMAP
consumes a k-nearest-neighbour graph -- optimising neighbourhood structure directly is
closer to the thing that actually determines the map than a triplet loss would be.
"""

import numpy as np
import torch

from . import models


def nca_loss(z, S, mask, temperature):
    """Negative expected genre agreement of a stochastic nearest neighbour.

    p_ij = softmax_j(-d(i,j)^2 / T) over permitted neighbours, then the objective is
    sum_j p_ij * S_ij where S is IDF-weighted Jaccard genre overlap. Maximising it
    pulls genre-sharing tracks into each other's neighbourhoods.
    """
    D = torch.cdist(z, z).pow(2)
    logits = (-D / temperature).masked_fill(~mask, float("-inf"))

    # A row whose every candidate is masked out (all of an artist's neighbours removed)
    # would softmax to NaN. Drop those rows rather than letting NaN poison the batch.
    valid = mask.any(dim=1)
    P = torch.softmax(logits[valid], dim=1)
    return -(P * S[valid]).sum(dim=1).mean()


def fit(kind, X, S, mask, weight_index, *, temperature=0.5, lam=0.0,
        epochs=600, lr=0.05, seed=42, verbose=False):
    """Train one model on one split. Full-batch and deterministic -- 1,252 tracks
    makes the entire 783k-pair objective exact, so there is no sampling noise."""
    torch.manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float64)
    St = torch.as_tensor(S, dtype=torch.float64)
    Mt = torch.as_tensor(mask, dtype=torch.bool)

    model = models.build(kind, weight_index, X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        opt.zero_grad()
        z = model(Xt)
        loss = nca_loss(z, St, Mt, temperature) + lam * model.regulariser()
        loss.backward()
        opt.step()
        if verbose and epoch % 100 == 0:
            print(f"  epoch {epoch:4d}  loss {loss.item():.5f}")

    return model


def transform(model, X):
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float64)).numpy()


def gradcheck():
    """Finite-difference check that the analytic gradient of the loss is correct.

    Cheap insurance: a silently wrong gradient produces a model that trains to a
    plausible-looking number and means nothing.
    """
    torch.manual_seed(0)
    n, d = 12, 5
    X = torch.randn(n, d, dtype=torch.float64)
    S = torch.rand(n, n, dtype=torch.float64)
    S = ((S + S.T) / 2).fill_diagonal_(0)
    mask = ~torch.eye(n, dtype=torch.bool)
    w_index = np.arange(d)

    w_index_t = torch.as_tensor(w_index, dtype=torch.long)

    def f(log_w):
        # Mirrors DiagonalMetric.forward exactly, but as a pure function of log_w so
        # the input stays attached to the graph.
        w = torch.exp(log_w - log_w.mean())
        z = X * w[w_index_t]
        z = z / torch.sqrt((z ** 2).sum(dim=1).mean()).clamp_min(1e-8)
        return nca_loss(z, S, mask, 0.5)

    log_w = (torch.randn(d, dtype=torch.float64) * 0.3).requires_grad_(True)
    return torch.autograd.gradcheck(f, (log_w,), eps=1e-6, atol=1e-7)
