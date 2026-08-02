"""Metrics for the learned metric, plus the guardrails against a degenerate result.

Two families of metric here, and both matter. The genre metrics say whether the
objective was achieved; the distortion metrics say whether achieving it destroyed the
thing the map is for. A model that wins on the first and loses badly on the second has
turned a map of sound into a genre browser.
"""

import numpy as np
from scipy.stats import spearmanr


def knn_genre_purity(z, S, mask, k=8):
    """Mean IDF-weighted genre overlap with the k nearest permitted neighbours.

    This is the headline metric. k defaults to 8 to match n_neighbors in
    map_visualization_3d.py -- the neighbourhood UMAP actually builds on.
    """
    D = _pairwise(z)
    D = np.where(mask, D, np.inf)

    scores = []
    for i in range(len(z)):
        cand = np.where(np.isfinite(D[i]))[0]
        if len(cand) == 0:
            continue
        nn = cand[np.argsort(D[i, cand])[:k]]
        scores.append(S[i, nn].mean())
    return float(np.mean(scores)) if scores else float("nan")


def participation_ratio(weights):
    """Effective number of features in use, from 1 (one feature) to d (all equally).

    PR = (sum w^2)^2 / sum w^4. A model that collapses onto two or three features has
    a low PR, which is a concrete, readable signal that the map has stopped using most
    of what makes tracks sound different.
    """
    w2 = np.asarray(weights, dtype=float) ** 2
    return float(w2.sum() ** 2 / np.maximum((w2 ** 2).sum(), 1e-12))


def geometry_drift(z_learned, z_baseline, sample=400, seed=0):
    """Spearman correlation between learned and uniform-weight pairwise distances.

    1.0 means the geometry is untouched; low values mean the map has been substantially
    rearranged. Sampled because the full matrix is unnecessary for a rank correlation.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(z_learned), size=min(sample, len(z_learned)), replace=False)
    a = _pairwise(z_learned[idx])[np.triu_indices(len(idx), k=1)]
    b = _pairwise(z_baseline[idx])[np.triu_indices(len(idx), k=1)]
    return float(spearmanr(a, b).statistic)


def umap_survival(z, S, mask, k=8, n_components=2, seed=42):
    """Genre purity measured *after* UMAP, not just in the re-weighted input space.

    UMAP is nonlinear, so a gain in the input metric is not guaranteed to survive into
    the embedding the user actually looks at. This is the number that counts.
    """
    from umap import UMAP

    reducer = UMAP(
        n_components=n_components, n_neighbors=8, min_dist=0.8, spread=2.5,
        metric="cosine", random_state=seed, negative_sample_rate=15,
    )
    emb = reducer.fit_transform(z)
    return knn_genre_purity(emb, S, mask, k=k)


def _pairwise(z):
    sq = (z ** 2).sum(1)
    D = sq[:, None] + sq[None, :] - 2.0 * (z @ z.T)
    return np.sqrt(np.maximum(D, 0.0))
