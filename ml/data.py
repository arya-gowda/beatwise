"""Data loading and target construction for feature-weight learning.

Trains only on tracks that carry a genre. Genres in Liked_Songs.csv are artist-level,
which is the central methodological hazard here: every track by an artist shares an
identical label set, so a model can score well by learning artist identity rather than
genre. Same-artist pairs are therefore masked out of the objective, and splits are
grouped by artist. See mask_same_artist() and grouped_folds().
"""

import numpy as np
import pandas as pd

# The ten features map_visualization_3d.py feeds to UMAP.
BASE_FEATURES = [
    "Danceability", "Energy", "Loudness", "Speechiness", "Acousticness",
    "Instrumentalness", "Liveness", "Valence", "Tempo", "Mode",
]

# Key becomes two circular dimensions sharing one weight (see build_features).
FEATURE_NAMES = BASE_FEATURES + ["Key"]

# Pitch class -> position on the circle of fifths. Adjacent keys are musically
# adjacent, which a raw 0-11 integer or a 12-way one-hot both fail to express.
_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: 7, 8: 8, 3: 9, 10: 10, 5: 11}


def load_labelled(csv_path="Liked_Songs.csv"):
    """Rows with a non-empty Genres field, plus parsed genre lists and primary artist."""
    df = pd.read_csv(csv_path)
    df["genres_raw"] = df["Genres"].fillna("").astype(str).str.strip()
    df = df[df["genres_raw"] != ""].copy()

    # Measured clean across all 2,404 label instances (no stray whitespace, casing,
    # empty tokens or intra-track duplicates) but normalise anyway -- that is a
    # property of this export, not a guarantee.
    df["genres"] = df["genres_raw"].apply(
        lambda v: list(dict.fromkeys(t.strip().lower() for t in v.split(",") if t.strip()))
    )
    df = _artists(df)
    df = df.dropna(subset=BASE_FEATURES + ["Key"]).reset_index(drop=True)
    return df


def split_artists(value):
    """Split one `Artist Name(s)` cell into credited artist names.

    Spotify exports these SEMICOLON-separated ("Consequence;Kanye West"). Splitting on
    a comma instead — as this module originally did — leaves the whole string as one
    composite name, so a track by "Daniel Caesar" and one by "Daniel Caesar;John Mayer"
    look like different artists. Measured consequence: 2,094 of 4,208 genre-overlapping
    same-artist pairs, half of them, went unmasked into the training objective. 27 rows
    also carry a comma *inside* a single name ("Tyler, The Creator"), which a comma split
    would cut in half.

    Public because it is the ONLY artist splitter in the repo and callers outside this
    module need it — see diagnostics/genre_gap_at_source.py. `Genres` in the same file is
    comma-separated and has its own splitter in pipeline/genres.py; one column's
    convention does not generalise to the next.
    """
    return [a.strip() for a in str(value).split(";") if a.strip()]


def _artists(df):
    """Attach the split credits and the primary artist to a frame."""
    raw = df["Artist Name(s)"].fillna("").astype(str)
    df["artists"] = raw.apply(split_artists)
    df["artist"] = df["artists"].apply(lambda a: a[0] if a else "")
    return df


def load_all(csv_path="Liked_Songs.csv"):
    """Every usable track, with a boolean flag for whether it carries a genre.

    The real map embeds the whole library, so layout tuning has to be measured on the
    whole library -- genre metrics are then read off the labelled subset inside that
    embedding rather than from a labelled-only map that does not exist in the product.
    """
    df = pd.read_csv(csv_path)
    df["genres_raw"] = df["Genres"].fillna("").astype(str).str.strip()
    df["genres"] = df["genres_raw"].apply(
        lambda v: list(dict.fromkeys(t.strip().lower() for t in v.split(",") if t.strip()))
    )
    df["artist"] = (
        df["Artist Name(s)"].fillna("").astype(str).str.split(",").str[0].str.strip()
    )
    df["has_genre"] = df["genres_raw"] != ""
    df = df.dropna(subset=BASE_FEATURES + ["Key"]).reset_index(drop=True)
    return df


def build_features(df):
    """Standardised feature matrix and the column->weight mapping.

    Returns X (n, 12), weight_index (12,) mapping each column to one of 11 weights.
    The two Key columns share a weight, so "how much does key matter" stays a single
    readable number rather than twelve uninterpretable ones.
    """
    base = df[BASE_FEATURES].to_numpy(dtype=np.float64)
    mu, sigma = base.mean(0), base.std(0)
    sigma[sigma == 0] = 1.0
    base = (base - mu) / sigma

    pos = df["Key"].astype(int).map(_FIFTHS).to_numpy(dtype=np.float64)
    angle = 2.0 * np.pi * pos / 12.0
    # Unit circle, so Key contributes fixed magnitude regardless of which key it is.
    key = np.stack([np.cos(angle), np.sin(angle)], axis=1)

    X = np.concatenate([base, key], axis=1)
    weight_index = np.array(list(range(len(BASE_FEATURES))) + [10, 10])
    scaler = {"mean": mu.tolist(), "scale": sigma.tolist(), "features": BASE_FEATURES}
    return X, weight_index, scaler


def genre_targets(df, min_count=2):
    """IDF-weighted Jaccard similarity between every pair of tracks.

    Binary same/different throws away most of the signal: two tracks sharing the rare
    'afropiano' are far stronger evidence than two sharing 'rap', which covers 137
    tracks. Genres appearing on fewer than min_count tracks are dropped -- singletons
    (106 of 314 here) generate no positive pairs and only add noise to the IDF.
    """
    from collections import Counter

    counts = Counter(g for gs in df["genres"] for g in gs)
    vocab = sorted(g for g, c in counts.items() if c >= min_count)
    index = {g: i for i, g in enumerate(vocab)}

    n = len(df)
    M = np.zeros((n, len(vocab)), dtype=np.float64)
    for i, gs in enumerate(df["genres"]):
        for g in gs:
            if g in index:
                M[i, index[g]] = 1.0

    # Rarer genre -> higher weight. log(n/count) is standard IDF.
    idf = np.log(n / np.maximum(M.sum(0), 1.0))
    W = M * idf

    # Weighted Jaccard: sum(min) / sum(max), computed via inclusion-exclusion on the
    # binary support since weights are per-column constants.
    inter = M @ (M * idf).T
    totals = W.sum(1, keepdims=True)
    union = totals + totals.T - inter
    S = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(S, 0.0)
    return S, vocab, idf


def mask_same_artist(df):
    """False where two tracks share ANY artist -- those pairs are excluded everywhere.

    Artist-level genres make same-artist pairs perfect positives for free. Left in,
    they are the easiest signal available and the model learns artist identity rather
    than genre.

    Sharing *any* credited artist counts, not merely the primary one: a solo track and
    a collaboration by the same person carry identical genre labels, so the pair is
    just as leaky as two solo tracks.
    """
    names = sorted({a for row in df["artists"] for a in row})
    index = {a: i for i, a in enumerate(names)}
    M = np.zeros((len(df), len(names)), dtype=bool)
    for i, row in enumerate(df["artists"]):
        for a in row:
            M[i, index[a]] = True

    shares = (M.astype(np.int16) @ M.astype(np.int16).T) > 0
    mask = ~shares
    np.fill_diagonal(mask, False)
    return mask


def artist_groups(df):
    """Group id per track, where any two tracks sharing an artist land in one group.

    Grouping on the primary artist alone still leaks: a collaboration can put the same
    person on both sides of a split. Connected components over the artist co-occurrence
    graph closes that, so a fold boundary never cuts through a shared credit.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    names = sorted({a for row in df["artists"] for a in row})
    index = {a: i for i, a in enumerate(names)}
    rows, cols = [], []
    for i, row in enumerate(df["artists"]):
        for a in row:
            rows.append(i)
            cols.append(index[a])
    # Bipartite tracks <-> artists; components link tracks through shared credits.
    n, m = len(df), len(names)
    inc = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, m))
    big = coo_matrix(
        (np.concatenate([inc.data, inc.data]),
         (np.concatenate([inc.row, inc.col + n]), np.concatenate([inc.col + n, inc.row]))),
        shape=(n + m, n + m),
    )
    _, labels = connected_components(big, directed=False)
    return labels[:n]


def grouped_folds(df, n_splits=5, seed=42):
    """Artist-disjoint folds. An artist never appears in both train and test."""
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(df))
    return list(gkf.split(idx, groups=artist_groups(df)))
