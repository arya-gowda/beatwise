"""Retune UMAP's min_dist and spread against genre coherence *and* usability.

The current settings (min_dist=0.8, spread=2.5) were chosen to make the galaxy look
good, and they wash out the local structure that genre coherence depends on. But tuning
purely for genre purity collapses the cloud into a clump, which breaks the interactions
the map exists for -- lasso selection and path drawing both need points you can tell
apart and hit with a cursor.

So three metrics per configuration, and the answer is a frontier rather than a maximum:

  purity          genre coherence -- the thing we are trying to improve
  trustworthiness whether the layout is faithful to the input space at all
  overplot        fraction of points visually on top of another point

    python tune_umap.py            # 2D grid sweep
    python tune_umap.py --quick    # smaller grid, one seed
"""

import argparse
import itertools
import json

import numpy as np
from sklearn.manifold import trustworthiness
from umap import UMAP

from ml import data, evaluate

SEEDS = (42, 7, 1234)


def overplot_rate(emb, tol=0.005):
    """Fraction of points whose nearest neighbour is closer than `tol` of the extent.

    The embedding is normalised to a unit bounding box first, so this is
    resolution-independent: at tol=0.005 on an 800px canvas, two points are within
    ~4px and are effectively one dot to the user. High values mean a map you cannot
    click accurately or lasso meaningfully.
    """
    lo, hi = emb.min(0), emb.max(0)
    span = np.maximum(hi - lo, 1e-12)
    z = (emb - lo) / span.max()

    sq = (z ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * z @ z.T, 0.0))
    np.fill_diagonal(D, np.inf)
    return float((D.min(1) < tol).mean())


def occupancy(emb, grid=48):
    """Fraction of grid cells containing at least one point -- how much of the canvas
    is actually used. Very low occupancy means everything is piled in one corner."""
    lo, hi = emb.min(0), emb.max(0)
    span = np.maximum(hi - lo, 1e-12)
    z = (emb - lo) / span
    idx = np.clip((z[:, :2] * grid).astype(int), 0, grid - 1)
    return float(len(set(map(tuple, idx))) / (grid * grid))


def run_config(X, lab_idx, S, mask, min_dist, spread, *, seeds, k, n_components):
    purity, trust, over, occ = [], [], [], []
    for seed in seeds:
        emb = UMAP(
            n_components=n_components, n_neighbors=8,
            min_dist=min_dist, spread=spread,
            metric="cosine", random_state=seed, negative_sample_rate=15,
        ).fit_transform(X)
        purity.append(evaluate.knn_genre_purity(emb[lab_idx], S, mask, k=k))
        trust.append(trustworthiness(X, emb, n_neighbors=k))
        over.append(overplot_rate(emb))
        occ.append(occupancy(emb))
    return {
        "min_dist": min_dist, "spread": spread,
        "purity": float(np.mean(purity)), "purity_std": float(np.std(purity)),
        "trust": float(np.mean(trust)),
        "overplot": float(np.mean(over)), "occupancy": float(np.mean(occ)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--components", type=int, default=2)
    args = ap.parse_args()

    df = data.load_all()
    X, _, _ = data.build_features(df)
    lab = df[df.has_genre].reset_index()
    lab_idx = lab["index"].to_numpy()
    S, vocab, _ = data.genre_targets(lab)
    mask = data.mask_same_artist(lab)

    print(f"{len(df)} tracks embedded, {len(lab_idx)} labelled, {len(vocab)} genres")
    print(f"current settings are min_dist=0.8, spread=2.5\n")

    if args.quick:
        grid = list(itertools.product([0.0, 0.1, 0.8], [1.0, 2.5]))
        seeds = SEEDS[:1]
    else:
        grid = list(itertools.product(
            [0.0, 0.05, 0.1, 0.25, 0.5, 0.8], [0.5, 1.0, 1.5, 2.5]))
        seeds = SEEDS
    # UMAP requires min_dist <= spread.
    grid = [(m, s) for m, s in grid if m <= s]

    print(f"{'min_dist':>9}{'spread':>8}{'purity':>10}{'±':>8}"
          f"{'trust':>8}{'overplot':>10}{'occupancy':>11}")
    print("-" * 64)
    rows = []
    for md, sp in grid:
        r = run_config(X, lab_idx, S, mask, md, sp,
                       seeds=seeds, k=args.k, n_components=args.components)
        rows.append(r)
        cur = "  <- current" if (md, sp) == (0.8, 2.5) else ""
        print(f"{md:>9.2f}{sp:>8.2f}{r['purity']:>10.4f}{r['purity_std']:>8.4f}"
              f"{r['trust']:>8.4f}{r['overplot']:>10.3f}{r['occupancy']:>11.3f}{cur}")

    base = next(r for r in rows if (r["min_dist"], r["spread"]) == (0.8, 2.5))

    # The current map is the known-acceptable density -- it is what the project has
    # been looking at -- so it sets the usability bar rather than an arbitrary absolute.
    # A config may buy genre coherence, but not by making the cloud denser than today.
    limit = base["overplot"] * 1.15
    usable = [r for r in rows if r["overplot"] <= limit]
    best = max(usable, key=lambda r: r["purity"]) if usable else None
    print(f"\nusability bar: overplot <= {limit:.3f} (current {base['overplot']:.3f} +15%)"
          f"  -> {len(usable)}/{len(rows)} configs qualify")

    print(f"\ncurrent : purity {base['purity']:.4f}  trust {base['trust']:.4f}  "
          f"overplot {base['overplot']:.3f}")
    if best:
        print(f"best    : min_dist={best['min_dist']} spread={best['spread']}  "
              f"purity {best['purity']:.4f} ({100*(best['purity']-base['purity'])/base['purity']:+.1f}%)  "
              f"trust {best['trust']:.4f}  overplot {best['overplot']:.3f}")

    with open("umap_tuning.json", "w") as fh:
        json.dump({"current": base, "best_usable": best, "grid": rows,
                   "n_tracks": int(len(df)), "k": args.k,
                   "n_components": args.components}, fh, indent=2)
    print("\nwrote umap_tuning.json")


if __name__ == "__main__":
    main()
