"""Learn per-feature weights so genre-sharing tracks become map neighbours.

Compares three model classes on identical artist-disjoint folds, then fits the chosen
model on the full labelled set and writes feature_weights.json.

    python train_weights.py --smoke     # one fold, quick
    python train_weights.py             # full 5-fold comparison + lambda sweep
"""

import argparse
import json
import time

import numpy as np

from ml import data, evaluate, train


def _subset(S, mask, idx):
    return S[np.ix_(idx, idx)], mask[np.ix_(idx, idx)]


def run_cv(X, S, mask, wi, folds, kinds, *, temperature, lam, epochs, k):
    """Train each model class on every fold; score on held-out artists."""
    results = {kind: [] for kind in kinds}
    results["uniform"] = []

    for f, (tr, te) in enumerate(folds):
        S_te, M_te = _subset(S, mask, te)

        # Baseline: today's behaviour -- all features weighted equally.
        z_base = X[te] / np.sqrt((X[te] ** 2).sum(1).mean())
        results["uniform"].append(evaluate.knn_genre_purity(z_base, S_te, M_te, k=k))

        S_tr, M_tr = _subset(S, mask, tr)
        for kind in kinds:
            t0 = time.time()
            model = train.fit(kind, X[tr], S_tr, M_tr, wi,
                              temperature=temperature, lam=lam, epochs=epochs)
            z_te = train.transform(model, X[te])
            score = evaluate.knn_genre_purity(z_te, S_te, M_te, k=k)
            results[kind].append(score)
            print(f"  fold {f}  {kind:<9} purity {score:.4f}  ({time.time()-t0:.1f}s)")

    return results


def summarise(results, k):
    """Paired per-fold comparison. Folds share variance, so a paired test is the
    right one -- the unpaired mean alone hid that lambda=0 wins only 7/10 folds."""
    from scipy.stats import wilcoxon

    base_scores = np.array(results["uniform"])
    base = base_scores.mean()
    n = len(base_scores)
    print(f"\n{'model':<12}{'purity@'+str(k):>12}{'std':>9}{'vs uniform':>13}"
          f"{'wins':>9}{'p':>9}")
    print("-" * 64)
    for kind, scores in results.items():
        s_arr = np.array(scores)
        m, s = s_arr.mean(), s_arr.std()
        if kind == "uniform":
            print(f"{kind:<12}{m:>12.4f}{s:>9.4f}{'':>13}{'':>9}{'':>9}")
            continue
        wins = int((s_arr > base_scores).sum())
        p = wilcoxon(s_arr, base_scores).pvalue if n >= 5 else float("nan")
        print(f"{kind:<12}{m:>12.4f}{s:>9.4f}{100*(m-base)/base:>12.1f}%"
              f"{wins:>6}/{n}{p:>9.4f}")
    return base


def random_floor(X, S, mask, k, trials=20, seed=0):
    """Purity of a random neighbourhood -- the zero point the numbers sit above."""
    rng = np.random.default_rng(seed)
    return float(np.mean([
        evaluate.knn_genre_purity(rng.normal(size=X.shape), S, mask, k)
        for _ in range(trials)
    ]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.5)
    # lambda=0.05 is the validated operating point: it matches the unregularised lift
    # (+11.9% vs +12.2%) while winning 10/10 folds instead of 7/10 (p=0.002 vs 0.131),
    # and it keeps ~9.7 of 11 features in play instead of collapsing onto ~5.
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=400)
    args = ap.parse_args()

    print("gradcheck:", train.gradcheck())

    df = data.load_labelled()
    X, wi, scaler = data.build_features(df)
    S, vocab, idf = data.genre_targets(df)
    mask = data.mask_same_artist(df)
    folds = data.grouped_folds(df, n_splits=args.folds)

    print(f"\n{len(df)} labelled tracks, {df.artist.nunique()} artists, "
          f"{len(vocab)} genres, {(S>0).sum()//2} genre-sharing pairs")
    floor = random_floor(X, S, mask, args.k)
    print(f"random-neighbour floor: {floor:.4f}  (the zero point for every score below)")

    kinds = ["diagonal"] if args.smoke else ["diagonal", "linear", "mlp"]
    if args.smoke:
        folds = folds[:1]

    print(f"\n=== artist-disjoint CV (k={args.k}, T={args.temperature}, "
          f"lambda={args.lam}) ===")
    results = run_cv(X, S, mask, wi, folds, kinds,
                     temperature=args.temperature, lam=args.lam,
                     epochs=args.epochs, k=args.k)
    summarise(results, args.k)

    if args.smoke:
        return

    # --- trade-off curve: genre coherence vs how far the map is distorted ---
    print("\n=== lambda sweep (diagonal, full set) ===")
    print(f"{'lambda':>8}{'purity':>10}{'PR':>8}{'drift':>9}  weights")
    print("-" * 74)
    z_base = X / np.sqrt((X ** 2).sum(1).mean())
    curve = []
    for lam in [0.0, 0.01, 0.05, 0.2, 1.0, 5.0]:
        model = train.fit("diagonal", X, S, mask, wi,
                          temperature=args.temperature, lam=lam, epochs=args.epochs)
        w = model.weights().detach().numpy()
        z = train.transform(model, X)
        row = {
            "lam": lam,
            "purity": evaluate.knn_genre_purity(z, S, mask, k=args.k),
            "pr": evaluate.participation_ratio(w),
            "drift": evaluate.geometry_drift(z, z_base),
            "weights": w.tolist(),
        }
        curve.append(row)
        top = ", ".join(f"{n}={v:.2f}" for n, v in
                        sorted(zip(data.FEATURE_NAMES, w), key=lambda p: -p[1])[:3])
        print(f"{lam:>8.2f}{row['purity']:>10.4f}{row['pr']:>8.2f}"
              f"{row['drift']:>9.3f}  {top}")

    # --- final model on the full labelled set ---
    model = train.fit("diagonal", X, S, mask, wi,
                      temperature=args.temperature, lam=args.lam, epochs=args.epochs)
    w = model.weights().detach().numpy()
    z = train.transform(model, X)

    print("\n=== learned weights (lambda=%.2f) ===" % args.lam)
    for name, val in sorted(zip(data.FEATURE_NAMES, w), key=lambda p: -p[1]):
        print(f"  {name:<18}{val:>7.3f}  {'#' * int(round(val * 20))}")

    # UMAP layout is stochastic and a single seed is not enough to call a difference:
    # one seed initially showed -6.5% here, which averaging over three revealed as noise.
    print("\n=== does the gain survive UMAP? (3 seeds) ===")
    seeds = (42, 7, 1234)
    b = np.array([evaluate.umap_survival(z_base, S, mask, k=args.k, seed=s) for s in seeds])
    a = np.array([evaluate.umap_survival(z, S, mask, k=args.k, seed=s) for s in seeds])
    print(f"  uniform weights -> UMAP : {b.mean():.4f} +/- {b.std():.4f}")
    print(f"  learned weights -> UMAP : {a.mean():.4f} +/- {a.std():.4f}"
          f"   ({100*(a.mean()-b.mean())/b.mean():+.1f}%)")
    if abs(a.mean() - b.mean()) < max(a.std(), b.std()):
        print("  NOTE: difference is smaller than seed-to-seed variation -- the "
              "feature-space\n        gain does not reliably survive the layout.")
    base_umap, learn_umap = float(b.mean()), float(a.mean())

    out = {
        "feature_names": data.FEATURE_NAMES,
        "weights": dict(zip(data.FEATURE_NAMES, w.tolist())),
        "weight_index": wi.tolist(),
        "scaler": scaler,
        "training": {
            "temperature": args.temperature, "lam": args.lam,
            "epochs": args.epochs, "k": args.k,
            "n_tracks": int(len(df)), "n_artists": int(df.artist.nunique()),
            "n_genres": len(vocab),
        },
        "metrics": {
            "cv": {kk: [float(x) for x in v] for kk, v in results.items()},
            "umap_uniform": base_umap, "umap_learned": learn_umap,
            "participation_ratio": evaluate.participation_ratio(w),
            "random_floor": floor,
        },
        "curve": curve,
    }
    with open("feature_weights.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote feature_weights.json")


if __name__ == "__main__":
    main()
