"""Build a versioned 2D map artifact from a Spotify CSV export.

    python -m pipeline.build Liked_Songs.csv

THIS IS THE ONLY MODULE THAT FITS AN EMBEDDING. It is a CLI, and the API must never
import it -- see tests/test_api_cannot_fit.py, which enforces that structurally by
asserting umap never reaches the API's import graph.

The reason is concept doc §5, seam item 3: a re-fit silently relocates every existing
point. Territories a user has learned move, and saved routes resolve to different
tracks, with nothing anywhere reporting an error. Fitting is therefore a deliberate,
versioned, offline act; everything else projects through the transform this produced.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from . import artifact, features

# Matches map_visualization_3d.py, reduced to 2 components. Settled by a 23-config x
# 3-seed sweep (concept doc §9.2): genre purity across the entire grid spans
# 0.0484-0.0526 against 0.0491 here, inside seed noise, and these values sit near the
# best available trustworthiness/overplot balance. Do not re-run that sweep.
UMAP_PARAMS = {
    "n_components": 2,
    "n_neighbors": 8,
    "min_dist": 0.8,
    "spread": 2.5,
    "metric": "cosine",
    "random_state": 42,
    "negative_sample_rate": 15,
}


def version_id(csv_hash):
    """Timestamp for ordering, config digest for identity. Two builds of the same input
    with the same settings produce the same digest, so a rebuild is recognisable as one."""
    config = json.dumps(
        {"umap": UMAP_PARAMS, "features": features.FEATURES,
         "key": features.KEY_ENCODING, "csv": csv_hash},
        sort_keys=True,
    )
    digest = hashlib.sha256(config.encode()).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{digest}", digest


def build(source, user_id="local"):
    df, dropped = features.load_features(source)
    print(f"loaded {len(df)} tracks" + (f", dropped {dropped} with missing features"
                                        if dropped else ", dropped none"))
    if dropped:
        # Silent row loss is the known hazard here; make it impossible to miss.
        print(f"  WARNING: {dropped} tracks are absent from the map entirely")

    csv_hash = features.source_hash(source)
    version, digest = version_id(csv_hash)

    X = df[features.FEATURES].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    reducer = UMAP(**UMAP_PARAMS)
    coords = reducer.fit_transform(Xs)

    points = [
        {
            "uri": r["Track URI"],
            "name": r["Track Name"],
            "artists": r["Artist Name(s)"],
            "album": r["Album Name"],
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "popularity": None if np.isnan(r["Popularity"]) else int(r["Popularity"]),
            "release_date": str(r["Release Date"]),
            "added_at": str(r["Added At"]),
            "tempo": float(r["Tempo"]),
            "explicit": bool(r["Explicit"]),
            "genres": [g.strip() for g in r["Genres"].split(",") if g.strip()],
        }
        for i, r in enumerate(df.to_dict("records"))
    ]

    manifest = {
        "embedding_version": version,
        "config_digest": digest,
        "user_id": user_id,          # one user today; carried so multi-user is not a rewrite
        "source_csv": str(source),
        "source_sha256": csv_hash,
        "n_tracks": len(points),
        "n_dropped": dropped,
        "features": features.FEATURES,
        "key_encoding": features.KEY_ENCODING,
        "umap": UMAP_PARAMS,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    out = artifact.write(version, manifest, points)
    joblib.dump(
        {"scaler": scaler, "reducer": reducer, "features": features.FEATURES},
        out / artifact.REDUCER,
    )
    print(f"wrote {out} ({len(points)} points, digest {digest})")
    return version


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", nargs="?", default="Liked_Songs.csv")
    ap.add_argument("--user-id", default="local")
    args = ap.parse_args()
    build(args.source, user_id=args.user_id)


if __name__ == "__main__":
    main()
