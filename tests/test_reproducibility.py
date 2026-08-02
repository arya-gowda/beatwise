"""The reproducibility guarantee: adding tracks must not move existing points.

Concept doc §5, seam item 3. This is the property the whole persisted-reducer design
exists to provide, and it fails silently when broken -- points shift, territories the
user has learned move, saved routes resolve to different tracks, and nothing errors.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline import artifact, features, project

pytestmark = pytest.mark.skipif(
    artifact.latest_version() is None,
    reason="no artifact built; run python -m pipeline.build",
)


def _artifact():
    version = artifact.latest_version()
    return version, pd.DataFrame(artifact.load_points(version))


def test_existing_points_are_never_recomputed():
    """Projecting new tracks must not touch stored coordinates."""
    version, pts = _artifact()
    before = pts[["x", "y"]].to_numpy().copy()

    df, _ = features.load_features("Liked_Songs.csv")
    project.project(version, df.iloc[:50])

    after = pd.DataFrame(artifact.load_points(version))[["x", "y"]].to_numpy()
    assert np.array_equal(before, after), "stored coordinates changed during projection"


def test_projection_places_tracks_in_the_right_region():
    """A projected track lands near where the fit put it.

    UMAP's transform() is a different optimisation from fit_transform() and is NOT
    exact, even for rows that were in the fit. Measured here: median displacement
    ~1.3x the median nearest-neighbour spacing, ~0.9% of map extent. So projection is
    reliable at the level of *region*, not of exact neighbour ordering.

    The bound below is deliberately loose and exists to catch a broken transform (wrong
    feature order, wrong scaler), not to police UMAP's inherent approximation.
    """
    version, pts = _artifact()
    coords = pts[["x", "y"]].to_numpy()
    df, _ = features.load_features("Liked_Songs.csv")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), 200, replace=False)
    displacement = np.linalg.norm(project.project(version, df.iloc[idx]) - coords[idx], axis=1)

    extent = float((coords.max(0) - coords.min(0)).max())
    assert np.median(displacement) < 0.05 * extent, (
        f"median displacement {np.median(displacement):.3f} exceeds 5% of the "
        f"{extent:.1f} map extent -- the transform is broken, not merely approximate"
    )


def test_feature_order_mismatch_is_caught():
    """A reducer applied to columns in a different order yields plausible coordinates
    that are wrong, with no error. The guard must raise instead."""
    version, _ = _artifact()
    df, _ = features.load_features("Liked_Songs.csv")

    original = features.FEATURES[:]
    features.FEATURES.reverse()
    try:
        with pytest.raises(ValueError, match="feature mismatch"):
            project.project(version, df.iloc[:5])
    finally:
        features.FEATURES[:] = original
