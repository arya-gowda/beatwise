"""Projecting tracks into an existing map.

This is the read side of the reproducibility guarantee (concept doc §5, seam item 3):
new tracks are placed into the space that already exists, using the reducer that was
fitted once. Nothing here fits. The API imports this module, and the absence of a fit
path is the point -- a re-fit silently moves every existing point, which scrambles the
territories a user has learned and breaks any saved route.
"""

import joblib
import numpy as np

from . import artifact, features


def load_transform(version):
    """The persisted scaler + reducer, with the feature list they were fitted on."""
    bundle = joblib.load(artifact.artifact_dir(version) / artifact.REDUCER)
    return bundle["scaler"], bundle["reducer"], bundle["features"]


def project(version, df):
    """Place rows into the existing embedding. Returns an (n, 2) array.

    Asserts the feature list matches what the reducer was fitted on. A reducer applied
    to columns in a different order produces plausible coordinates that are wrong, with
    no error anywhere -- so the check is worth more than it costs.
    """
    scaler, reducer, fitted_features = load_transform(version)
    if list(fitted_features) != list(features.FEATURES):
        raise ValueError(
            f"feature mismatch: reducer was fitted on {fitted_features}, "
            f"current pipeline uses {features.FEATURES}"
        )
    X = scaler.transform(df[fitted_features].to_numpy(dtype=np.float64))
    return reducer.transform(X)
