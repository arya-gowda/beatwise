"""Reading and writing the map artifact.

Read-only with respect to the embedding: nothing here fits anything. The API imports
this module; it must never gain the ability to produce a new embedding. See
pipeline/build.py for why.
"""

import json
from pathlib import Path

ARTIFACTS = Path("artifacts")
MANIFEST = "manifest.json"
POINTS = "points.json"
REDUCER = "reducer.joblib"


def artifact_dir(version):
    return ARTIFACTS / version


def latest_version():
    """Most recently built artifact, or None. Version directories sort by build time
    because the version string is prefixed with a UTC timestamp."""
    if not ARTIFACTS.exists():
        return None
    versions = sorted(p.name for p in ARTIFACTS.iterdir() if (p / MANIFEST).exists())
    return versions[-1] if versions else None


def load_manifest(version):
    with open(artifact_dir(version) / MANIFEST) as fh:
        return json.load(fh)


def load_points(version):
    with open(artifact_dir(version) / POINTS) as fh:
        return json.load(fh)


def write(version, manifest, points):
    """Write manifest and points. The reducer is written by the build CLI itself,
    since this module is imported by a process that must not be able to fit one."""
    out = artifact_dir(version)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    # Records rather than columns: 2,389 rows is small and records are far easier to
    # inspect by hand. Revisit at Phase 4 scale, where deck.gl wants typed arrays and
    # this format stops being free -- an artifact version bump, not a migration.
    with open(out / POINTS, "w") as fh:
        json.dump(points, fh)
    return out
