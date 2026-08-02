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

# Genre artifacts live in their own namespace under artifacts/ and carry their own
# version line. They are derived from the same CSV but they never move a point, so
# rebuilding them must not require a new embedding id. See
# docs/decisions/0004-genre-artifact.md.
GENRES = "genres"
LABELS = "labels.json"
TRACK_GENRES = "tracks.json"
VOCABULARY = "vocabulary.json"
# Tokens the P1-11 curation pass has not seen. Derived, so it belongs here rather than in
# data/ -- unlike the map it is generated fresh on every build. The map itself is source
# and lives in data/genre_macro_map.json; see pipeline/macro.py.
MACRO_REVIEW = "macro_review_queue.json"


def artifact_dir(version):
    return ARTIFACTS / version


def latest_version():
    """Most recently built artifact, or None. Version directories sort by build time
    because the version string is prefixed with a UTC timestamp."""
    if not ARTIFACTS.exists():
        return None
    versions = sorted(
        p.name for p in ARTIFACTS.iterdir()
        # The genres namespace is not an embedding. It is excluded by name rather than
        # only by the manifest check below, because "genres" sorts after every timestamp
        # and would become the "latest" embedding the moment it gained a manifest.
        if p.name != GENRES and (p / MANIFEST).exists()
    )
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


# --- genre artifact ------------------------------------------------------------------
#
# Same read/write conventions, separate version line. Written by pipeline/genres.py,
# read by the API. Nothing here can fit or re-fit anything.


def genre_dir(version):
    return ARTIFACTS / GENRES / version


def latest_genre_version():
    root = ARTIFACTS / GENRES
    if not root.exists():
        return None
    versions = sorted(p.name for p in root.iterdir() if (p / MANIFEST).exists())
    return versions[-1] if versions else None


def _read(version, name):
    with open(genre_dir(version) / name) as fh:
        return json.load(fh)


def load_genre_manifest(version):
    return _read(version, MANIFEST)


def load_genre_labels(version):
    """The §8.6 table: one row per (track, label). The authoritative form."""
    return _read(version, LABELS)


def load_genre_tracks(version):
    """Per-track rollups, derived from the label table. What the map renders."""
    return _read(version, TRACK_GENRES)


def load_genre_vocabulary(version):
    """Distinct labels with counts. What the P1-11 curation pass works from."""
    return _read(version, VOCABULARY)


def load_macro_review(version):
    """Tokens with no row in the curated macro table. Empty file when there are none —
    a missing file cannot be told apart from a build that never checked."""
    return _read(version, MACRO_REVIEW)


def write_genres(version, manifest, labels, tracks, vocabulary, macro_review):
    out = genre_dir(version)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    # Records, matching points.json above, and revisitable at the same point: 2,404 rows
    # is nothing, ~2M rows at Phase 4's corpus scale is a columnar format.
    with open(out / LABELS, "w") as fh:
        json.dump(labels, fh)
    with open(out / TRACK_GENRES, "w") as fh:
        json.dump(tracks, fh)
    # Indented: this one is read by a human during P1-11's curation pass.
    with open(out / VOCABULARY, "w") as fh:
        json.dump(vocabulary, fh, indent=2)
    # Indented for the same reason -- it exists to be acted on by a person.
    with open(out / MACRO_REVIEW, "w") as fh:
        json.dump(macro_review, fh, indent=2, ensure_ascii=False)
    return out
