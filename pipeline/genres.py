"""Genre tier 1: parse the CSV's `Genres` column into the per-label schema (§8.6).

    python -m pipeline.genres Liked_Songs.csv

Tier 1 is NOT a fetch. The `Genres` column in a Spotify CSV export already IS Spotify
artist genres — the same strings the Web API would return for those artists. This module
parses what is already there into a queryable shape. Fetching from the API is tier 2
(P1-13) and lives elsewhere.

Nothing here fits anything, and nothing here can move a point. Genre labels decorate a
track; they are live-layer-shaped data that happens to arrive in the substrate file, so
they are versioned SEPARATELY from the embedding — see docs/decisions/0004-genre-artifact.md.

SEPARATOR WARNING, and it is a real one. `Genres` is COMMA-separated: measured at 628
comma-bearing rows and zero semicolon-bearing rows. `Artist Name(s)` in the same file is
SEMICOLON-separated, and a comma split there silently fused multi-artist credits into
composite names until commit 5533a94 (27 rows have a comma *inside* one artist name, e.g.
"Tyler, The Creator"). One column's separator does not generalise to the next. The
semicolon guard in parse_labels() exists so that if a future export changes this column's
convention, the build stops instead of producing 2,404 subtly wrong labels.
"""

import argparse
import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone

from . import artifact, features

# Bump when the emitted row shape or the normalisation changes. Recorded in the manifest
# and folded into the version digest, so artifacts built under different rules are
# distinguishable rather than silently comparable.
SCHEMA_VERSION = 1

# The separator for THIS COLUMN. See the module docstring before reusing it anywhere.
SEPARATOR = ","

SOURCE_CSV = "csv"
SCOPE_ARTIST = "artist"
SCOPE_TRACK = "track"

STATUS_NATIVE = "native"
STATUS_UNLABELLED = "unlabelled"
# Phase 1 has exactly two states. Tiers 2-4 (§8.8) add more; until then any third value
# is a bug, and the tests assert the set is closed.
PHASE_1_STATUSES = (STATUS_NATIVE, STATUS_UNLABELLED)

# The §8.6 row, in this order. Recorded in the manifest for the same reason
# features.FEATURES is: a consumer reading columns positionally must be able to check.
LABEL_FIELDS = ("track_uri", "label", "rank", "source", "scope", "confidence", "fetched_at")
ROLLUP_FIELDS = (
    "track_uri", "genre_micro_primary", "genre_micro_all",
    "genre_label_count", "genre_status",
)

# Track scope beats artist scope in primary selection (§8.6). Phase 1 emits artist only.
_SCOPE_PRECEDENCE = {SCOPE_TRACK: 0, SCOPE_ARTIST: 1}

_WHITESPACE = re.compile(r"\s+")


def normalise_labels(value):
    """Split one `Genres` cell into an ordered, deduplicated list of micro tokens.

    Order is preserved because RANK IS SIGNAL, not incidental: "surf rock" is the first
    listed genre in 45 of the 45 tracks it appears on, "west coast hip hop" in 0 of 37.
    Whatever produced these strings puts the defining genre first often enough that
    discarding the order would throw away the only free ranking we have.

    Normalisation is insurance, not repair. Measured across all 2,404 label instances in
    this export: zero tokens differ from their normalised form, zero intra-track
    duplicates, zero empty tokens. That is a property of today's file, not a guarantee,
    and tests/test_genres.py asserts it stays true so a change surfaces as a failure
    rather than as a quietly larger vocabulary.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []

    text = unicodedata.normalize("NFKC", str(value))
    tokens = []
    for raw in text.split(SEPARATOR):
        token = _WHITESPACE.sub(" ", raw).strip().lower()
        if token:
            tokens.append(token)
    # dict.fromkeys keeps the FIRST occurrence, so a duplicate never steals the better
    # rank from the token that earned it.
    return list(dict.fromkeys(tokens))


def _check_separator(df):
    """Fail loudly if `Genres` ever starts carrying semicolons.

    A silent switch here would look exactly like the artist bug: plausible output, wrong
    identities, no error. Cheap to check once per build, and the alternative is finding
    out via a map coloured by genres that do not exist.
    """
    offenders = df.index[df["Genres"].astype(str).str.contains(";", regex=False)]
    if len(offenders):
        sample = df.loc[offenders[:3], "Genres"].tolist()
        raise ValueError(
            f"{len(offenders)} rows in `Genres` contain a semicolon, e.g. {sample}. "
            "This column was measured as comma-separated with zero semicolons; a change "
            "means the export convention moved and the split must be re-derived, not "
            "guessed. See ml/data.py for what guessing cost last time."
        )


def parse_labels(df, fetched_at, source=SOURCE_CSV, scope=SCOPE_ARTIST, confidence=1.00):
    """One row per (track, label), never a joined string.

    A joined string cannot answer "which tracks are shoegaze" without a substring match
    that also matches "blackgaze", and cannot carry per-label rank, scope or confidence
    at all. The relational shape is what makes tiers 2-4 additive: a fetched label is a
    new row with a different source, not a re-parse of an existing one.

    Tracks with an empty `Genres` cell produce ZERO rows. An empty-label row would be a
    lie — it asserts a label exists and that it is the empty string.
    """
    _check_separator(df)

    rows = []
    for uri, raw in zip(df["Track URI"], df["Genres"], strict=True):
        for rank, label in enumerate(normalise_labels(raw)):
            rows.append({
                "track_uri": uri,
                "label": label,
                "rank": rank,          # 0-based; contiguous per track
                "source": source,
                "scope": scope,
                "confidence": float(confidence),
                "fetched_at": fetched_at,
            })
    return rows


def primary_sort_key(row):
    """§8.6 primary selection: confidence desc, track scope over artist scope, lowest
    rank, then alphabetical.

    Every Phase 1 label is artist-scope at 1.00, so in practice rank decides and
    alphabetical is the tiebreak that never fires. Implemented in full anyway because
    P1-11 selects a macro primary with the same rule, and because without the last two
    terms a 14-label track (the maximum here) is a 14-way tie resolved by dict ordering.
    """
    return (
        -float(row["confidence"]),
        _SCOPE_PRECEDENCE.get(row["scope"], len(_SCOPE_PRECEDENCE)),
        int(row["rank"]),
        row["label"],
    )


def select_primary(rows):
    """The one label that represents a track, or None if it has none."""
    return min(rows, key=primary_sort_key)["label"] if rows else None


def rollups(df, rows):
    """Denormalised per-track view of the label table.

    Kept derived rather than authoritative: labels.json is the source of truth and this
    is a materialised join, so P1-12 can colour 2,389 points without grouping 2,404 rows
    in the browser on every render. `genre_micro_all` is ordered by rank.
    """
    by_track = {}
    for row in rows:
        by_track.setdefault(row["track_uri"], []).append(row)

    out = []
    for uri in df["Track URI"]:
        track_rows = sorted(by_track.get(uri, []), key=primary_sort_key)
        out.append({
            "track_uri": uri,
            "genre_micro_primary": select_primary(track_rows),
            "genre_micro_all": [r["label"] for r in track_rows],
            "genre_label_count": len(track_rows),
            "genre_status": STATUS_NATIVE if track_rows else STATUS_UNLABELLED,
        })
    return out


def vocabulary(rows):
    """Every distinct token with the evidence P1-11 needs to hand-map it.

    `track_count` says how much of the library a mapping decision moves; `first_count`
    says how often the token leads, which is how you tell a defining genre from a
    modifier without listening to anything.
    """
    stats = {}
    for row in rows:
        entry = stats.setdefault(row["label"], {"label": row["label"],
                                                "track_count": 0, "first_count": 0})
        entry["track_count"] += 1
        if row["rank"] == 0:
            entry["first_count"] += 1
    # Most common first: a hand-curation pass should spend its attention in proportion
    # to coverage, and the 106 singletons sort to the end where they belong.
    return sorted(stats.values(), key=lambda e: (-e["track_count"], e["label"]))


def version_id(csv_hash):
    """Timestamp for ordering, config digest for identity — same shape as the embedding
    version so the two are visibly the same kind of thing, and deliberately NOT the same
    value so a genre rebuild never implies the map moved."""
    config = json.dumps(
        {"schema": SCHEMA_VERSION, "separator": SEPARATOR, "fields": LABEL_FIELDS,
         "source": SOURCE_CSV, "scope": SCOPE_ARTIST, "csv": csv_hash},
        sort_keys=True,
    )
    digest = hashlib.sha256(config.encode()).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{digest}", digest


def build(source, user_id="local"):
    df, dropped = features.load_features(source)
    csv_hash = features.source_hash(source)
    version, digest = version_id(csv_hash)

    # For source=csv there is no fetch event to timestamp: the export carries no date and
    # Spotify's own labelling time is unknowable. This is when Beatwise obtained the
    # labels. Real provenance is source_sha256 in the manifest; tier 2 rows will carry
    # genuinely varying values here.
    fetched_at = datetime.now(timezone.utc).isoformat()

    rows = parse_labels(df, fetched_at)
    tracks = rollups(df, rows)
    vocab = vocabulary(rows)

    labelled = sum(1 for t in tracks if t["genre_status"] == STATUS_NATIVE)
    unlabelled = len(tracks) - labelled
    singletons = sum(1 for v in vocab if v["track_count"] == 1)

    manifest = {
        "genre_version": version,
        "config_digest": digest,
        "schema_version": SCHEMA_VERSION,
        "user_id": user_id,          # one user today; carried so multi-user is not a rewrite
        "source_csv": str(source),
        "source_sha256": csv_hash,   # the join back to the embedding built from these bytes
        "separator": SEPARATOR,
        "label_fields": list(LABEL_FIELDS),
        "rollup_fields": list(ROLLUP_FIELDS),
        "statuses": list(PHASE_1_STATUSES),
        "tiers": [SOURCE_CSV],
        "n_tracks": len(tracks),
        "n_dropped": dropped,
        "n_label_instances": len(rows),
        "n_labelled_tracks": labelled,
        "n_unlabelled_tracks": unlabelled,
        "n_unique_labels": len(vocab),
        "n_singleton_labels": singletons,
        "coverage": round(labelled / len(tracks), 4) if tracks else 0.0,
        "fetched_at": fetched_at,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    out = artifact.write_genres(version, manifest, rows, tracks, vocab)
    print(f"wrote {out}")
    print(f"  {len(rows)} label instances across {labelled} tracks "
          f"({unlabelled} unlabelled, {manifest['coverage']:.0%} coverage)")
    print(f"  {len(vocab)} unique labels, {singletons} singletons")
    return version


def main():
    ap = argparse.ArgumentParser(description="Parse CSV genres into the per-label schema")
    ap.add_argument("source", nargs="?", default="Liked_Songs.csv")
    ap.add_argument("--user-id", default="local")
    args = ap.parse_args()
    build(args.source, user_id=args.user_id)


if __name__ == "__main__":
    main()
