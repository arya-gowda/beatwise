"""Genre tier 1 — the acceptance figures, asserted rather than assumed.

These numbers were measured against Liked_Songs.csv and are pinned here so that a
different export, a changed separator, or a normalisation tweak fails loudly instead of
producing a plausibly-sized vocabulary that is quietly wrong. If one of them changes, the
question to answer is which of the file and the expectation moved — not which constant to
edit.
"""

import subprocess
import sys
from collections import Counter

import pandas as pd
import pytest

from pipeline import artifact, features, genres

CSV = "Liked_Songs.csv"

# Measured 2026-08-02 against Liked_Songs.csv
# (sha256 d95aa828...bf8, 2,389 rows, zero dropped for missing features).
EXPECTED = {
    "tracks": 2389,
    "label_instances": 2404,
    "labelled_tracks": 1252,
    "unlabelled_tracks": 1137,
    "unique_labels": 314,
    "singleton_labels": 106,
    "comma_bearing_rows": 628,
    "semicolon_bearing_rows": 0,
    "max_labels_on_one_track": 14,
}

FETCHED_AT = "2026-08-02T00:00:00+00:00"


@pytest.fixture(scope="module")
def parsed():
    df, dropped = features.load_features(CSV)
    assert dropped == 0, f"{dropped} rows dropped for missing features; figures assume 0"
    rows = genres.parse_labels(df, FETCHED_AT)
    return df, rows, genres.rollups(df, rows), genres.vocabulary(rows)


# --- the counts ----------------------------------------------------------------------


def test_label_instances_and_labelled_tracks(parsed):
    df, rows, tracks, _ = parsed
    assert len(df) == EXPECTED["tracks"]
    assert len(rows) == EXPECTED["label_instances"]
    assert len({r["track_uri"] for r in rows}) == EXPECTED["labelled_tracks"]
    assert len(tracks) == EXPECTED["tracks"], "one rollup per track, labelled or not"


def test_vocabulary_size_and_singletons(parsed):
    _, rows, _, vocab = parsed
    assert len(vocab) == EXPECTED["unique_labels"]
    assert len([v for v in vocab if v["track_count"] == 1]) == EXPECTED["singleton_labels"]
    # The vocabulary must be the whole of it -- P1-11 maps every token, and a token in
    # the label table but not the vocabulary would be silently unmappable.
    assert {v["label"] for v in vocab} == {r["label"] for r in rows}
    assert sum(v["track_count"] for v in vocab) == EXPECTED["label_instances"]


def test_genre_status_is_native_or_unlabelled_and_nothing_else(parsed):
    _, _, tracks, _ = parsed
    counts = Counter(t["genre_status"] for t in tracks)
    assert set(counts) <= set(genres.PHASE_1_STATUSES), (
        f"unexpected genre_status values {set(counts) - set(genres.PHASE_1_STATUSES)}; "
        "Phase 1 has exactly two states (§8.8 tiers 2-4 add more)"
    )
    assert counts[genres.STATUS_NATIVE] == EXPECTED["labelled_tracks"]
    assert counts[genres.STATUS_UNLABELLED] == EXPECTED["unlabelled_tracks"]
    assert sum(counts.values()) == EXPECTED["tracks"]


# --- rank is signal ------------------------------------------------------------------


def test_rank_is_zero_based_and_contiguous(parsed):
    _, rows, _, _ = parsed
    by_track = {}
    for r in rows:
        by_track.setdefault(r["track_uri"], []).append(r["rank"])
    for uri, ranks in by_track.items():
        assert ranks == list(range(len(ranks))), f"{uri} has non-contiguous ranks {ranks}"
    assert max(len(r) for r in by_track.values()) == EXPECTED["max_labels_on_one_track"]


def test_first_position_carries_information(parsed):
    """If rank were noise these two tokens would lead at similar rates. They do not."""
    _, rows, _, _ = parsed
    appear, first = Counter(), Counter()
    for r in rows:
        appear[r["label"]] += 1
        if r["rank"] == 0:
            first[r["label"]] += 1

    assert appear["surf rock"] == 45
    assert first["surf rock"] == 45, "surf rock leads 100% of the time it appears"
    assert appear["west coast hip hop"] == 37
    assert first["west coast hip hop"] == 0, "west coast hip hop never leads"


def test_vocabulary_first_counts_match_the_label_table(parsed):
    _, rows, _, vocab = parsed
    lead = Counter(r["label"] for r in rows if r["rank"] == 0)
    for entry in vocab:
        assert entry["first_count"] == lead[entry["label"]]


# --- the separator -------------------------------------------------------------------


def test_genres_is_comma_separated_with_no_semicolons():
    """The measurement that justifies the split, kept as a test.

    `Artist Name(s)` in the same file is semicolon-separated and a comma split there
    corrupted artist identity (commit 5533a94). This asserts the two columns really do
    differ rather than trusting that one convention holds across the export.
    """
    df, _ = features.load_features(CSV)
    g = df["Genres"].astype(str)
    assert int(g.str.contains(",", regex=False).sum()) == EXPECTED["comma_bearing_rows"]
    assert int(g.str.contains(";", regex=False).sum()) == EXPECTED["semicolon_bearing_rows"]

    a = df["Artist Name(s)"].astype(str)
    assert int(a.str.contains(";", regex=False).sum()) == 484
    assert int(a.str.contains(",", regex=False).sum()) == 27, (
        "commas inside single artist names -- the reason that column is not comma-split"
    )


def test_a_semicolon_in_genres_stops_the_build():
    df = pd.DataFrame({
        "Track URI": ["spotify:track:a"],
        "Genres": ["indie rock;shoegaze"],
    })
    with pytest.raises(ValueError, match="semicolon"):
        genres.parse_labels(df, FETCHED_AT)


# --- normalisation -------------------------------------------------------------------


def test_normalisation_is_a_noop_on_this_export(parsed):
    """Normalisation is insurance. Assert it is currently doing nothing, so that the day
    it starts doing something is visible rather than absorbed into the vocabulary."""
    df, rows, _, _ = parsed
    raw = [
        token
        for cell in df["Genres"].astype(str)
        for token in cell.split(",")
        if token != ""
    ]
    assert len(raw) == EXPECTED["label_instances"], "normalisation dropped or merged tokens"
    assert [t for t in raw if t != t.strip().lower()] == []
    assert all(r["label"] == r["label"].strip().lower() for r in rows)


def test_accented_labels_are_composed_and_stay_that_way(parsed):
    """14 tokens are non-ASCII ('sierreño', 'música mexicana', 'variété française').

    They arrive composed (NFC), and the NFKC step keeps them there. This matters because
    a decomposed 'é' (e + combining acute) is a DIFFERENT string with the same glyph: it
    would sit in the vocabulary as a 315th token that P1-11's hand mapping cannot match
    and nobody can see is wrong. Pinned so the failure mode is a test, not a colour hole.
    """
    import unicodedata

    _, rows, _, vocab = parsed
    non_ascii = sorted({v["label"] for v in vocab if not v["label"].isascii()})
    assert len(non_ascii) == 14, non_ascii
    assert all(unicodedata.normalize("NFKC", label) == label for label in non_ascii)
    assert all(unicodedata.normalize("NFKC", r["label"]) == r["label"] for r in rows)
    # Decomposed input must collapse onto the same token rather than forking it.
    assert genres.normalise_labels(unicodedata.normalize("NFD", "Sierreño")) == ["sierreño"]


def test_normalisation_handles_what_it_claims_to():
    assert genres.normalise_labels("  Indie Rock , SHOEGAZE ") == ["indie rock", "shoegaze"]
    assert genres.normalise_labels("indie  rock") == ["indie rock"], "internal whitespace"
    # First occurrence wins, so a duplicate cannot steal the better rank.
    assert genres.normalise_labels("rock, pop, rock") == ["rock", "pop"]


def test_empty_genres_produce_zero_rows_not_an_empty_label():
    for value in ("", "   ", ",", ",,", " , ", None, float("nan")):
        assert genres.normalise_labels(value) == [], repr(value)

    df = pd.DataFrame({
        "Track URI": ["spotify:track:a", "spotify:track:b", "spotify:track:c"],
        "Genres": ["", "  ,  ", "rock"],
    })
    rows = genres.parse_labels(df, FETCHED_AT)
    assert [r["track_uri"] for r in rows] == ["spotify:track:c"]
    assert all(r["label"] for r in rows)

    tracks = {t["track_uri"]: t for t in genres.rollups(df, rows)}
    assert tracks["spotify:track:a"]["genre_label_count"] == 0
    assert tracks["spotify:track:a"]["genre_micro_all"] == []
    assert tracks["spotify:track:a"]["genre_micro_primary"] is None
    assert tracks["spotify:track:a"]["genre_status"] == genres.STATUS_UNLABELLED


def test_no_empty_label_survives_the_real_parse(parsed):
    _, rows, _, _ = parsed
    assert all(r["label"].strip() for r in rows)


# --- schema shape --------------------------------------------------------------------


def test_every_row_carries_the_full_schema(parsed):
    _, rows, _, _ = parsed
    for r in rows:
        assert tuple(r) == genres.LABEL_FIELDS, "field set or order drifted from §8.6"
    assert {r["source"] for r in rows} == {genres.SOURCE_CSV}
    assert {r["scope"] for r in rows} == {genres.SCOPE_ARTIST}
    assert {r["confidence"] for r in rows} == {1.00}
    assert {r["fetched_at"] for r in rows} == {FETCHED_AT}


def test_rollups_carry_the_full_schema_and_agree_with_the_labels(parsed):
    _, rows, tracks, _ = parsed
    by_track = {}
    for r in rows:
        by_track.setdefault(r["track_uri"], []).append(r["label"])

    for t in tracks:
        assert tuple(t) == genres.ROLLUP_FIELDS
        labels = by_track.get(t["track_uri"], [])
        assert t["genre_micro_all"] == labels, "rollup must preserve rank order"
        assert t["genre_label_count"] == len(labels)
        assert t["genre_micro_primary"] == (labels[0] if labels else None)


def test_track_uri_is_the_join_key(parsed):
    """Identity is the Spotify track URI throughout -- it is the only key that survives
    swapping the feature provider."""
    df, rows, tracks, _ = parsed
    uris = set(df["Track URI"])
    assert len(uris) == EXPECTED["tracks"], "Track URI is not unique in the source"
    assert {t["track_uri"] for t in tracks} == uris
    assert {r["track_uri"] for r in rows} <= uris
    assert all(str(u).startswith("spotify:track:") for u in uris)


# --- primary selection (§8.6), which P1-11 reuses for macro --------------------------


def test_primary_selection_precedence():
    def row(label, rank=0, scope=genres.SCOPE_ARTIST, confidence=1.0):
        return {"track_uri": "x", "label": label, "rank": rank, "source": "test",
                "scope": scope, "confidence": confidence, "fetched_at": FETCHED_AT}

    assert genres.select_primary([]) is None
    # confidence first
    assert genres.select_primary([row("a", confidence=0.5), row("b")]) == "b"
    # then track scope over artist scope, even at a worse rank
    assert genres.select_primary(
        [row("a", rank=0), row("b", rank=3, scope=genres.SCOPE_TRACK)]
    ) == "b"
    # then lowest rank
    assert genres.select_primary([row("z", rank=0), row("a", rank=1)]) == "z"
    # then alphabetical -- without which a 14-label tie resolves by dict ordering
    assert genres.select_primary([row("z", rank=0), row("a", rank=0)]) == "a"


def test_phase_1_primary_is_always_the_rank_zero_label(parsed):
    """The claim that makes P1-11's tie-break rule tractable: every Phase 1 label is
    artist-scope at 1.00, so rank alone decides and alphabetical never fires."""
    _, rows, tracks, _ = parsed
    rank_zero = {r["track_uri"]: r["label"] for r in rows if r["rank"] == 0}
    for t in tracks:
        assert t["genre_micro_primary"] == rank_zero.get(t["track_uri"])


# --- the artifact --------------------------------------------------------------------


def test_the_genre_parser_does_not_drag_in_the_fit_path():
    """The API reads genres, so genre code sits in its import graph. If it ever reached
    umap, tests/test_api_cannot_fit.py would fail -- catch it here with a clearer message."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import pipeline.genres; print('umap' in sys.modules)"],
        capture_output=True, text=True, cwd=".",
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "pipeline.genres must not import umap"


def test_genre_namespace_does_not_shadow_the_embedding_version():
    """`artifacts/genres/` sorts after every timestamped version. If latest_version()
    ever returned it, the API would serve a directory with no points."""
    assert artifact.latest_version() != artifact.GENRES


@pytest.mark.skipif(
    artifact.latest_genre_version() is None,
    reason="no genre artifact built; run python -m pipeline.genres",
)
def test_built_artifact_matches_the_measured_figures():
    version = artifact.latest_genre_version()
    manifest = artifact.load_genre_manifest(version)
    assert manifest["n_label_instances"] == EXPECTED["label_instances"]
    assert manifest["n_labelled_tracks"] == EXPECTED["labelled_tracks"]
    assert manifest["n_unlabelled_tracks"] == EXPECTED["unlabelled_tracks"]
    assert manifest["n_unique_labels"] == EXPECTED["unique_labels"]
    assert manifest["n_singleton_labels"] == EXPECTED["singleton_labels"]
    assert manifest["n_tracks"] == EXPECTED["tracks"]
    assert manifest["separator"] == ","
    assert manifest["statuses"] == list(genres.PHASE_1_STATUSES)
    assert manifest["user_id"], "user id must be carried even with one user"

    assert len(artifact.load_genre_labels(version)) == EXPECTED["label_instances"]
    assert len(artifact.load_genre_tracks(version)) == EXPECTED["tracks"]
    assert len(artifact.load_genre_vocabulary(version)) == EXPECTED["unique_labels"]


@pytest.mark.skipif(
    artifact.latest_genre_version() is None or artifact.latest_version() is None,
    reason="needs both a map and a genre artifact",
)
def test_genre_artifact_and_map_were_built_from_the_same_csv():
    """The two version lines are independent by design, which means they can drift. They
    join on Track URI, and a stale genre artifact would silently colour a subset."""
    gm = artifact.load_genre_manifest(artifact.latest_genre_version())
    em = artifact.load_manifest(artifact.latest_version())
    assert gm["source_sha256"] == em["source_sha256"], (
        "genre artifact was built from different CSV bytes than the map -- "
        "rerun python -m pipeline.genres"
    )

    tracks = {t["track_uri"] for t in artifact.load_genre_tracks(
        artifact.latest_genre_version())}
    points = {p["uri"] for p in artifact.load_points(artifact.latest_version())}
    assert tracks == points, "every point must have a rollup, colourable or not"
