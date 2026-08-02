"""P1-11 — the curated micro->macro table, asserted rather than assumed.

Same contract as tests/test_genres.py: these numbers were measured against
Liked_Songs.csv and data/genre_macro_map.json, and they are pinned so that a recuration,
a new export, or a lost row fails loudly rather than producing a plausibly-coloured map
that is quietly wrong.

The distinction that matters here: the counts in EXPECTED are properties of a HAND-MADE
TABLE, not of the source data. Revising the taxonomy is supposed to move them. When one
changes, the question is whether the curation changed on purpose — not which constant to
edit to make the suite green.
"""

from collections import Counter

import pytest

from pipeline import artifact, features, genres, macro

CSV = "Liked_Songs.csv"

# Measured 2026-08-02 against Liked_Songs.csv + data/genre_macro_map.json.
EXPECTED = {
    "tokens": 314,            # every one mapped, including all 106 singletons
    "families": 12,
    "labelled_tracks": 1252,
    "multi_macro_tracks": 177,
    "max_macros_on_one_track": 5,
    "largest_family_primary": ("rock", 320),
    "smallest_family_primary": ("reggae", 7),
}

FETCHED_AT = "2026-08-02T00:00:00+00:00"


@pytest.fixture(scope="module")
def macro_map():
    return macro.load()


@pytest.fixture(scope="module")
def rolled(macro_map):
    df, dropped = features.load_features(CSV)
    assert dropped == 0
    rows = genres.parse_labels(df, FETCHED_AT)
    return rows, genres.rollups(df, rows, macro_map), genres.vocabulary(rows)


# --- the table covers everything ------------------------------------------------------


def test_every_token_in_the_vocabulary_is_mapped(rolled, macro_map):
    """The acceptance criterion, and the only one that cannot be partially met.

    An unmapped token is not a small gap: it is a point whose colour has no explanation,
    which is the exact property hand curation was chosen to buy (§8.3).
    """
    _, _, vocab = rolled
    tokens = [v["label"] for v in vocab]
    assert len(tokens) == EXPECTED["tokens"]
    missing = macro_map.unmapped(tokens)
    assert missing == [], f"{len(missing)} unmapped tokens: {missing}"


def test_the_singletons_are_mapped_too(rolled, macro_map):
    """106 of the 314 tokens appear on exactly one track. They are the cheapest to skip
    and the most likely to be skipped, so they get their own assertion."""
    _, _, vocab = rolled
    singletons = [v["label"] for v in vocab if v["track_count"] == 1]
    assert len(singletons) == 106
    assert macro_map.unmapped(singletons) == []


def test_the_table_maps_nothing_that_is_not_in_the_vocabulary(rolled, macro_map):
    """A row for a token that does not exist is a curation decision that will never fire,
    and usually means the token was typed rather than copied -- which for the 14 non-ASCII
    tokens (`sierreño`, `variété française`) is how a decomposed-accent twin gets in."""
    _, _, vocab = rolled
    orphans = sorted(set(macro_map.mapping) - {v["label"] for v in vocab})
    assert orphans == [], f"table maps tokens absent from the vocabulary: {orphans}"


def test_the_non_ascii_tokens_are_mapped_in_their_composed_form(macro_map):
    """The 315th-token trap, from the other side. The table was curated from
    vocabulary.json rather than retyped, so its accented keys must be NFKC-composed; a
    decomposed key would look identical in an editor and never match a single track."""
    import unicodedata

    non_ascii = sorted(t for t in macro_map.mapping if not t.isascii())
    assert len(non_ascii) == 14, non_ascii
    for token in non_ascii:
        assert unicodedata.normalize("NFKC", token) == token, repr(token)


# --- the shape of the taxonomy --------------------------------------------------------


def test_family_count_is_in_range(macro_map):
    assert len(macro_map.families) == EXPECTED["families"]
    assert 12 <= len(macro_map.families) <= 15, (
        "P1-11 fixes the range at 12-15, modelled on the Discogs genre/style split"
    )


def test_there_is_no_other_family(macro_map):
    """A catch-all is how a taxonomy rots: it grows, and the one thing the name promises
    -- that someone looked at each row -- silently stops being true."""
    names = {f["id"].lower() for f in macro_map.families}
    names |= {f["name"].lower() for f in macro_map.families}
    for banned in ("other", "misc", "miscellaneous", "unknown", "unreviewed", "world"):
        assert banned not in names, f"{banned!r} is a bucket, not a family"


def test_every_family_is_reachable_and_defined(macro_map):
    used = set(macro_map.mapping.values())
    for family in macro_map.families:
        assert family["id"] in used, f"{family['id']} has no tokens"
        assert family["name"] and family["definition"]


def test_a_token_belongs_to_exactly_one_family(macro_map):
    """Guaranteed by the JSON object shape, asserted anyway: the day the table becomes a
    family->tokens listing, this is the invariant that quietly breaks."""
    assert all(isinstance(v, str) for v in macro_map.mapping.values())
    assert set(macro_map.mapping.values()) <= set(macro_map.family_ids)


# --- the rollups ----------------------------------------------------------------------


def test_macro_primary_agrees_with_the_micro_primary(rolled, macro_map):
    """§8.6 is applied once, not twice. If the macro primary were selected by any other
    rule (most labels in a family, say) a track could read `Rock` while its own primary
    micro label was `rap`, and nothing would report the contradiction."""
    _, tracks, _ = rolled
    for track in tracks:
        if track["genre_status"] == genres.STATUS_UNLABELLED:
            continue
        assert track["genre_macro_primary"] == \
            macro_map.macro(track["genre_micro_primary"])


def test_primary_selection_uses_the_full_precedence_chain(macro_map):
    """The chain lives in genres.primary_sort_key and is reused, not reimplemented. This
    pins the two terms that never fire in Phase 1 but decide a 14-label track: without
    rank-then-alphabetical, the macro primary is whatever dict ordering hands over."""
    def row(label, rank=0, scope=genres.SCOPE_ARTIST, confidence=1.0):
        return {"track_uri": "x", "label": label, "rank": rank, "source": "test",
                "scope": scope, "confidence": confidence, "fetched_at": FETCHED_AT}

    # lowest rank wins, and it is the rank-0 label's family that lands
    primary, _ = genres.select_macros([row("rap", rank=1), row("punk", rank=0)], macro_map)
    assert primary == "punk-metal"
    # alphabetical breaks a rank tie: "punk" < "rap"
    primary, _ = genres.select_macros([row("rap", rank=0), row("punk", rank=0)], macro_map)
    assert primary == "punk-metal"
    # track scope beats artist scope even at a worse rank
    primary, _ = genres.select_macros(
        [row("punk", rank=0), row("rap", rank=3, scope=genres.SCOPE_TRACK)], macro_map
    )
    assert primary == "hip-hop"
    # confidence outranks everything
    primary, _ = genres.select_macros(
        [row("punk", rank=0, confidence=0.5), row("rap", rank=9)], macro_map
    )
    assert primary == "hip-hop"
    assert genres.select_macros([], macro_map) == (None, [])


def test_macro_set_is_ordered_deduplicated_and_headed_by_the_primary(rolled):
    _, tracks, _ = rolled
    for track in tracks:
        macros = track["genre_macro_set"]
        assert list(dict.fromkeys(macros)) == macros, "duplicate family in the set"
        assert track["genre_macro_count"] == len(macros)
        if macros:
            assert macros[0] == track["genre_macro_primary"]
        else:
            assert track["genre_status"] == genres.STATUS_UNLABELLED


def test_multi_macro_tracks_are_preserved(rolled):
    """The figure P1-11 exists to protect. Collapsing a track to one family would discard
    what the source said about it: Latin trap really is Latin AND Hip Hop, and a Sinatra
    Christmas record really is Jazz AND a holiday record.

    177, not the 184 the ticket estimated. That number is a property of a mapping, not of
    the data -- moving `indie` from Rock to Pop alone takes it to 215 -- so it was
    measured after curating rather than curated towards. See docs/decisions/0005.
    """
    _, tracks, _ = rolled
    labelled = [t for t in tracks if t["genre_status"] == genres.STATUS_NATIVE]
    assert len(labelled) == EXPECTED["labelled_tracks"]
    multi = [t for t in labelled if t["genre_macro_count"] > 1]
    assert len(multi) == EXPECTED["multi_macro_tracks"]
    assert max(t["genre_macro_count"] for t in labelled) == \
        EXPECTED["max_macros_on_one_track"]


def test_no_track_is_ever_unreviewed(rolled):
    """The sentinel must not reach a rollup while the table is complete. If it does, the
    map has a colour with no legend entry."""
    _, tracks, _ = rolled
    assert not any(macro.UNREVIEWED in t["genre_macro_set"] for t in tracks)


def test_the_distribution_has_no_vestigial_family(rolled):
    """A family with a handful of tracks is a signal that the taxonomy is wrong, so it is
    measured here rather than left to a reader of the manifest. Reggae at 7 is the floor
    this library supports; anything smaller means merging or dropping a family."""
    _, tracks, _ = rolled
    counts = Counter(t["genre_macro_primary"] for t in tracks
                     if t["genre_macro_primary"] is not None)
    assert counts.most_common(1)[0] == EXPECTED["largest_family_primary"]
    assert min(counts.items(), key=lambda kv: (kv[1], kv[0])) == \
        EXPECTED["smallest_family_primary"]
    assert len(counts) == EXPECTED["families"], "a declared family colours nothing"
    assert sum(counts.values()) == EXPECTED["labelled_tracks"]


# --- the review queue -----------------------------------------------------------------


def test_an_unmapped_token_goes_to_the_queue_and_not_to_a_default(macro_map):
    """The behaviour that keeps the taxonomy honest as tiers 2-4 add labels.

    A new token resolves to the UNREVIEWED sentinel -- which is not a declared family, so
    it has no colour and no legend entry -- and appears in the queue with the counts a
    curator needs. It does not become `Other`.
    """
    assert macro_map.macro("mumble hyperfolk") == macro.UNREVIEWED
    assert macro.UNREVIEWED not in macro_map.family_ids

    vocab = [
        {"label": "rap", "track_count": 137, "first_count": 107},
        {"label": "mumble hyperfolk", "track_count": 2, "first_count": 1},
    ]
    queue = macro.review_queue(vocab, macro_map,
                               examples={"mumble hyperfolk": ["spotify:track:a"] * 5})
    assert queue["n_unreviewed_labels"] == 1
    entry = queue["tokens"][0]
    assert entry["label"] == "mumble hyperfolk"
    assert entry["track_count"] == 2 and entry["first_count"] == 1
    assert entry["example_track_uris"] == ["spotify:track:a"] * 3
    assert [f["id"] for f in queue["families"]] == list(macro_map.family_ids)


def test_the_table_is_rejected_rather_than_half_loaded(tmp_path):
    """Every one of these is invisible downstream: a typo'd family id becomes a colour
    nobody defined, and a family with no tokens becomes a legend entry that never appears.
    Loading is where they are cheap to catch."""
    import json

    good = json.loads(macro.MAP_PATH.read_text())

    def write(doc):
        path = tmp_path / "map.json"
        path.write_text(json.dumps(doc))
        return path

    bad = json.loads(json.dumps(good))
    bad["mapping"]["rap"] = "hiphop"          # typo for "hip-hop"
    with pytest.raises(ValueError, match="undeclared families"):
        macro.load(write(bad))

    bad = json.loads(json.dumps(good))
    bad["families"].append({"id": "ambient", "name": "Ambient", "definition": "x"})
    with pytest.raises(ValueError, match="zero tokens"):
        macro.load(write(bad))

    bad = json.loads(json.dumps(good))
    bad["families"].append({"id": macro.UNREVIEWED, "name": "Unreviewed", "definition": "x"})
    bad["mapping"]["rap"] = macro.UNREVIEWED
    with pytest.raises(ValueError, match="must not be a declared family"):
        macro.load(write(bad))

    bad = json.loads(json.dumps(good))
    bad["notes"]["a token that does not exist"] = "..."
    with pytest.raises(ValueError, match="not mapped"):
        macro.load(write(bad))


# --- the table's committed home -------------------------------------------------------


def test_the_curated_table_is_not_a_build_output():
    """It lives in data/, not artifacts/. artifacts/ is gitignored because everything in
    it is reproducible from a rebuild; this table is an hour of judgement that nothing
    regenerates, and one `git clean` would take it."""
    import subprocess

    assert macro.MAP_PATH.parts[0] == "data"
    assert artifact.ARTIFACTS.name not in macro.MAP_PATH.parts

    ignored = subprocess.run(["git", "check-ignore", str(macro.MAP_PATH)],
                             capture_output=True, text=True)
    assert ignored.returncode != 0, (
        f"{macro.MAP_PATH} is covered by .gitignore -- hand-curated source data cannot "
        "live somewhere a clean checkout does not have it"
    )


def test_the_table_is_an_input_so_recurating_mints_a_new_genre_version():
    """Two artifacts built from the same CSV under different taxonomies must not share an
    id. Without the map in the digest, a recuration would overwrite the meaning of every
    colour while the version string claimed nothing had changed."""
    a, _ = genres.version_id("csv-hash", "macro-hash-1")
    b, _ = genres.version_id("csv-hash", "macro-hash-2")
    assert a.split("-")[-1] != b.split("-")[-1]


def test_the_macro_layer_does_not_drag_in_the_fit_path():
    """The API serves rollups, so this module is in its import graph. Same guard as
    tests/test_genres.py, with a clearer message than test_api_cannot_fit.py's."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import pipeline.macro; print('umap' in sys.modules)"],
        capture_output=True, text=True, cwd=".",
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "pipeline.macro must not import umap"


# --- the built artifact ---------------------------------------------------------------


@pytest.mark.skipif(
    artifact.latest_genre_version() is None,
    reason="no genre artifact built; run python -m pipeline.genres",
)
def test_built_artifact_carries_the_macro_layer():
    version = artifact.latest_genre_version()
    manifest = artifact.load_genre_manifest(version)

    assert manifest["n_macro_families"] == EXPECTED["families"]
    assert manifest["n_mapped_labels"] == EXPECTED["tokens"]
    assert manifest["n_unreviewed_labels"] == 0
    assert manifest["n_tracks_multi_macro"] == EXPECTED["multi_macro_tracks"]
    assert manifest["macro_map"] == str(macro.MAP_PATH)
    assert manifest["macro_map_sha256"] == macro.load().sha256, (
        "the artifact was built from a different macro table than the one on disk -- "
        "rerun python -m pipeline.genres"
    )

    # Ranked by value, not by iteration order: the manifest is written sort_keys=True, so
    # any ordering macro_counts() produced is gone by the time it lands on disk.
    counts = manifest["macro_primary_counts"]
    assert max(counts.items(), key=lambda kv: kv[1]) == EXPECTED["largest_family_primary"]
    assert min(counts.items(), key=lambda kv: kv[1]) == EXPECTED["smallest_family_primary"]
    assert sum(counts.values()) == EXPECTED["labelled_tracks"]
    # in-set counts are >= primary counts by construction, and strictly greater wherever
    # the family shows up as somebody's second label
    for family, n in counts.items():
        assert manifest["macro_set_counts"][family] >= n

    tracks = artifact.load_genre_tracks(version)
    assert all(tuple(t) == genres.ROLLUP_FIELDS for t in tracks)


@pytest.mark.skipif(
    artifact.latest_genre_version() is None,
    reason="no genre artifact built; run python -m pipeline.genres",
)
def test_the_review_queue_is_written_even_when_empty():
    """An empty queue file is a positive statement that the check ran. A missing one is
    indistinguishable from a build that never looked."""
    queue = artifact.load_macro_review(artifact.latest_genre_version())
    assert queue["n_unreviewed_labels"] == 0
    assert queue["tokens"] == []
    assert queue["how_to_clear"], "the queue must say what to do about itself"
