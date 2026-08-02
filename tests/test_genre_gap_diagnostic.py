"""P1-13's diagnostic, tested where it can be tested without spending an API call.

The network half is not mocked into a fake pass — a stubbed Spotify would only prove the
stub agrees with itself. What is pinned here is everything that decides WHICH question the
run answers: the population it samples from, the determinism of that sample, the
three-way classification that keeps "we could not find them" out of the "Spotify has no
genres" bucket, and the credential handling that must never touch a VITE_ name.
"""

import subprocess

import pytest

from diagnostics import genre_gap_at_source as gap

CSV = "Liked_Songs.csv"

# Measured 2026-08-02 against Liked_Songs.csv, alongside tests/test_genres.py's figures.
EXPECTED = {
    "n_tracks": 2389,
    "n_empty_genre_rows": 1137,     # matches test_genres.py's unlabelled_tracks
    "n_artists_total": 1495,
    "n_artists_on_an_empty_row": 718,
    "n_artists_on_both": 83,
    "n_population": 635,
}


@pytest.fixture(scope="module")
def population():
    return gap.empty_artists(CSV)


# --- the population the sample is drawn from ------------------------------------------


def test_population_figures(population):
    _, counts = population
    assert counts == {**counts, **EXPECTED}, (
        "the empty-artist population moved. Either the CSV changed or the artist/genre "
        "splitting did; find out which before re-pinning these."
    )


def test_ambiguous_artists_are_excluded_not_guessed(population):
    """718 artists sit on at least one empty row; 83 of those also sit on a labelled one.

    Those 83 are excluded. Their labels may belong entirely to a collaborator, so calling
    them "an artist Spotify has no genres for" would be a guess dressed as a measurement.
    """
    names, counts = population
    assert counts["n_artists_on_an_empty_row"] - counts["n_artists_on_both"] == len(names)


def test_population_uses_the_repo_artist_splitter():
    """Semicolons, and commas that live INSIDE one name. There is one splitter for this
    column and it is ml.data.split_artists — see commit 5533a94 for what a second one
    cost last time."""
    assert gap.split_artists("Consequence;Kanye West") == ["Consequence", "Kanye West"]
    assert gap.split_artists("Tyler, The Creator") == ["Tyler, The Creator"]


# --- sampling -------------------------------------------------------------------------


def test_sample_is_deterministic_and_the_right_size(population):
    names, _ = population
    first = gap.sample(names, size=100, seed=gap.SEED)
    second = gap.sample(names, size=100, seed=gap.SEED)
    assert first == second
    assert len(first) == 100
    assert len(set(first)) == 100
    assert set(first) <= set(names)


def test_a_different_seed_gives_a_different_sample(population):
    names, _ = population
    assert gap.sample(names, 100, gap.SEED) != gap.sample(names, 100, gap.SEED + 1)


def test_the_sample_is_a_sample(population):
    """100, not 660. The ticket is one cheap run, not a full sweep."""
    names, _ = population
    assert gap.SAMPLE_SIZE == 100 < len(names)


# --- resolution: the distinction the whole ticket rests on ----------------------------


def test_no_search_hit_is_not_evidence_of_an_empty_genre_list():
    assert gap.resolve("Nobody", [])["outcome"] == gap.NOT_FOUND


def test_a_near_miss_is_a_resolution_failure_not_a_match():
    """Search will happily return a plausible neighbour. Accepting it would file a
    different artist's empty genre list under our artist's name."""
    result = gap.resolve("Bad Bunny", [{"id": "x", "name": "Bad Bunny Tribute Band",
                                        "genres": []}])
    assert result["outcome"] == gap.NO_EXACT_MATCH
    assert "artist_id" not in result


def test_an_exact_match_resolves_case_and_accent_insensitively():
    result = gap.resolve("beyoncé", [{"id": "b", "name": "BEYONCÉ",
                                            "genres": ["pop"]}])
    assert result["outcome"] == gap.RESOLVED
    assert result["artist_id"] == "b"
    assert result["ambiguous"] is False


def test_same_named_artists_are_recorded_as_ambiguous():
    result = gap.resolve("Nadia", [
        {"id": "1", "name": "Nadia", "genres": []},
        {"id": "2", "name": "Nadia", "genres": ["jazz"]},
    ])
    assert result["outcome"] == gap.RESOLVED
    assert result["artist_id"] == "1"       # search relevance order
    assert result["ambiguous"] is True
    assert result["other_exact_matches_with_genres"] == 1


# --- the tally ------------------------------------------------------------------------


def _record(outcome, has_genres=None):
    """`has_genres=None` on a RESOLVED record means Spotify sent no `genres` key."""
    return {"csv_name": "x", "outcome": outcome, "has_genres": has_genres,
            "genres_field_present": has_genres is not None}


def test_failures_and_empties_are_counted_separately():
    counts = gap.tally([
        _record(gap.NOT_FOUND),
        _record(gap.NO_EXACT_MATCH),
        _record(gap.RESOLVED, False),
        _record(gap.RESOLVED, True),
    ])
    assert counts["resolution_failures"] == 2
    assert counts["resolved"] == 2
    assert counts["resolved_empty"] == 1
    assert counts["resolved_with_genres"] == 1
    assert counts["resolved_field_absent"] == 0


def test_an_absent_field_is_not_counted_as_an_empty_one():
    """The mistake that would have turned this ticket into a false confirmation of §8."""
    counts = gap.tally([_record(gap.RESOLVED, None)] * 10)
    assert counts["resolved"] == 10
    assert counts["resolved_field_absent"] == 10
    assert counts["resolved_empty"] == 0
    assert counts["genres_field_present"] == 0


def test_a_missing_key_never_becomes_an_empty_list():
    assert gap._field({"genres": ["pop"]}, "genres") == ["pop"]
    assert gap._field({"genres": []}, "genres") == []
    assert gap._field({}, "genres") is None
    assert gap.resolve("A", [{"id": "1", "name": "A"}])["search_genres"] is None
    assert gap.resolve("A", [{"id": "1", "name": "A", "genres": []}])["search_genres"] == []


def test_verdict_thresholds():
    assert gap.verdict(gap.tally([_record(gap.NOT_FOUND)]))[0] == "inconclusive"
    assert gap.verdict(gap.tally([_record(gap.RESOLVED, False)]))[0] == "gap_is_real"

    handful = [_record(gap.RESOLVED, True)] * gap.HANDFUL
    assert gap.verdict(gap.tally(handful))[0] == "gap_is_real"
    assert gap.verdict(gap.tally(handful + [_record(gap.RESOLVED, True)]))[0] == \
        "diagnosis_wrong"


# --- the positive control -------------------------------------------------------------


def test_control_population_is_disjoint_from_the_sample(population):
    """The control must be artists the CSV DOES attribute genres to, credited alone so
    the labels cannot belong to a collaborator."""
    names, _ = population
    controls = gap.control_artists(CSV)
    assert controls
    assert not set(controls) & set(names)


def test_an_empty_field_for_everyone_voids_the_run():
    """`genres` is a deprecated field. "These 100 artists have none" and "the field
    returns nothing for anybody" are the same measurement without this check — and the
    second one would read as a clean confirmation of §8.
    """
    dark = gap.tally([_record(gap.RESOLVED, False)] * 10)
    real = gap.tally([_record(gap.RESOLVED, False)] * 100)
    assert gap.control_holds(dark) is False
    assert gap.verdict(real, dark)[0] == "method_void"


def test_an_absent_field_is_reported_as_removed_not_as_a_confirmation():
    """What this run actually found on 2026-08-02. The verdict must name the cause."""
    gone = gap.tally([_record(gap.RESOLVED, None)] * 100)
    flag, text = gap.verdict(gone, gap.tally([_record(gap.RESOLVED, None)] * 10))
    assert flag == "field_removed"
    assert "IMPOSSIBLE" in text


def test_a_healthy_control_lets_the_verdict_stand():
    healthy = gap.tally([_record(gap.RESOLVED, True)] * 10)
    real = gap.tally([_record(gap.RESOLVED, False)] * 100)
    assert gap.control_holds(healthy) is True
    assert gap.verdict(real, healthy)[0] == "gap_is_real"


def test_a_control_that_never_resolved_is_not_a_pass():
    assert gap.control_holds(gap.tally([_record(gap.NOT_FOUND)] * 10)) is False
    assert gap.control_holds(None) is False


# --- credentials ----------------------------------------------------------------------


def test_the_env_file_is_gitignored_and_untracked():
    """The secret's only protection. Checked against git, not against the file's text."""
    ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], capture_output=True)
    assert ignored.returncode == 0, ".env is NOT gitignored — a client secret could land in a commit"

    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", ".env"],
                             capture_output=True)
    assert tracked.returncode != 0, ".env is tracked by git"


def test_vite_prefixed_names_are_not_readable(tmp_path):
    """Vite inlines VITE_* into the browser bundle at build time. This reader is a
    whitelist, so no edit to .env can route a secret into one — the PKCE flow in
    docs/decisions/0003-spotify-auth.md depends on the browser having no secret."""
    env = tmp_path / ".env"
    env.write_text(
        "VITE_SPOTIFY_CLIENT_SECRET=leaked\n"
        "VITE_SPOTIFY_CLIENT_ID=public\n"
        "SPOTIFY_CLIENT_ID=id\n"
        "SPOTIFY_CLIENT_SECRET='quoted-secret'\n"
        "# comment\n"
        "\n",
        encoding="utf-8",
    )
    values = gap.read_env_file(env)
    assert values == {"SPOTIFY_CLIENT_ID": "id", "SPOTIFY_CLIENT_SECRET": "quoted-secret"}


def test_missing_credentials_fail_legibly(tmp_path):
    with pytest.raises(gap.MissingCredentials) as excinfo:
        gap.load_credentials(tmp_path / "absent.env", environ={})
    message = str(excinfo.value)
    assert "SPOTIFY_CLIENT_ID" in message and "SPOTIFY_CLIENT_SECRET" in message
    assert "absent.env" in message
    assert "dashboard" in message


def test_partial_credentials_do_not_half_run(tmp_path):
    """An id with no secret must stop, not fall back to the browser's public client id."""
    env = tmp_path / ".env"
    env.write_text("SPOTIFY_CLIENT_ID=id\n", encoding="utf-8")
    with pytest.raises(gap.MissingCredentials) as excinfo:
        gap.load_credentials(env, environ={})
    assert "SPOTIFY_CLIENT_SECRET" in str(excinfo.value)


# --- endpoints ------------------------------------------------------------------------


def test_the_diagnostic_names_no_withdrawn_endpoint():
    """Same guarantee tests/test_no_deprecated_spotify_endpoints.py gives web/src, for
    the one piece of Python that talks to Spotify. Also blocks the batch artist call,
    whose reference page gained an endpoint-level Deprecated tag on 2026-08-02."""
    source = (gap.REPO / "diagnostics" / "genre_gap_at_source.py").read_text("utf-8")
    withdrawn = ("audio-" "features", "audio-" "analysis", "recommend" "ations",
                 "related-" "artists", "featured-" "playlists")
    named = [name for name in withdrawn if name in source]
    assert not named, f"the diagnostic names withdrawn endpoints: {named}"

    # Matched on the URL-BUILDING shape, not on the bare path: the docstring names the
    # batch call in order to explain why it is not used, and a check that forbade saying
    # so would delete the explanation rather than the mistake.
    built = "{API}/artists"
    assert built + "/" in source, "the per-artist call is how genres are read"
    assert built + "?" not in source, (
        "the batch artist endpoint gained an endpoint-level Deprecated tag on "
        "2026-08-02; build GET /v1/artists/{id} instead"
    )


def test_search_limit_respects_the_documented_maximum():
    """Verified on the live reference 2026-08-02: max 10, default 5. Every tutorial and
    most memories say 50, and 50 is a 400."""
    assert gap.SEARCH_LIMIT <= 10


def test_the_diagnostic_is_not_wired_into_the_app():
    """A diagnostic that becomes a pipeline stage is how live-layer data reaches the
    substrate. Nothing that builds or serves the map may import this."""
    for package in ("pipeline", "api"):
        for path in sorted((gap.REPO / package).rglob("*.py")):
            assert "diagnostics" not in path.read_text("utf-8"), (
                f"{path.relative_to(gap.REPO)} references diagnostics/"
            )
