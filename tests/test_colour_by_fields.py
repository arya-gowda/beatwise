"""The display fields P1-09 colours by — the shape of the data, asserted not assumed.

Colour-by reads five columns the substrate carries for display only. None of them moves a
point; all of them decide what the map means once a mode is on. Three facts about their
shape drove design decisions in web/src/map/colour.ts, and each would fail silently rather
than loudly if a future export changed it:

  * `Release Date` arrives at three precisions. The year mode reads the leading four
    digits so a bare `1972` and a `1972-06-01` land on the same colour. If a future export
    were full-precision throughout, that decision would look like pointless defensiveness
    rather than the load-bearing thing it is.
  * `Added By` is entirely null, which is WHY it is not one of the five modes. Without a
    test, that absence is indistinguishable from an oversight.
  * `Popularity` is nullable in the artifact schema even though this export has no nulls,
    and colour.ts renders a null neutral rather than as zero.

If one of these figures changes, the question is which of the file and the expectation
moved — not which constant to edit.
"""

import re

import pandas as pd
import pytest

from pipeline import features

CSV = "Liked_Songs.csv"

# Measured 2026-08-02 against Liked_Songs.csv (2,389 rows, zero dropped).
EXPECTED = {
    "rows": 2389,
    "release_full": 2253,      # YYYY-MM-DD
    "release_year_month": 11,  # YYYY-MM
    "release_year_only": 125,  # YYYY
    "explicit_true": 662,
    "popularity_null": 0,
}


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV)


def precision(value):
    s = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return "full"
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return "year_month"
    if re.fullmatch(r"\d{4}", s):
        return "year_only"
    return "other"


def test_release_date_arrives_at_three_precisions(df):
    counts = df["Release Date"].map(precision).value_counts().to_dict()
    assert counts.get("other", 0) == 0, "an unrecognised release-date shape appeared"
    assert counts["full"] == EXPECTED["release_full"]
    assert counts["year_month"] == EXPECTED["release_year_month"]
    assert counts["year_only"] == EXPECTED["release_year_only"]
    assert sum(counts.values()) == EXPECTED["rows"]


def test_every_release_date_yields_a_year(df):
    """No track can fall off the year ramp. A missing year would have to render as
    'no data', and 'no data' on a field this complete would mean something broke."""
    years = df["Release Date"].astype(str).str.extract(r"^(\d{4})")[0]
    assert years.notna().all()
    assert years.astype(int).between(1900, 2100).all()


def test_added_by_is_entirely_null(df):
    """The reason `added by` is not a colour mode. `Added At` — the `added` mode — is a
    different column and is fully populated."""
    assert df["Added By"].notna().sum() == 0
    assert df["Added At"].notna().sum() == EXPECTED["rows"]


def test_added_at_parses_as_an_instant(df):
    added = pd.to_datetime(df["Added At"], format="ISO8601", utc=True)
    assert added.notna().all()


def test_popularity_is_present_but_the_schema_allows_null(df):
    """Zero nulls today. The artifact still types popularity as nullable and the map still
    renders a null neutral — colouring an absent popularity as 0 would be a lie about a
    track, and 0 is a real value here (249 tracks have it)."""
    assert df["Popularity"].isna().sum() == EXPECTED["popularity_null"]
    assert (df["Popularity"] == 0).sum() > 0
    assert "Popularity" not in features.FEATURES, "popularity must never reach the embedding"


def test_explicit_is_boolean_and_split(df):
    """Two clearly distinct colours, not a ramp — and both categories are populated
    enough to be worth a legend entry."""
    assert set(df["Explicit"].unique()) <= {True, False}
    assert df["Explicit"].sum() == EXPECTED["explicit_true"]


def test_colour_by_fields_are_all_carried_but_none_are_features(df):
    """The two-layer rule at its narrowest: everything colour-by reads is carried through
    for display, and nothing colour-by reads except tempo positions a point."""
    for column in ["Popularity", "Release Date", "Added At", "Explicit"]:
        assert column in features.CARRY
        assert column not in features.FEATURES
    # Tempo is the one exception, and it is deliberate: it is both a display field and one
    # of the ten features. The legend says so.
    assert "Tempo" in features.FEATURES and "Tempo" in features.CARRY
