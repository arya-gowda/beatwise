"""P1-12 — the genre palette, recomputed rather than trusted.

Every claim in the comment block above `GENRE` in web/src/map/colour.ts is a MEASUREMENT,
and a measurement written in a comment rots. This module reparses the palette out of the
source and recomputes all of it: pairwise separation in normal vision and under three
colour vision deficiency simulations, distance from the two colours the interaction owns,
contrast against the canvas, and the unlabelled neutral's separation from the twelve.

It caught one already. The pre-P1-12 version of that comment claimed 43 to white and 58 to
the selection cyan; the palettes actually in the file measure 39.4 and 35.2, because dusk
and the truncated viridis landed after the sentence was written. Nothing failed, nothing
looked wrong, and the number a later ticket would have held itself to was fiction.

Everything is measured on the COMPOSITED colour -- 0.85 over #0b0d12, the layer's opacity
under a colour mode -- because that is the colour that reaches the eye. Measuring the raw
hex would tune something nobody sees.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from pipeline import artifact, features, genres, macro

REPO = Path(__file__).resolve().parent.parent
COLOUR_TS = REPO / "web" / "src" / "map" / "colour.ts"
LEGEND_TSX = REPO / "web" / "src" / "map" / "Legend.tsx"
CSV = "Liked_Songs.csv"

# Measured 2026-08-02. Each is a property of the palette in colour.ts, not of the data.
EXPECTED = {
    "families": 12,
    "min_pairwise_normal": 30.0,
    "min_pairwise_tritan": 15.7,
    "min_pairwise_deutan": 7.2,
    "min_pairwise_protan": 6.9,
    "min_to_cyan": 40.6,
    "min_to_white": 42.3,
    "min_to_unlabelled": 43.5,
    "largest_four_apart": 55.6,
    "tail_vs_largest_apart": 33.0,
}

# P1-09's palettes measure this far from the two reserved colours. P1-12 holds the same
# bar; anything closer and a hovered or selected point stops reading as one.
FLOOR_CYAN = 35.2
FLOOR_WHITE = 39.4

# The whole point of the ticket's hardest judgement call, pinned so a recuration that
# quietly grows a tiny family surfaces here.
TAIL = {"reggae": 7, "stage-screen": 12, "folk-country": 15, "holiday-novelty": 23}
LARGEST = {"rock": 320, "hip-hop": 317, "pop": 213, "rnb-soul": 168}

COVERAGE_PCT = "52.41"  # 1,252 of 2,389. Two places, measured, not rounded up to 53.

BG = np.array([11.0, 13.0, 18.0])  # #0b0d12
ALPHA = 0.85


# --- colour maths, self-contained so the assertions do not depend on the code under test -


def _linear(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_WHITE_XYZ = np.array([0.95047, 1.0, 1.08883])


def _lab(rgb):
    """CIELAB of the composited colour -- what is actually on screen."""
    rgb = ALPHA * np.atleast_2d(np.asarray(rgb, float)) + (1 - ALPHA) * BG
    xyz = _linear(rgb) @ _M.T / _WHITE_XYZ
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    return np.stack([116 * f[:, 1] - 16,
                     500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], axis=1)


def _de(a, b):
    return np.linalg.norm(np.atleast_2d(a)[:, None, :] - np.atleast_2d(b)[None, :, :], axis=2)


def _contrast(rgb):
    lin = _linear(ALPHA * np.atleast_2d(np.asarray(rgb, float)) + (1 - ALPHA) * BG)
    lum = lin @ np.array([0.2126, 0.7152, 0.0722])
    bg = float((_linear(BG.reshape(1, 3)) @ np.array([0.2126, 0.7152, 0.0722]))[0])
    return (np.maximum(lum, bg) + 0.05) / (np.minimum(lum, bg) + 0.05)


# Machado, Oliveira & Fernandes 2009, severity 1.0.
_CVD = {
    "protan": np.array([[0.152286, 1.052583, -0.204868],
                        [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968],
                        [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779],
                        [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]]),
}


def _sim_lab(rgb, kind):
    lin = _linear(ALPHA * np.atleast_2d(np.asarray(rgb, float)) + (1 - ALPHA) * BG)
    out = np.clip(lin @ _CVD[kind].T, 0, 1)
    srgb = np.where(out <= 0.0031308, out * 12.92,
                    1.055 * out ** (1 / 2.4) - 0.055) * 255.0
    # Already composited; feed straight through the Lab conversion without compositing
    # twice.
    xyz = _linear(srgb) @ _M.T / _WHITE_XYZ
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    return np.stack([116 * f[:, 1] - 16,
                     500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], axis=1)


def _hex(s):
    return np.array([int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)], float)


# --- parsing the palette out of the source ---------------------------------------------


@pytest.fixture(scope="module")
def source():
    return COLOUR_TS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def palette(source):
    """`const GENRE: Record<string, Rgb> = { ... }` -> {family_id: rgb}."""
    block = re.search(r"const GENRE:[^=]*=\s*\{(.*?)\n\}", source, re.S)
    assert block, "could not find the GENRE palette in colour.ts"
    pairs = re.findall(r"'?([\w-]+)'?:\s*hex\('(#[0-9a-fA-F]{6})'\)", block.group(1))
    assert len(pairs) == EXPECTED["families"], pairs
    return {k: _hex(v) for k, v in pairs}


@pytest.fixture(scope="module")
def reserved(source):
    out = {}
    for name in ("UNLABELLED", "SELECTED", "HOVERED", "DIMMED"):
        # Anchored at the declaration. An unanchored search for SELECTED happily matches
        # inside UNSELECTED, which silently measures the wrong colour and passes.
        m = re.search(
            rf"^(?:export )?const {name}: Rgb = "
            rf"(?:hex\('(#[0-9a-fA-F]{{6}})'\)|\[([\d, ]+)\])",
            source, re.M)
        assert m, f"{name} not found in colour.ts"
        out[name] = _hex(m.group(1)) if m.group(1) else np.array(
            [float(x) for x in m.group(2).split(",")])
    return out


@pytest.fixture(scope="module")
def rolled():
    df, dropped = features.load_features(CSV)
    assert dropped == 0
    rows = genres.parse_labels(df, "2026-08-02T00:00:00+00:00")
    return genres.rollups(df, rows, macro.load())


# --- the palette covers the taxonomy, exactly --------------------------------------------


def test_every_family_has_a_colour_and_nothing_else_does(palette):
    """A family with no colour renders in the light no-data neutral -- visibly wrong rather
    than plausibly wrong -- and a colour with no family is a legend entry that never
    appears. Both are silent in the browser; this is where they are cheap to catch."""
    declared = {f["id"] for f in json.loads(macro.MAP_PATH.read_text())["families"]}
    assert set(palette) == declared, (
        f"colour.ts and {macro.MAP_PATH} disagree: "
        f"only in colour.ts {sorted(set(palette) - declared)}, "
        f"only in the taxonomy {sorted(declared - set(palette))}"
    )


def test_the_unlabelled_key_cannot_collide_with_a_family(source, palette):
    """`unlabelled` must never be able to look like a decision -- same reasoning as
    macro.UNREVIEWED, which is not a declared family either."""
    m = re.search(r"NO_GENRE = '([^']*)'", source)
    assert m, "NO_GENRE not found in colour.ts"
    key = m.group(1)
    assert key not in palette
    assert not re.fullmatch(r"[\w-]+", key), (
        f"{key!r} is shaped like a family id; it must be impossible to collide with one"
    )


# --- separation --------------------------------------------------------------------------


def test_the_twelve_are_separable_in_normal_vision(palette):
    lab = _lab(np.array(list(palette.values())))
    u = np.triu_indices(len(lab), 1)
    worst = float(_de(lab, lab)[u].min())
    assert worst == pytest.approx(EXPECTED["min_pairwise_normal"], abs=0.15)
    # tab10 -- ten categories, the default almost everywhere -- measures 24.1 here.
    assert worst > 24.1, "twelve families should not be tighter than the ten of tab10"


def test_small_dots_do_not_collapse_the_palette(palette):
    """At 2-3px the S-cone contribution collapses for EVERY viewer, not only for the
    0.01% with tritanopia. So a tritan simulation is the closest cheap proxy for what the
    dense core actually looks like, and it is part of the design objective rather than an
    afterthought."""
    rgb = np.array(list(palette.values()))
    lab = _sim_lab(rgb, "tritan")
    u = np.triu_indices(len(rgb), 1)
    assert float(_de(lab, lab)[u].min()) == pytest.approx(
        EXPECTED["min_pairwise_tritan"], abs=0.15)


def test_red_green_deficiency_is_measured_and_not_claimed_to_be_solved(palette):
    """Twelve categorical hues cannot be made red-green safe: a dichromat's colour space is
    essentially two-dimensional and the accepted ceiling is eight to nine (Okabe-Ito is
    eight, Paul Tol stops at nine and says so). This asserts the honest number rather than
    a passing one -- if a future palette claims to have solved it, that claim gets measured
    here. The mitigation is the legend focus control, not the palette."""
    rgb = np.array(list(palette.values()))
    u = np.triu_indices(len(rgb), 1)
    for kind, key in (("deutan", "min_pairwise_deutan"), ("protan", "min_pairwise_protan")):
        lab = _sim_lab(rgb, kind)
        assert float(_de(lab, lab)[u].min()) == pytest.approx(EXPECTED[key], abs=0.15)
    # Still better than tab10, which measures 6.1 / 5.0 with two fewer categories.
    assert EXPECTED["min_pairwise_deutan"] > 6.1
    assert EXPECTED["min_pairwise_protan"] > 5.0


def test_the_interaction_colours_stay_reachable(palette, reserved):
    """The map has to keep saying "hovered" and "selected" while a colour mode is on. Both
    floors are P1-09's own measured minimums across every ramp in the file."""
    lab = _lab(np.array(list(palette.values())))
    to_cyan = float(_de(lab, _lab(reserved["SELECTED"])).min())
    to_white = float(_de(lab, _lab(reserved["HOVERED"])).min())
    assert to_cyan == pytest.approx(EXPECTED["min_to_cyan"], abs=0.15)
    assert to_white == pytest.approx(EXPECTED["min_to_white"], abs=0.15)
    assert to_cyan >= FLOOR_CYAN, "a genre colour is closer to the selection cyan than any ramp stop"
    assert to_white >= FLOOR_WHITE, "a genre colour is closer to the hover white than any ramp stop"


def test_the_headroom_goes_where_the_tracks_are(palette, rolled):
    """Assignment is not arbitrary. How often two families meet on screen goes with
    n_i * n_j; what a confusion costs per track goes with 1/n_i + 1/n_j; multiply and the
    weight is n_i + n_j. The floor above is unweighted so no two swatches can collapse, but
    above the floor the distance goes to the pairs that matter: the four largest families
    sit far apart from each other, and every tiny family sits far from every large one."""
    from collections import Counter

    counts = Counter(t["genre_macro_primary"] for t in rolled
                     if t["genre_macro_primary"])
    assert {k: counts[k] for k in LARGEST} == LARGEST
    assert {k: counts[k] for k in TAIL} == TAIL

    ids = list(palette)
    lab = _lab(np.array(list(palette.values())))
    d = _de(lab, lab)
    idx = {k: ids.index(k) for k in ids}

    among_largest = min(d[idx[a], idx[b]] for a in LARGEST for b in LARGEST if a < b)
    tail_vs_largest = min(d[idx[a], idx[b]] for a in TAIL for b in LARGEST)
    assert among_largest == pytest.approx(EXPECTED["largest_four_apart"], abs=0.2)
    assert tail_vs_largest == pytest.approx(EXPECTED["tail_vs_largest_apart"], abs=0.2)
    assert among_largest > EXPECTED["min_pairwise_normal"] * 1.5


# --- absence reads as absence ------------------------------------------------------------


def test_the_unlabelled_neutral_is_outside_the_categorical_set(palette, reserved):
    """Not a thirteenth genre: no chroma to speak of, further from every family colour than
    any two families are from each other, and dimmer than all of them."""
    lab = _lab(np.array(list(palette.values())))
    neutral = _lab(reserved["UNLABELLED"])
    to_families = float(_de(lab, neutral).min())
    assert to_families == pytest.approx(EXPECTED["min_to_unlabelled"], abs=0.2)
    assert to_families > EXPECTED["min_pairwise_normal"], (
        "the neutral must sit further from the families than they sit from each other, or "
        "it reads as one of them"
    )
    # Near-grey rather than achromatic. The canvas is a cool #0b0d12 and the dim grey the
    # interaction already uses is cool too; a literally neutral grey reads warm against
    # both. Chroma under ~12 in CIELAB does not register as a hue at 2-3px.
    chroma = float(np.hypot(neutral[0, 1], neutral[0, 2]))
    assert chroma < 12, f"the unlabelled neutral has chroma {chroma:.1f}; it must read as grey"
    assert _contrast(reserved["UNLABELLED"])[0] < _contrast(
        np.array(list(palette.values()))).min(), (
        "the neutral must be the dimmest thing on the map -- 1,137 points is nearly half "
        "the library, and at the light no-data grey it would be the loudest"
    )


def test_the_unlabelled_neutral_is_not_the_no_data_grey(source):
    """Two different absences, deliberately two colours. A handful of missing years is an
    exception worth spotting and gets the light neutral; 47.6% of the library missing a
    genre is the ground, and at that lightness it would drown the twelve."""
    no_data = re.search(r"const NO_DATA: Rgb = \[([\d, ]+)\]", source)
    assert no_data, "NO_DATA not found"
    light = np.array([float(x) for x in no_data.group(1).split(",")])
    unlabelled = re.search(r"UNLABELLED: Rgb = hex\('(#[0-9a-fA-F]{6})'\)", source)
    assert unlabelled
    assert float(_de(_lab(light), _lab(_hex(unlabelled.group(1))))[0, 0]) > 20


def test_the_neutral_cannot_be_mistaken_for_a_selected_point(reserved):
    neutral = _lab(reserved["UNLABELLED"])
    assert float(_de(neutral, _lab(reserved["SELECTED"]))[0, 0]) >= FLOOR_CYAN
    assert float(_de(neutral, _lab(reserved["HOVERED"]))[0, 0]) >= FLOOR_WHITE


# --- the map still survives a selection --------------------------------------------------


def test_genre_survives_dimming_and_lifting(palette, reserved):
    """P1-09's contract: dimming CONTRACTS toward grey rather than replacing hue, and a
    selected point keeps its own colour with a cyan ring around it. Under genre that has to
    keep meaning something -- lasso a region and the families around it must not go blank,
    and a selected point must still say which family it is."""
    rgb = np.array(list(palette.values()))
    u = np.triu_indices(len(rgb), 1)

    dimmed = rgb + (reserved["DIMMED"] - rgb) * 0.62   # DIM_STRENGTH
    lab = _lab(dimmed)
    worst = float(_de(lab, lab)[u].min())
    assert worst > 8.0, (
        f"dimmed families collapse to {worst:.1f} -- under a selection the map would go "
        "monochrome, which is the failure DIM_STRENGTH exists to avoid"
    )

    lifted = rgb + (reserved["HOVERED"] - rgb) * 0.25  # LIFT_STRENGTH
    lab = _lab(lifted)
    assert float(_de(lab, lab)[u].min()) > 15.0, "selected points lose their family colour"
    assert float(_de(lab, _lab(reserved["HOVERED"])).min()) >= 25.0, (
        "a selected point is drifting into the hover white"
    )


# --- what the legend says ----------------------------------------------------------------


def test_the_coverage_figure_is_the_measured_one(rolled):
    """52.41%, not 52% and not "over half". The legend computes it in the browser from the
    points actually drawn rather than reading the manifest, so a genre artifact built from
    a different export shows up as a disagreement instead of as a number quietly about a
    different set of tracks."""
    labelled = sum(1 for t in rolled if t["genre_macro_primary"])
    assert labelled == 1252
    assert len(rolled) == 2389
    assert f"{labelled / len(rolled) * 100:.2f}" == COVERAGE_PCT


def test_the_legend_states_counts_rather_than_hardcoding_them(source):
    """Every figure in the key comes from the artifact. A count typed into the source is a
    count that silently stops being true on the next genre build."""
    block = re.search(r"function genreScale.*?\n\}", source, re.S)
    assert block
    for literal in ("1137", "1,137", "1252", "1,252", "52.4", "2389", "2,389"):
        assert literal not in block.group(0), (
            f"{literal!r} is hardcoded in genreScale; it must be measured from the points"
        )


def test_every_key_is_a_control(source):
    """The tail is thin: four families are under 2% of the labelled tracks and reggae is
    seven tracks. No palette makes seven dots findable among 2,389, so the legend rows are
    buttons that isolate a family. That is also the only accommodation available for
    red-green colour vision deficiency at twelve categories."""
    legend = LEGEND_TSX.read_text(encoding="utf-8")
    assert "aria-pressed" in legend and "onPointerEnter" in legend
    assert "matches" in source, "the scale needs a focus predicate"
    # Focus matches the macro SET, not just the primary -- 42 tracks are coloured jazz but
    # 75 carry it, and the legend row shows both.
    assert "genre_macro_set.includes" in source


def test_the_tail_is_smaller_than_the_ticket_said(rolled):
    """Documenting a real discrepancy rather than smoothing it. P1-12's carried note names
    four families under 2% of the labelled tracks. 2% of 1,252 is 25.04, so Punk & Metal at
    24 is under it too -- five families, not four. It changes nothing about the palette;
    it changes what "the tail" means when someone reads the legend."""
    from collections import Counter

    counts = Counter(t["genre_macro_primary"] for t in rolled if t["genre_macro_primary"])
    labelled = sum(counts.values())
    under_two_pct = {k for k, n in counts.items() if n < 0.02 * labelled}
    assert under_two_pct == set(TAIL) | {"punk-metal"}
    assert counts["punk-metal"] == 24 < 0.02 * labelled


# --- the built artifact agrees -----------------------------------------------------------


@pytest.mark.skipif(
    artifact.latest_genre_version() is None,
    reason="no genre artifact built; run python -m pipeline.genres",
)
def test_the_endpoint_carries_everything_the_legend_needs():
    """The web app reads the manifest for family names, counts and coverage, and never
    retypes any of them. If a field disappears the legend degrades silently, so the
    contract is asserted here."""
    manifest = artifact.load_genre_manifest(artifact.latest_genre_version())
    for field in ("macro_families", "macro_primary_counts", "macro_set_counts",
                  "coverage", "n_tracks", "n_labelled_tracks", "n_unlabelled_tracks",
                  "n_tracks_multi_macro"):
        assert field in manifest, f"the legend needs manifest.{field}"
    assert [f["id"] for f in manifest["macro_families"]]
    assert all("name" in f for f in manifest["macro_families"])
    assert f"{manifest['coverage'] * 100:.2f}" == COVERAGE_PCT
