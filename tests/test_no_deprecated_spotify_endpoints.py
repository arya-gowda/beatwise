"""Beatwise must never call Spotify's withdrawn endpoints.

`audio-features`, `audio-analysis` and `recommendations` were withdrawn for applications
registered on or after 2024-11-27. Beatwise is a new application, so they are not
"discouraged" -- they are gone, and the whole reason the substrate is an uploaded CSV
(docs/decisions/0001-stack.md, concept doc §5).

Enforced here rather than trusted, for the same reason test_api_cannot_fit.py exists.
Two of these names describe things a future contributor will genuinely want -- "just
fetch the audio features for the new tracks", "just ask for recommendations near this
region" -- and either would pull live-layer data into the decision of where a point sits.
That is precisely the boundary the two-layer model exists to hold. A guardrail that fails
in CI is a better guard than a paragraph someone has to have read.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEB_SRC = REPO / "web" / "src"

FORBIDDEN = ("audio-features", "audio-analysis", "recommendations")

# The single file allowed to name them: it exists to declare and block them.
DECLARATION = WEB_SRC / "spotify" / "deprecated.ts"

# The single file allowed to name the API host. Any second one would be a door around
# the guard.
API_HOST = "api.spotify.com"
API_HOST_HOME = WEB_SRC / "spotify" / "config.ts"

SOURCE_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".css"}


def _sources():
    return sorted(
        p
        for p in WEB_SRC.rglob("*")
        if p.is_file() and p.suffix in SOURCE_SUFFIXES
    )


def test_web_source_exists():
    """Guard the guard: an empty scan must not read as a pass."""
    assert WEB_SRC.is_dir(), f"expected {WEB_SRC} to exist"
    assert _sources(), "found no web sources to scan -- the check below would be vacuous"


def test_deprecated_endpoints_are_not_referenced():
    offenders = []
    for path in _sources():
        if path == DECLARATION:
            continue
        text = path.read_text(encoding="utf-8")
        for name in FORBIDDEN:
            if name in text:
                offenders.append(f"{path.relative_to(REPO)} mentions {name!r}")

    assert not offenders, (
        "withdrawn Spotify endpoints referenced outside the file that blocks them:\n  "
        + "\n  ".join(offenders)
        + "\n\nThese endpoints are unavailable to apps registered after 2024-11-27, and "
        "audio features must come from the substrate rather than the live layer. "
        "See docs/decisions/0003-spotify-auth.md."
    )


def test_the_blocklist_still_blocks_everything():
    """Deleting a name from the blocklist would silently disarm the check above."""
    assert DECLARATION.is_file(), f"{DECLARATION.relative_to(REPO)} is missing"
    text = DECLARATION.read_text(encoding="utf-8")
    missing = [name for name in FORBIDDEN if f"'{name}'" not in text]
    assert not missing, f"{DECLARATION.relative_to(REPO)} no longer blocks {missing}"


def test_there_is_exactly_one_door_to_the_spotify_api():
    """The runtime guard only guarantees anything if every call goes through it.

    `spotifyFetch` calls `assertNotDeprecated` before touching the network, so the check
    is only as good as the claim that nothing else builds a Spotify API request. The base
    URL living in one place is what makes that claim checkable.
    """
    client = WEB_SRC / "spotify" / "client.ts"
    assert "assertNotDeprecated(" in client.read_text(encoding="utf-8"), (
        "spotifyFetch must call assertNotDeprecated before issuing a request"
    )

    elsewhere = [
        str(p.relative_to(REPO))
        for p in _sources()
        if p != API_HOST_HOME and API_HOST in p.read_text(encoding="utf-8")
    ]
    assert not elsewhere, (
        f"{API_HOST} is named outside {API_HOST_HOME.relative_to(REPO)}: {elsewhere}. "
        "Every Spotify request goes through spotifyFetch in web/src/spotify/client.ts, "
        "which is where the deprecated-endpoint guard lives."
    )


# --- February 2026 replacements -------------------------------------------------------
#
# A different failure from the withdrawals above. These endpoints have a working successor
# one word away, so the mistake is not reaching for something gone -- it is writing the
# spelling a decade of tutorials teach and finding out at runtime. Playlist write-back
# (P1-07) rides on exactly these two paths.
#
# Verified against the live reference pages on 2026-08-02, not the changelog alone:
#   - "Add Items to Playlist" documents POST /playlists/{playlist_id}/items, no banner.
#   - The /tracks page is titled "Add Items to Playlist [DEPRECATED]" and says
#     "Deprecated: Use Add Items to Playlist instead."
#   - "Create Playlist" documents POST /me/playlists; the changelog records
#     POST /users/{user_id}/playlists as removed in its favour.

WRITE_BACK = WEB_SRC / "spotify" / "playlists.ts"

# Both tolerate a template interpolation in the id position, which is how a real call is
# written. The `(?!\{)` skips `{playlist_id}` / `{user_id}` -- documentation placeholders,
# which the comments above and in playlists.ts legitimately contain, and which no fetch
# could ever resolve. This catches the shape a mistake actually takes; `assertNotDeprecated`
# in web/src/spotify/deprecated.ts is the backstop for anything assembled at runtime.
ADD_TO_PLAYLIST_OLD = re.compile(r"/playlists/(?!\{)[^\s\"'`]*/tracks")
CREATE_PLAYLIST_OLD = re.compile(r"/users/(?!\{)[^\s\"'`]*/playlists")


def test_playlist_write_back_uses_the_items_endpoint():
    """POST /playlists/{id}/items, not the deprecated /tracks."""
    source = WRITE_BACK.read_text(encoding="utf-8")
    assert "/items`" in source, (
        f"{WRITE_BACK.relative_to(REPO)} does not build a /items path -- playlist "
        "write-back must POST /playlists/{playlist_id}/items"
    )
    assert "'/me/playlists'" in source, (
        f"{WRITE_BACK.relative_to(REPO)} does not POST /me/playlists to create"
    )


def test_replaced_playlist_endpoints_are_not_referenced():
    offenders = []
    for path in _sources():
        if path == DECLARATION:  # it exists to name and block them
            continue
        text = path.read_text(encoding="utf-8")
        if ADD_TO_PLAYLIST_OLD.search(text):
            offenders.append(f"{path.relative_to(REPO)} builds /playlists/.../tracks")
        if CREATE_PLAYLIST_OLD.search(text):
            offenders.append(f"{path.relative_to(REPO)} builds /users/.../playlists")

    assert not offenders, (
        "Spotify replaced these in February 2026:\n  "
        + "\n  ".join(offenders)
        + "\n\nUse POST /playlists/{playlist_id}/items and POST /me/playlists."
    )


def test_the_replacement_guard_still_blocks_both():
    """Deleting an entry from REPLACED_PATHS would silently disarm the runtime guard."""
    text = DECLARATION.read_text(encoding="utf-8")
    assert "REPLACED_PATHS" in text, f"{DECLARATION.relative_to(REPO)} lost REPLACED_PATHS"
    for expected in (r"\/playlists\/[^/?#]+\/tracks", r"\/users\/[^/?#]+\/playlists"):
        assert expected in text, (
            f"{DECLARATION.relative_to(REPO)} no longer blocks {expected!r} at runtime"
        )
