"""Feature loading — the one boundary between a source of audio features and the map.

There is exactly one implementation (a Spotify CSV export). This is a function, not a
plugin framework: the concept doc's "feature provider interface" is about keeping the
seam visible so a second source can be added without touching the embedding, not about
building extensibility nobody needs yet.
"""

import hashlib

import pandas as pd

# The ten features the prototype feeds to UMAP, in this order. Order is load-bearing:
# a reducer fitted on a different column order is silently wrong, not broken, so the
# manifest records this list and the projection path asserts against it.
FEATURES = [
    "Danceability", "Energy", "Loudness", "Speechiness", "Acousticness",
    "Instrumentalness", "Liveness", "Valence", "Tempo", "Mode",
]

# Key is deliberately NOT in the embedding. See docs/decisions/0002-key-encoding.md.
KEY_ENCODING = "dropped"

# Carried through for display and colour-by, never for positioning. Popularity in
# particular is excluded from FEATURES on purpose -- position means sound, and
# popularity stays an independent lens (concept doc, Context).
CARRY = [
    "Track URI", "Track Name", "Artist Name(s)", "Album Name",
    "Popularity", "Release Date", "Added At", "Tempo", "Explicit", "Genres",
]


def source_hash(path):
    """SHA-256 of the source file, recorded in the manifest so an artifact can always
    be traced back to the exact bytes it was built from."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_features(source):
    """Read a Spotify CSV export into (kept_rows, n_dropped).

    pandas strips the UTF-8 BOM this file carries on its first header via encoding
    sniffing, so 'Track URI' arrives clean. That is a property of reading it here in
    Python; a JS parser would produce a mangled first column name.
    """
    df = pd.read_csv(source)

    before = len(df)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    dropped = before - len(df)

    # Guard the BOM explicitly rather than trusting the sniff -- a mangled key here
    # would propagate into the artifact and only surface in the browser.
    if df.columns[0] != "Track URI":
        raise ValueError(
            f"first column is {df.columns[0]!r}, expected 'Track URI' -- "
            "the BOM was not stripped"
        )

    df["Artist Name(s)"] = df["Artist Name(s)"].fillna("").astype(str)
    df["Genres"] = df["Genres"].fillna("").astype(str)
    return df, dropped
