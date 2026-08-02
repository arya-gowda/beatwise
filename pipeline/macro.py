"""Micro -> macro genre families: loading and validating the hand-curated table (P1-11).

The table lives at `data/genre_macro_map.json` — under `data/`, NOT under `artifacts/`.
`artifacts/` is gitignored because everything in it is a build output that a rebuild
reproduces. This table is the opposite: hand-curated source data, an hour of judgement
per revision, and nothing regenerates it. In `artifacts/` it would be one `git clean`
from gone. It is an INPUT to the genre build, which is why its sha256 is folded into the
genre version digest — recurating is a new genre version, exactly as a changed CSV is.

There is no `Other` family and no default. Every one of the 314 tokens in the P1-10
vocabulary has an explicit row, so every macro label can be explained by pointing at one.
A token that is not in the table resolves to UNREVIEWED and lands in the review queue the
build writes; it does not quietly join a bucket that looks like a decision. A silent
`Other` is how a taxonomy rots: it grows, nobody can say why anything is in it, and the
one thing the family name promises — that someone looked — stops being true.

This module deliberately does not import `genres`: `genres` imports it. The §8.6
precedence chain lives in `genres.primary_sort_key` and is reused there, not reimplemented
here.
"""

import hashlib
import json
from pathlib import Path

MAP_PATH = Path("data") / "genre_macro_map.json"

# The macro of a token the curation pass has not seen. NOT a family: it is absent from
# `families`, so a renderer that colours by family has no colour for it and the gap is
# visible rather than plausible.
UNREVIEWED = "unreviewed"

# Appended to genres.ROLLUP_FIELDS. `genre_macro_set` is a list, not a single value:
# 177 of the 1,252 labelled tracks span more than one family, and collapsing them would
# discard information the source actually provided.
MACRO_FIELDS = ("genre_macro_primary", "genre_macro_set", "genre_macro_count")


class MacroMap:
    """The curated table, validated at load and read-only thereafter."""

    def __init__(self, doc, path, sha256):
        self.path = Path(path)
        self.sha256 = sha256
        self.schema_version = doc["schema_version"]
        self.curated_at = doc.get("curated_at")
        self.curated_from = doc.get("curated_from", {})
        self.rules = tuple(doc.get("rules", ()))
        self.families = tuple(doc["families"])
        self.family_ids = tuple(f["id"] for f in self.families)
        self.mapping = dict(doc["mapping"])
        self.notes = dict(doc.get("notes", {}))

    def macro(self, label):
        """The family for one micro token, or UNREVIEWED. Never raises: an unknown token
        is a curation gap to be queued, not a build failure."""
        return self.mapping.get(label, UNREVIEWED)

    def family_name(self, family_id):
        for f in self.families:
            if f["id"] == family_id:
                return f["name"]
        return None

    def unmapped(self, labels):
        """Tokens in `labels` with no row in the table, in the order given."""
        return [t for t in dict.fromkeys(labels) if t not in self.mapping]


def _validate(doc, path):
    """Fail at load, not at render.

    Every check here has a failure mode that is invisible downstream: a typo'd family id
    becomes a colour nobody defined, a duplicate token is a silent overwrite of one
    curation decision by another, and a family with no tokens is a legend entry that can
    never appear.
    """
    for key in ("schema_version", "families", "mapping"):
        if key not in doc:
            raise ValueError(f"{path}: missing required key {key!r}")

    ids = [f["id"] for f in doc["families"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate family ids")
    if UNREVIEWED in ids:
        raise ValueError(
            f"{path}: {UNREVIEWED!r} must not be a declared family. It is the sentinel "
            "for tokens nobody has curated yet; making it a family turns the review "
            "queue into an `Other` bucket."
        )

    used = set(doc["mapping"].values())
    unknown = sorted(used - set(ids))
    if unknown:
        raise ValueError(f"{path}: mapping references undeclared families {unknown}")
    empty = [i for i in ids if i not in used]
    if empty:
        raise ValueError(
            f"{path}: families with zero tokens {empty} — a family that cannot be "
            "anyone's colour is a taxonomy the library does not support"
        )

    if not (12 <= len(ids) <= 15):
        raise ValueError(
            f"{path}: {len(ids)} families; P1-11 fixes the range at 12-15. Going outside "
            "it is allowed but must be an argued change to this bound, not a side effect."
        )

    stray = sorted(set(doc.get("notes", {})) - set(doc["mapping"]))
    if stray:
        raise ValueError(f"{path}: notes for tokens that are not mapped {stray}")


def load(path=MAP_PATH):
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    _validate(doc, path)
    # Hashed as bytes, so reformatting the file counts as a change. It is an input to the
    # build; a silent edit that did not move the version would be undetectable later.
    return MacroMap(doc, path, hashlib.sha256(raw).hexdigest())


def review_queue(vocabulary, macro_map, examples=None):
    """Every token the table does not cover, with the evidence needed to curate it.

    Written on every build, empty or not. A missing file cannot be distinguished from a
    build that never checked; an empty one is a positive statement that it did.
    """
    examples = examples or {}
    tokens = [
        {
            "label": entry["label"],
            "track_count": entry["track_count"],
            "first_count": entry["first_count"],
            "example_track_uris": examples.get(entry["label"], [])[:3],
        }
        for entry in vocabulary
        if entry["label"] not in macro_map.mapping
    ]
    return {
        "macro_map": str(macro_map.path),
        "macro_map_sha256": macro_map.sha256,
        "n_unreviewed_labels": len(tokens),
        "families": [{"id": f["id"], "name": f["name"]} for f in macro_map.families],
        "how_to_clear": (
            "Add a row per token to `mapping` in data/genre_macro_map.json choosing one "
            "of the families above, add a `notes` entry if the call is not self-evident, "
            "then rerun `python -m pipeline.genres`. Do not add an `Other` family."
        ),
        "tokens": tokens,
    }
