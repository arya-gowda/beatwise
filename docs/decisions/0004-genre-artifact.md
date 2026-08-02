# 0004 — Genres are a separate artifact, not part of the embedding

**Status:** accepted
**Ticket:** P1-10

## Decision

Genre labels live in their own versioned artifact at `artifacts/genres/<version>/`, on
their own version line, built by its own CLI:

```
python -m pipeline.build      -> artifacts/<version>/          points, reducer, scaler
python -m pipeline.genres     -> artifacts/genres/<version>/   labels, rollups, vocabulary
```

The two join on **Track URI** and share nothing else. Each records `source_sha256`, so a
mismatch is detectable rather than silent.

## Why not fold them into the embedding artifact

Because a genre change would then require a new embedding id, and a new embedding id
means a re-fit — which moves every point.

That is the exact failure mode 0001 and `pipeline/build.py` exist to prevent, arriving
through a side door. The genre work ahead is all *iteration*: P1-11 curates 314 micro
tokens into ~13 macro families and will be revised, P1-13 may add fetched labels, and
tiers 2-4 add more sources still. Every one of those is a reason to rebuild genres, and
none of them is a reason to move a single point.

The rule from the concept doc reads cleanly here: **if it moves a point it belongs to the
substrate; if it decorates a point it belongs to the live layer.** Genres arrive in the
substrate *file* but they behave like live-layer data — they colour points, they never
place them. Popularity is excluded from `FEATURES` for the same reason and this is the
same call applied to a second column.

## What the artifact holds

`labels.json` — the §8.6 table, one row per (track, label), authoritative:

```
track_uri, label, rank, source=csv, scope=artist, confidence=1.00, fetched_at
```

Never a joined string. A joined string cannot answer "which tracks are shoegaze" without
a substring match that also hits "blackgaze", and has nowhere to put rank, scope or
confidence. The relational shape is what makes tiers 2-4 additive: a fetched label is a
new row with a different `source`, not a re-parse of an existing one.

`tracks.json` — the rollups, one per track, a materialised join so the renderer does not
regroup 2,404 rows per frame: `genre_micro_primary`, `genre_micro_all`,
`genre_label_count`, `genre_status`.

`vocabulary.json` — 314 tokens with `track_count` and `first_count`. Written indented
because a human reads this one during P1-11's curation pass.

`manifest.json` — provenance and the counts, `user_id` carried at artifact scope exactly
as the embedding manifest carries it.

## Notes on the parse

**`fetched_at` for `source=csv` is when Beatwise obtained the labels, not when Spotify
assigned them.** The export carries no date and tier 1 is not a fetch. Real provenance is
`source_sha256`. Tier 2 rows will carry genuinely varying values in this column, which is
why it exists now rather than later.

**The separator is a comma, and that fact does not generalise.** `Genres` measures at 628
comma-bearing rows and zero semicolon-bearing rows. `Artist Name(s)` in the same file is
semicolon-separated, and a comma split there fused multi-artist credits until `5533a94` —
27 rows have a comma inside one artist name. `pipeline/genres.py` raises if a semicolon
ever appears in `Genres`, because the failure mode of guessing is 2,404 plausible wrong
labels and no error.

**Normalisation runs even though it currently does nothing.** Zero of 2,404 tokens differ
from their normalised form today. That is a property of this export, not a guarantee, and
`tests/test_genres.py` asserts it stays true so a change surfaces as a failure rather than
as a quietly larger vocabulary. NFKC is the part that earns its keep: 14 tokens are
accented (`sierreño`, `variété française`), and a decomposed `é` is a different string
with the same glyph — it would sit in the vocabulary as a 315th token that P1-11's mapping
cannot match and nobody can see is wrong.

## Reversing this

Cheap. The rollups are derived from `labels.json`, and `labels.json` is derived from the
CSV; deleting `artifacts/genres/` and rebuilding costs a second. Merging the two artifacts
later is a schema change, not a data migration.

The one thing that is not cheap to reverse is the opposite choice — having shipped genres
inside the embedding artifact and then needing to split them, by which point every genre
revision has minted an embedding version and the version history no longer distinguishes
"the map changed" from "a label changed".

`points.json` keeps a convenience `genres` list for hover, produced by the same splitter
so the two cannot drift. It is a copy, not a second source of truth; drop it whenever the
web app reads `/genres` instead.
