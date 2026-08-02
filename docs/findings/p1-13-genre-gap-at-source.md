# P1-13 — is the genre gap empty at source?

**Run date:** 2026-08-02 · **Status:** the question could not be answered, for a reason
that matters more than the answer would have.

## The headline

**Spotify no longer sends artist genres to this application at all.** Not an empty array —
the `genres` key is absent from the artist object entirely. So is `popularity`, and so is
`followers`.

    GET /v1/artists/7CajNmpbOovFoOoasH2HaY        (Calvin Harris)
    -> keys: external_urls, href, id, images, name, type, uri

That means §8's premise — that the 1,137 unlabelled tracks are unlabelled because Spotify
genuinely has no genres for those artists — is **neither confirmed nor refuted**. It
remains an assumption, and it is now an unfalsifiable one via the Spotify Web API.

It also means **§8 tier 2 (fetch artist genres from Spotify) is impossible, not
unnecessary.** That is a change to the cascade's shape, not to its priority.

## The numbers

Sample of 100 drawn (seed 42) from the 635 artists the CSV never attributes a genre to.

| | |
|---|---|
| sampled | 100 |
| **resolution failures** | **0** (0 no search hit, 0 no exact name match) |
| resolved to a Spotify artist id | 100 |
| → resolved, Spotify says no genres | 0 |
| → resolved, Spotify HAS genres the CSV omitted | 0 |
| → **resolved, `genres` field absent from the response** | **100** |
| ambiguous resolutions (>1 exact name match) | 21 |

Name→id resolution worked perfectly — 100 of 100, zero failures — so the two numbers the
ticket asked to keep apart are 0 and 0. Neither is the interesting one. The third column is.

### The positive control is what caught it

Ten artists the CSV *does* carry genres for, credited alone on their rows so the labels
cannot belong to a collaborator, fetched by the identical path:

| | |
|---|---|
| control artists | 10 |
| resolved | 10 |
| **came back with genres** | **0** |

Al Green, Brenda Lee, Calvin Harris, Creedence Clearwater Revival, Farruko, Frankie Ruiz,
Gus Dapperton, Sugar's Campaign, Thin Lizzy, Toco. All ten resolved to the correct artist
id. None returned a `genres` key.

Without the control this run would have reported *"0 of 100 — the gap is empty at source,
§8 confirmed"* and been wrong. "These 100 artists have no genres" and "the field returns
nothing for anybody" are the same measurement unless something known-good is measured
alongside. The first reads as a clean confirmation of the very thing it fails to test.

## Population, and why 635 rather than 660

| | |
|---|---|
| tracks | 2,389 |
| rows with an empty `Genres` cell | 1,137 |
| distinct artists in the library | 1,495 |
| artists credited on ≥1 empty row | 718 |
| …of those, also credited on a labelled row | 83 |
| **artists the export never attributes a genre to** | **635** |

The `Genres` cell is the union over the credited artists, so an empty cell means every
artist on that row contributed nothing. The 83 who also appear on a labelled row are
excluded rather than guessed at — their row's labels may belong entirely to a
collaborator, so calling them "an artist Spotify has no genres for" would be an inference
dressed as a measurement. The ticket's "~660" sits between 635 and 718; both are reported
so neither has to be reconstructed later.

## What was verified, and what was not

**Verified first-hand on 2026-08-02 with a working Client Credentials token:**

- `GET /v1/artists/{id}` returns artist objects with no `genres`, `popularity` or
  `followers` key, for 110 distinct artists including ten known to have genres.
- `GET /v1/search?type=artist` returns the same shape. Its `limit` maximum is **10**, not
  the 50 that older docs and most tutorials give.
- `GET /v1/artists?ids=` (batch) carries an endpoint-level **Deprecated** tag on its
  reference page, and the February 2026 changelog lists it as removed. Not used.

**Documented but contradicted by observation:** the [Get an Artist][artist] reference page
still lists `genres` on the artist object, marked as a deprecated *field*. The
[February 2026 changelog][feb26] records `followers` and `popularity` as removed from the
artist object and says nothing about `genres`; the [March 2026 changelog][mar26] is two
`external_ids` reversions and does not mention it either. So the removal of `genres` is
undocumented, and behaviour wins over documentation here.

**Not confirmed, and it needs a second Spotify app to settle:** whether this is universal
or specific to applications registered on or after 2024-11-27 (Beatwise is one of those —
see `docs/decisions/0003-spotify-auth.md`). One app cannot distinguish "removed for
everyone" from "removed for new apps". Third-party reports of artist genres thinning out
through 2025 exist but were not corroborated well enough to build on.

**Not tested:** whether album objects still carry genres, and whether any other endpoint
exposes genre data. Out of scope for one cheap run.

[artist]: https://developer.spotify.com/documentation/web-api/reference/get-an-artist
[feb26]: https://developer.spotify.com/documentation/web-api/references/changes/february-2026
[mar26]: https://developer.spotify.com/documentation/web-api/references/changes/march-2026

## What this changes

Scope calls belong to the product-manager agent; these are the systems consequences.

1. **The CSV's `Genres` column is not refreshable.** It is the only genre data Beatwise
   can obtain from Spotify, it arrived as a historical artefact of when the export was
   taken, and it cannot be topped up, re-derived or re-fetched. `Liked_Songs.csv` is now
   an archive as well as a feature substrate.
2. **Re-exporting the CSV may return fewer genres than it does today.** If Spotify has
   been thinning artist genres, a fresh export is not guaranteed parity with the current
   2,404 label instances. Worth keeping the current file rather than assuming it can be
   regenerated. This is the same class of hazard P1-11 flagged about the hand-curated
   macro table, arriving from the other direction.
3. **§8 tier 2 has no implementation path.** Tiers 3 and 4 — a third-party provider, or
   labels derived from the substrate — are the only remaining routes to coverage above
   52%. Whether that is worth doing is a scope question, not a systems one.
4. **P1-09's popularity colouring is unaffected.** `Popularity` is a CSV column and lives
   in the substrate. The artist object's `popularity` is gone, but nothing reads it.

## Reproducing it

Requires `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` in the repo-root `.env`
(gitignored; the client credentials flow needs a secret, unlike the browser's PKCE flow,
which has none and must keep it that way). Then, from the repo root:

    env/bin/python -m diagnostics.genre_gap_at_source

The first run spends ~220 requests and writes
`artifacts/diagnostics/genre-gap/675effce33a2/result.json`. Every run after that reads the
cache and spends nothing. `--refresh` forces a re-fetch.

Because `artifacts/` is gitignored, **this note is the durable record** — the cache is
derived and one `git clean` from gone.
