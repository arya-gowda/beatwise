# data/

Hand-curated source data. Committed, versioned, and **not** reproducible by any build.

This directory exists to be the opposite of `artifacts/`. `artifacts/` is gitignored
because everything in it is a build output — delete it, run the CLI, get it back byte for
byte. Nothing here comes back. `data/genre_macro_map.json` is roughly an hour of human
judgement over 314 genre tokens; in `artifacts/` it would be one `git clean` from gone.

| file | what it is | who reads it |
|---|---|---|
| `genre_macro_map.json` | 314 micro genre tokens → 12 macro families, one explicit row each (P1-11) | `pipeline/macro.py` |

## Rules for anything added here

1. **If a build can regenerate it, it does not belong here.** It belongs in `artifacts/`.
2. **It is an input, so its hash goes in the consuming artifact's version digest.** Two
   artifacts built from the same CSV under different curation must not share an id.
   `pipeline/genres.py` folds `macro_map_sha256` into the genre version for this reason.
3. **No catch-all rows.** The point of curating by hand is that every value can be
   explained by pointing at a row. A default that absorbs whatever is left over ends that
   property quietly. Unknown inputs go to a review queue — see `macro_review_queue.json`
   in the genre artifact — never to `Other`.
4. **Curate against the emitted tokens, not a fresh scrape.** The genre vocabulary is
   NFKC-composed; 14 tokens are non-ASCII (`sierreño`, `variété française`). A decomposed
   accent is a different string with an identical glyph, so a retyped key silently matches
   nothing. Copy from `vocabulary.json`.
