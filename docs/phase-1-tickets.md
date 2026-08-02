# Phase 1 tickets

Durable copy of the Phase 1 ticket list. The in-session task list does not survive a
restart; this file does. Re-create tasks from these entries after restarting.

**Status: 1–4 done and on `main`. Next unblocked: #5 (lasso), #6 (Spotify auth) — those
two can run in parallel. #7 is the gate ticket.**

Workflow per ticket: feature branch → granular commits → squash → rebase onto `main` → push.

---

## P1-01 Scaffold Vite + React + deck.gl app and FastAPI service — DONE

Vite + React + TS (5173), FastAPI (8000), deck.gl, Spotify PKCE. Vite proxies `/api`.
Decision and accepted costs in `docs/decisions/0001-stack.md`.
Commit: `chore: scaffold Vite + deck.gl app with FastAPI service and record stack decision`

## P1-02 Build the versioned 2D embedding artifact from the CSV — DONE

`python -m pipeline.build` → `artifacts/<version>/` with points, manifest, fitted
scaler+reducer. `python -m pipeline.preview` renders it. All 2,389 tracks, byte-identical
rebuilds. Key dropped (`docs/decisions/0002-key-encoding.md`). Fitting lives only in the
build CLI; `tests/test_api_cannot_fit.py` enforces that structurally.
Commit: `feat(pipeline): build versioned 2D embedding artifact with persisted reducer`

## P1-03 Render 2,389 static points in the browser — DONE

`GET /map` serves points + manifest. One loader (`web/src/map/useMapData.ts`); no component
parses the payload. `OrthographicView` set `flipY:false` so the browser matches
`pipeline.preview` rather than rendering a vertical mirror.
Commit: `feat(map): render static point cloud from embedding artifact`

## P1-04 Add pan, zoom, and hover identification — DONE

Cursor-anchored zoom (~11px drift, verified). `pickingRadius:6` for the dense core. Reset
compares target *and* zoom — comparing zoom alone left it disabled after a pure pan.
Tooltip verified against a 170-char title with 7 artists at the right edge.
Commit: `feat(map): add pan, zoom, and hover track identification`

---

## P1-05 Lasso a region and list the selected tracks

OUTCOME: drag a shape around a corner of the map; points inside light up and a side panel
lists the selected tracks with name, artist, and a count. No export yet.

ACCEPTANCE CRITERIA:
- Freehand or polygon lasso; selected points visually distinct from unselected.
- Selection count displayed and matches the panel row count.
- Clearing the selection is obvious and one action.
- Selection survives pan/zoom — lasso, then zoom in to inspect, without losing it.
- Selecting ~200 points does not stutter.
- Panel readable at 200 rows, not just 12.

WATCH ITEM: measured overplot at the current layout is 0.256 (2D, full 2,389 tracks) —
about a quarter of points are visually on top of another at 800px. If lasso feels
imprecise or grabs too much, suspect the LAYOUT before rewriting the interaction. The fix
lives in P1-02's UMAP params, not here. Do not spend a day tuning hit-testing against a
density problem.

DEPENDENCIES: P1-04. BLOCKS: P1-07.
COMMIT: `feat(map): lasso selection with track list panel`

## P1-06 Connect Spotify and show the signed-in user

OUTCOME: click "Connect Spotify", authorise, your display name appears in the header.
That is the entire ticket.

ACCEPTANCE CRITERIA:
- Authorization Code + PKCE completes end to end.
- Scopes are exactly what playlist write-back needs (`playlist-modify-private`,
  `playlist-modify-public`) plus identity. No speculative Phase 3 scopes.
- Token persists across reload and refreshes on expiry without a re-prompt.
- Disconnect exists and actually clears the token.
- Deprecated endpoints (`audio-features`, `audio-analysis`, `recommendations`) are never
  called. They are gone for new apps and the substrate does not need them.

DEPENDENCIES: P1-01. Runs in parallel with P1-03..P1-05. BLOCKS: P1-07, P1-13.
COMMIT: `feat(spotify): PKCE auth flow with persisted session`

## P1-07 Export the lassoed selection as a Spotify playlist — THE GATE TICKET

OUTCOME: lasso a region, click "Create playlist", a playlist with exactly those tracks
appears in your Spotify account. A link opens it.

PM CALLS ON THINGS THE CONCEPT DOC LEAVES OPEN:
- Name: editable, defaults to `Beatwise — <n> tracks — <date>`.
- Order: nearest-to-centroid first. Arbitrary but not random. Sequencing is Phase 2/3 work
  and must not be quietly invented here.
- Visibility: private by default.

ACCEPTANCE CRITERIA:
- Track count in Spotify equals the selection count. No silent drops.
- Selections over 100 tracks chunked correctly across add-items calls.
- Failed or expired auth produces a legible error, not a silent no-op.
- Exporting twice creates two playlists rather than mutating the first.
- Lasso release → playlist link under ~5s for a 100-track selection.

DEPENDENCIES: P1-05, P1-06. BLOCKS: P1-08.
COMMIT: `feat(playlist): export lasso selection to a Spotify playlist`

## P1-08 GATE: run the Phase 1 gate and record pass or fail

NOT A CODE TICKET. A verification ticket with a real pass/fail, allowed to fail.

THE GATE (§6): dragging a box around a corner of the library and getting a playlist back
has to FEEL GOOD.

PROCEDURE:
1. Lasso the beabadoobee neighbourhood — the cluster the prototype's debug block probes.
2. Export it. Listen end to end.
3. Repeat for two clusters you know well and one you do not recognise.

PASS REQUIRES ALL OF:
- The lassoed region is coherent to the ear. Not "mostly", not "with a few outliers you
  can explain away".
- The map's neighbourhoods correspond to something you recognise about your own taste.
- Lasso → playlist is fast enough that you do it a second time without being asked.
- The unrecognised cluster turns out to be a real thing when you listen, not noise.

IF IT FAILS, diagnose in this order before touching UI:
1. Layout density — overplot 0.256 means the lasso may grab neighbours you did not aim at.
2. The Key encoding decision from P1-02.
3. UMAP params — see `umap_tuning.json` and concept doc §9.2.
4. The interaction itself. Last, not first.

DO NOT START PHASE 2 UNTIL THIS PASSES. P1-09 onward may proceed regardless, but path
drawing, saved routes, and territory naming do not begin on a failed gate.

DEPENDENCIES: P1-07.
COMMIT: `docs: record Phase 1 gate result`

## P1-09 Colour the map by popularity, year, tempo, explicitness, date added

DELIBERATELY AFTER THE GATE. §6 lists colour-by before lasso; that ordering is wrong here.
Colour-by is decoration on a map that has not yet proven it can produce a playlist, and
"build the visualisation and stop" is this project's most likely failure.

SCOPE: popularity, release year, tempo, explicitness, date added. Genre is NOT in this
ticket — it needs P1-10 and P1-11.

ACCEPTANCE CRITERIA:
- Five colour modes, each with a legend or continuous scale.
- Popularity colouring reproduces the prototype's popularity view — the founding idea
  (position means sound, popularity is a separate lens) made visible in the product.
- 125 tracks have year-only `Release Date`; year colouring handles them without dropping
  or misplacing them.
- `Added By` is entirely null — not a colour mode, must not appear as one.
- Switching modes does not reset pan/zoom or clear a selection.
- Colours distinguishable at full zoom-out where points are 2-3px.

DEPENDENCIES: P1-04. BLOCKS: P1-12.
COMMIT: `feat(map): colour-by selector for popularity, year, tempo, explicit, added`

## P1-10 Parse Genres into the per-label schema with rollups

Genre tier 1, per §8.8. Tier 1 is not a fetch — the CSV's `Genres` column already IS
Spotify artist genres. This ticket parses what is there into a queryable shape.

SCHEMA (§8.6): `track_uri, label, rank, source=csv, scope=artist, confidence=1.00,
fetched_at`. One row per (track,label), never a joined string.
ROLLUPS: `genre_micro_primary, genre_micro_all, genre_label_count, genre_status`.

ACCEPTANCE CRITERIA:
- 2,404 label instances across 1,252 tracks. Both asserted, not assumed.
- Rank preserved 0-based. Order is real signal — "surf rock" is first 100% of the time it
  appears, "west coast hip hop" 0 of 37.
- 314 unique tokens after normalisation, 106 singletons.
- Plain comma split, normalisation applied anyway. Measured clean today; a property of
  this export, not a guarantee.
- `genre_status` is `native` for 1,252 and `unlabelled` for 1,137. Only those two values
  exist in Phase 1.
- Empty Genres strings produce zero rows, not one row with an empty label.

DEPENDENCIES: P1-02. BLOCKS: P1-11, P1-13.
COMMIT: `feat(genre): parse CSV genres into per-label schema with track rollups`

## P1-11 Curate the 314 micro to ~12-15 macro genre mapping table

A data ticket, ~1 hour of hand review (§8.3). Hand curation is the only approach where a
macro label can be explained by pointing at a row.

ACCEPTANCE CRITERIA:
- All 314 tokens mapped. Zero unmapped, including the 106 singletons.
- 12-15 macro families, modelled on the Discogs genre/style split.
- Unmapped tokens arriving later surface in a review queue file rather than silently
  defaulting to Other.
- `genre_macro_set` populated. 184 of the 1,252 labelled tracks span more than one macro
  family; collapsing them loses information the source provided.
- Primary selection follows §8.6: confidence, then track scope over artist scope, then
  lowest rank, then alphabetical. In Phase 1 every label is artist-scope at 1.00, so rank
  then alphabetical decides — without which a 14-label track is a 14-way tie.

DEPENDENCIES: P1-10. BLOCKS: P1-12.
COMMIT: `feat(genre): curated micro-to-macro mapping table and macro rollups`

## P1-12 Colour the map by macro genre, with unlabelled legible

PM CALL ON A DOC CONFLICT: §8.7 wants low-confidence labels desaturated; §Verification
forbids showing confidence before calibration (Phase 3). Resolution: in Phase 1 every
label is ground truth at 1.00, so there is nothing uncalibrated to desaturate. Phase 1
renders exactly TWO states — sourced and unlabelled. No confidence slider, no desaturation
ramp.

ACCEPTANCE CRITERIA:
- Colour keys off `genre_macro`; a point is one colour.
- The 1,137 unlabelled render in a neutral that reads as absence, not a 13th genre.
- Hover reveals the full `genre_macro_set` for multi-macro tracks.
- 12-15 colours distinguishable at 2-3px in the dense core. If not, the macro count is the
  problem, not the palette — say so rather than shipping 15 similar shades.
- The legend states the coverage figure plainly. 52% is the honest number.

DEPENDENCIES: P1-09, P1-11.
COMMIT: `feat(map): colour by macro genre with unlabelled rendered distinctly`

## P1-13 Confirm the genre gap is empty at source (one cheap Spotify run)

OPTIONAL, LOW PRIORITY, gate-independent. §8.1 calls this "worth one cheap test run to
confirm; not worth planning around."

ACCEPTANCE CRITERIA:
- Artist-genres fetched for a sample of the 660 empty artists (100 is plenty).
- Result recorded as a number: how many came back non-empty.
- If materially more than a handful come back populated, STOP and re-open §8 — the
  diagnosis is wrong and the tier 2-4 cascade may be unnecessary.
- Runs once, cached. Not wired into the app.

DEPENDENCIES: P1-06, P1-10.
COMMIT: `chore(genre): confirm artist-genre gap is empty at source`

## P1-14 OPTIONAL: pick a CSV in the browser and rebuild the map

OPTIONAL AND GATE-INDEPENDENT. Ship only if "CSV upload" being literally true in Phase 1
matters more than starting Phase 2. Became cheaper once FastAPI was chosen, but still
contributes nothing to the gate.

ACCEPTANCE CRITERIA:
- Pick `Liked_Songs.csv`, wait, the map reloads with the rebuilt artifact.
- Malformed or wrong-schema CSV produces a legible error, not a blank map.
- The uploaded file is PROJECTED through the persisted reducer where tracks already exist;
  only genuinely new tracks move anything. Existing points must not move (§5 seam item 3).
- No CSV parsing happens in JS — the UTF-8 BOM is handled in Python where pandas already
  strips it.

DEPENDENCIES: P1-02, P1-03.
COMMIT: `feat(ingest): rebuild the map from a browser-selected CSV`
