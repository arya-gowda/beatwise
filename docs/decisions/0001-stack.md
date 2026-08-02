# 0001 — Web stack

**Status:** accepted
**Ticket:** P1-01

## Decision

- **Vite + React + TypeScript** for the app shell
- **deck.gl** for map rendering
- **FastAPI** for the Python service
- **Spotify Authorization Code + PKCE** for auth

## Why

**Vite over Next.js.** PKCE needs no backend for auth, and FastAPI already provides a
server for everything else. That removes Next's main justification, leaving only the
App Router's server/client component overhead on an app whose server does very little.
Faster dev loop, fewer concepts.

*Cost accepted:* if Beatwise ever opens to the public, a Vite → Next migration is real
work. Judged the cheaper end of the trade for a personal tool.

**deck.gl.** The only renderer that covers all three of: Phase 4's ~1M-point discovery
corpus, Phase 2's custom path-drawing interaction, and Phase 5's 3D view. Charting
libraries handle the first badly and the second not at all.

*Cost accepted:* deck.gl core does **not** ship lasso selection. Phase 1 either adds
nebula.gl edit-modes or hand-rolls polygon + point-in-polygon. Budgeted in P1-05, and
the hand-rolled path is wanted anyway for P2's path drawing.

**FastAPI over an offline CLI + file seam.** User decision, overriding the PM's
recommendation. It makes browser CSV upload natural later and gives Spotify token
exchange a home.

*Cost accepted, and the reason this needs watching:* a live service is a standing
invitation to re-fit the embedding on request, and a re-fit silently moves every
existing point — scrambling learned territories and breaking saved routes. **Fitting
happens only in the build CLI.** The API reads the artifact and projects new tracks
through the persisted reducer; it must not be able to fit. Enforced structurally in
P1-02, not by convention.

**Python stays Python.** UMAP is not ported to JS.

## Rejected

- **Next.js** — server overhead without payoff once FastAPI exists.
- **Streamlit / Plotly Dash** — fastest to a map, but the rerun-on-interaction model
  makes Phase 2's path drawing effectively impossible, and OAuth redirects are awkward.
  A dead end, not a shortcut.
- **Plotly.js scattergl** — lasso for free and continuous with the existing prototype,
  but a low ceiling on both point count and custom interaction. A visible future rewrite.

## Layout

```
web/        Vite + React app          npm run dev      (5173)
api/        FastAPI service           uvicorn api.main:app --port 8000
pipeline/   embedding build CLI       python -m pipeline.build
ml/         completed metric-learning work
```

Vite proxies `/api/*` to port 8000, so both halves are same-origin in development and
there is no CORS middleware to configure now and unwind later.
