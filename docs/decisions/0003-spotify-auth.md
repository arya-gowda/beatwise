# 0003 — Spotify auth: where the refresh token lives

**Status:** accepted
**Ticket:** P1-06

## Decision

- **Authorization Code + PKCE, run entirely in the browser.** The `/authorize` redirect,
  the code-for-token exchange, and every refresh are `fetch` calls from the web app
  straight to `accounts.spotify.com`. FastAPI is not involved in auth at all.
- **The refresh token is persisted in `localStorage`**, under a single versioned record
  (`beatwise.spotify.session.v1`) alongside the access token, its expiry, the granted
  scope string, and the Spotify user id.
- **Scopes are exactly two:** `playlist-modify-private`, `playlist-modify-public`.
  Identity adds nothing — see below.
- **Storage sits behind a `SessionStore` interface** (`web/src/spotify/storage.ts`) so
  that moving custody to the server later is a change to one file, not to the flow.
- **The redirect URI is `http://127.0.0.1:5173/`.** Not `localhost`. Not a sub-path.

## The tension this resolves

0001 says two things that do not sit together:

> PKCE needs no backend for auth *(the argument for Vite over Next)*

> [FastAPI] gives Spotify token exchange a home *(the argument for FastAPI over a CLI)*

Both are true; they are answers to different questions. PKCE *needs* no backend — that is
what makes a static-ish Vite app a viable shape for this product, and 0001's Vite reasoning
survives intact. FastAPI merely *offers* token exchange a home; it never obliged it to
move in. 0003 declines the offer and keeps auth in the browser. The offer stays open, and
"Reversing this" below is the route to accepting it.

## Why the browser and not FastAPI

**The multi-tenancy argument for a server-side token vault is foreclosed by Spotify, not
by us.** The usual reason to route token exchange through a backend is that browser
storage does not scale to many users. Beatwise cannot have many users. Apps registered
now start in development mode, which caps at **five** authorised users added by hand
under Settings → User Management, and extended quota mode has since April 2025 required an
organisation with 250k+ MAU. Beatwise is permanently a five-seat personal tool as far as
Spotify is concerned. The expansion seam that matters elsewhere in this system — carrying
a user id through the data model — costs nothing and is honoured (the session record
stores `userId`). The one that would have cost a stateful API buys nothing, because the
population it serves is capped at five by someone else's policy.

**The blast radius of the thing we are accepting is small and bounded by the scope list.**
The honest cost of browser custody is that an XSS in the app can exfiltrate a refresh
token that stays valid for six months. What that token can do is the entire question, and
here it is: create and modify playlists. It cannot read the library, cannot control
playback, cannot reach the email address (Spotify removed `email`, `country`, `product`,
`followers` and `explicit_content` from `/me` in February 2026). Keeping the scope list to
two is therefore not tidiness — it is the mitigation.

**A server-side vault would make the API stateful, which is a real change in its
character.** `api/main.py` is deliberately read-only and has no session concept; 0001 made
that read-only-ness load-bearing and `tests/test_api_cannot_fit.py` enforces a piece of
it. Adding cookie sessions and a token table is not forbidden, but it is the largest new
idea in an app whose server currently answers two GETs — and it would make "who am I
signed in as" depend on uvicorn being up, in a tool whose front end otherwise runs fine
against a stale artifact.

*Cost accepted:* the refresh token is readable by any script that achieves execution in
the app's origin. Mitigated by the two-scope ceiling above, and by the fact that the only
untrusted strings the app renders are track and artist names, which React escapes. This is
a **worse** posture than a backend-for-frontend and is stated as such. If Beatwise ever
escapes development mode, this decision is revisited before, not after.

*Cost accepted:* `localStorage` over `sessionStorage`. `sessionStorage` survives a reload
but not a closed tab, which fails what "Connect Spotify" implies. The transient half of
the flow — the PKCE verifier and the CSRF `state` — does go in `sessionStorage`, where
tab-scoped and short-lived is exactly right, and is deleted the moment the code is
exchanged.

## Scopes: why identity contributes zero

P1-06 needs two fields from `GET /v1/me`: `display_name` and `id`. The
[Get Current User's Profile reference][me] documents `user-read-private` and
`user-read-email` as gating *specific fields* — `country`, `product`, `explicit_content`,
`email` — and attaches no scope note to `display_name` or `id`. Neither scope therefore
buys anything this ticket uses. So "plus identity" resolves to the empty set, and the
scope list is precisely the two the P1-07 playlist write needs.

**What this argument deliberately does not rest on.** Spotify's
[February 2026 changelog][feb26] lists `country`, `email`, `product`, `explicit_content`
and `followers` as *removed* from the user profile, which would make the two identity
scopes empty outright. The reference page still documents those fields, and the changelog
carries no effective date. The two sources disagree and the disagreement was not resolved,
so the scope decision is built without the stronger claim. If the changelog is accurate,
the conclusion only gets safer.

*Unverified, and it needs a real account to settle:* that `/v1/me` answers `200` rather
than `403` for a token carrying only playlist scopes. The reference lists the scopes under
an "Authorization scopes" heading that can read as an endpoint requirement, while the
per-field notes read as conditional. Every reading of the field notes says `display_name`
is unscoped, but "the field is unscoped" and "the endpoint is unscoped" are not the same
statement. **If `/me` 403s, adding `user-read-private` to `SCOPES` is the one-line fix and
this section is the thing that was wrong.**

[me]: https://developer.spotify.com/documentation/web-api/reference/get-current-users-profile
[feb26]: https://developer.spotify.com/documentation/web-api/references/changes/february-2026

## The redirect URI is a trap and is worth writing down

`localhost` is **not permitted** as a Spotify redirect URI. Loopback must be the explicit
literal — `http://127.0.0.1:PORT` or `http://[::1]:PORT`. HTTP is allowed for loopback and
nowhere else; Spotify banned plain-HTTP redirect URIs and localhost aliases on
27 November 2025 and carved out loopback IPs.

The failure this causes is nastier than a rejected URI. `http://localhost:5173` and
`http://127.0.0.1:5173` are **different browser origins with different storage**. Browse to
`localhost`, and the PKCE verifier is written to `localhost`'s `sessionStorage` while
Spotify returns you to `127.0.0.1`, where it does not exist — producing a "missing
verifier" error that looks like a bug in the app. Handled twice, deliberately:
`web/vite.config.ts` pins `server.host` to `127.0.0.1` so the URL Vite prints is the
correct one, and `web/src/spotify/config.ts` refuses to start the flow on a `localhost`
hostname with a message naming the fix.

## Disconnect is local-only, and says so

Spotify publishes no token revocation endpoint. "Disconnect" deletes the stored record —
after it, the app holds no token and cannot call the API, which is the acceptance
criterion. It does **not** withdraw the grant on Spotify's side; the app stays listed at
<https://www.spotify.com/account/apps>. The UI links there rather than implying an
authority it does not have.

## Deprecated endpoints

`audio-features`, `audio-analysis` and `recommendations` (and `related-artists`,
`featured-playlists`, 30-second preview URLs) are unavailable to apps registered on or
after 27 November 2024. Beatwise is new, so they are simply gone — this is the reason the
substrate is a CSV at all (0001, concept doc §5). Two guards, because "we just won't call
them" is not a guarantee:

- Every Spotify request goes through `spotifyFetch` in `web/src/spotify/client.ts`, which
  throws before the network call if the path matches the forbidden list.
- `tests/test_no_deprecated_spotify_endpoints.py` fails if those names appear anywhere in
  `web/src` outside the one file that declares them.

No fallback path is written against them. It would be dead code on arrival.

## Rejected

- **Backend-for-frontend (FastAPI holds the refresh token, browser holds a session
  cookie).** The correct answer for a multi-user product and the reason this record exists
  at all. Rejected on the five-user ceiling, the two-scope blast radius, and the cost of
  making a deliberately stateless read-only API stateful. Not rejected on merit.
- **Authorization Code with a client secret.** Requires a backend anyway and puts a
  long-lived secret in the repo's deployment story for no gain over PKCE.
- **Implicit grant.** Discontinued by Spotify on 27 November 2025. Not an option.
- **A dedicated `/callback` route.** Would work under Vite's SPA fallback but adds a
  history-fallback dependency to every future host for no benefit. The app root reads
  `?code=` and scrubs it with `history.replaceState` on the same tick.

## Reversing this

Cheap, and deliberately so. Everything outside `web/src/spotify/` sees only
`useSpotifyAuth()` → `{ status, profile, connect, disconnect }` and `spotifyFetch(path)`.
Nothing else knows a token exists, let alone where it is kept. Moving to the BFF means:

1. Add `/auth/*` routes to FastAPI holding the verifier and doing the exchange (it may
   hold tokens; it still must not import umap — 0001's guarantee is orthogonal and stays).
2. Replace the `SessionStore` implementation in `web/src/spotify/storage.ts` with one
   whose read is a cookie-authenticated call to the API.
3. Point `spotifyFetch` at a proxy route instead of `api.spotify.com`.

The user re-authorises once. With one user, that is a click.

## Configuration

`VITE_SPOTIFY_CLIENT_ID` is required and read from `web/.env.local`, which is gitignored.
`web/.env.example` documents it. There is no client secret in this flow — PKCE is a public
client — so nothing secret is committed and nothing secret needs to exist. Register
`http://127.0.0.1:5173/` as a redirect URI in the Spotify dashboard, and add yourself
under Settings → User Management. The dashboard app owner needs Spotify Premium for a
development-mode app to function.
