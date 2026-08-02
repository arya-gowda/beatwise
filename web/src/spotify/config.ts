/**
 * Auth parameters, and the one place a misconfiguration turns into a legible sentence
 * rather than an opaque Spotify error page.
 *
 * Decision record: docs/decisions/0003-spotify-auth.md
 */

export const AUTHORIZE_URL = 'https://accounts.spotify.com/authorize'
export const TOKEN_URL = 'https://accounts.spotify.com/api/token'
export const API_BASE = 'https://api.spotify.com/v1'

/**
 * Exactly what playlist write-back (P1-07) needs. Identity adds nothing to this list.
 *
 * The reason, stated only as far as it is actually verifiable: the Get Current User's
 * Profile reference documents `user-read-private` and `user-read-email` as gating
 * specific FIELDS (`country`, `product`, `explicit_content`, `email`), not the endpoint,
 * and attaches no scope note to `display_name` or `id`. P1-06 needs only those two
 * fields, so neither scope buys anything here.
 *
 * NOT relied on: Spotify's February 2026 changelog lists those same profile fields as
 * removed, while the reference page still documents them. The two disagree and the
 * changelog carries no effective date, so the argument above is deliberately built
 * without it. If it is accurate, the scopes are even emptier than claimed.
 *
 * This list is also the security ceiling on browser token custody (0003). Every scope
 * added here widens what an exfiltrated refresh token can do. Add one only against a
 * shipped feature, never against a planned one.
 */
export const SCOPES = ['playlist-modify-private', 'playlist-modify-public'] as const

export type SpotifyConfig = {
  clientId: string
  redirectUri: string
}

export type ConfigResult =
  | { ok: true; config: SpotifyConfig }
  | { ok: false; detail: string }

/**
 * Resolve the client id and redirect URI, refusing to proceed on the two
 * misconfigurations that produce confusing failures rather than obvious ones.
 */
export function resolveConfig(): ConfigResult {
  // Spotify does not permit `localhost` as a redirect URI -- loopback must be the
  // explicit literal. The subtle part is not the rejection: `http://localhost:5173` and
  // `http://127.0.0.1:5173` are different origins with different storage, so the verifier
  // written before the redirect is unreachable after it. Caught here, by name.
  const host = window.location.hostname
  if (host === 'localhost' || host.endsWith('.localhost')) {
    return {
      ok: false,
      detail:
        `open the app at http://127.0.0.1:${window.location.port || '5173'} instead of ` +
        `${window.location.origin}. Spotify does not allow "localhost" as a redirect ` +
        'URI, and the two are different browser origins, so the PKCE verifier written ' +
        'before the redirect would not be readable after it.',
    }
  }

  const clientId = (import.meta.env.VITE_SPOTIFY_CLIENT_ID ?? '').trim()
  if (!clientId) {
    return {
      ok: false,
      detail:
        'VITE_SPOTIFY_CLIENT_ID is not set. Copy web/.env.example to web/.env.local, ' +
        'paste the client id from developer.spotify.com/dashboard, and restart the dev ' +
        'server. There is no client secret in the PKCE flow.',
    }
  }

  // Defaults to wherever the app is being served from, which is what should be registered
  // in the dashboard. Overridable for a deployed origin.
  const redirectUri = (
    import.meta.env.VITE_SPOTIFY_REDIRECT_URI ?? `${window.location.origin}/`
  ).trim()

  // The redirect must come back to this origin or the verifier is again unreachable.
  let redirectOrigin: string
  try {
    redirectOrigin = new URL(redirectUri).origin
  } catch {
    return { ok: false, detail: `VITE_SPOTIFY_REDIRECT_URI is not a valid URL: ${redirectUri}` }
  }
  if (redirectOrigin !== window.location.origin) {
    return {
      ok: false,
      detail:
        `VITE_SPOTIFY_REDIRECT_URI points at ${redirectOrigin} but the app is being ` +
        `served from ${window.location.origin}. The PKCE verifier is stored per origin, ` +
        'so the flow cannot complete across the two.',
    }
  }

  return { ok: true, config: { clientId, redirectUri } }
}
