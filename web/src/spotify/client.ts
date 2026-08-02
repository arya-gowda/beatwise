/**
 * The one door to the Spotify Web API.
 *
 * Everything the live layer ever fetches -- artwork, metadata, search, playlist
 * write-back -- goes through `spotifyFetch`. Concentrating it here buys three things that
 * are awkward to get any other way: the deprecated-endpoint guard applies to every call
 * rather than to the calls someone remembered, token refresh on 401 happens once instead
 * of at every call site, and the boundary between "substrate" and "live layer" has a
 * physical location in the codebase rather than only a documented one.
 *
 * Nothing here may influence a point's coordinates. Live-layer data decorates and acts on
 * points; it never moves them.
 */
import { API_BASE } from './config'
import type { SpotifyConfig } from './config'
import { assertNotDeprecated } from './deprecated'
import { NotConnectedError, getAccessToken, rememberIdentity } from './session'

export class SpotifyApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'SpotifyApiError'
    this.status = status
  }
}

/**
 * Call the Spotify Web API with the current access token.
 *
 * @param path a path relative to /v1, e.g. `/me` or `/users/x/playlists`
 */
export async function spotifyFetch(
  config: SpotifyConfig,
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  assertNotDeprecated(path)

  const token = await getAccessToken(config)
  if (!token) throw new NotConnectedError()

  const res = await request(path, init, token)
  if (res.status !== 401) return res

  // The token was rejected despite looking unexpired -- revoked, or our clock is wrong.
  // One forced refresh, one retry, then give up rather than loop.
  const fresh = await getAccessToken(config, { force: true })
  if (!fresh) throw new NotConnectedError('Spotify rejected the stored token')
  return request(path, init, fresh)
}

function request(path: string, init: RequestInit, token: string): Promise<Response> {
  return fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { ...init.headers, Authorization: `Bearer ${token}` },
  })
}

/** The signed-in user, as the header needs them. */
export type SpotifyProfile = {
  id: string
  /** Spotify allows this to be null. Callers fall back to `id`. */
  displayName: string | null
  imageUrl: string | null
  profileUrl: string | null
}

/**
 * `GET /v1/me`.
 *
 * Called with only the two playlist scopes. `display_name` and `id` carry no documented
 * scope requirement; `user-read-private` / `user-read-email` gate other fields this app
 * does not read. See config.ts for the full reasoning and its one unverified assumption —
 * a `403` here rather than a `200` is the signal that the assumption was wrong.
 */
export async function fetchProfile(config: SpotifyConfig): Promise<SpotifyProfile> {
  const res = await spotifyFetch(config, '/me')
  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as { error?: { message?: string } } | null
    throw new SpotifyApiError(
      res.status,
      body?.error?.message ?? `Spotify returned ${res.status} for /me`,
    )
  }

  const me = (await res.json()) as {
    id: string
    display_name: string | null
    images?: { url: string; height: number | null }[]
    external_urls?: { spotify?: string }
  }

  const profile: SpotifyProfile = {
    id: me.id,
    displayName: me.display_name ?? null,
    imageUrl: smallestImage(me.images),
    profileUrl: me.external_urls?.spotify ?? null,
  }

  // The user id is the tenant key the data model carries from day one, so it belongs on
  // the persisted session rather than only in React state.
  rememberIdentity(profile.id, profile.displayName)
  return profile
}

/** The header shows a 24px avatar; taking the 640px original would be silly. */
function smallestImage(images?: { url: string; height: number | null }[]): string | null {
  if (!images?.length) return null
  const sized = [...images].sort((a, b) => (a.height ?? 1e6) - (b.height ?? 1e6))
  return sized[0]?.url ?? null
}
