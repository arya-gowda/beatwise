/**
 * Playlist write-back -- the only thing Beatwise ever creates on Spotify.
 *
 * Live layer, strictly. A playlist is a route through the map; it decorates and acts on
 * points and never moves one. Nothing here may feed back into a coordinate.
 *
 * ENDPOINTS, verified against the live reference on 2026-08-02 rather than remembered:
 *
 *   POST /v1/me/playlists                  create
 *   POST /v1/playlists/{playlist_id}/items add
 *
 * Both were checked on the reference pages themselves, not only in the February 2026
 * changelog -- that changelog has already been wrong once in this repo (it claimed profile
 * fields were removed that the reference still documents, see 0003). Here the two sources
 * agree, and the reference is decisive:
 *
 *   - "Add Items to Playlist" documents `POST /playlists/{playlist_id}/items`, carries no
 *     deprecation banner, and caps a request at 100 items.
 *   - The older page is titled "Add Items to Playlist [DEPRECATED]", documents
 *     `POST /playlists/{playlist_id}/tracks`, and says verbatim:
 *     "Deprecated: Use Add Items to Playlist instead."
 *   - "Create Playlist" documents `POST /me/playlists`. The changelog records
 *     `POST /users/{user_id}/playlists` as removed in favour of it. This one the ticket
 *     did not flag; it would have been the second way to fail the same way.
 *   - The playlist object's `tracks` field is marked "Deprecated: Use items instead."
 *     Nothing here reads either: creation returns the id and the URL, and that is all the
 *     export needs.
 */
import { SpotifyApiError, spotifyFetch } from './client'
import type { SpotifyConfig } from './config'

/** Documented ceiling on `uris` for one add-items request. Not a tuning knob. */
export const MAX_ITEMS_PER_ADD = 100

/**
 * Spotify accepts anything shaped like a URI and rejects the whole request if one is
 * wrong, so a malformed entry does not lose one track -- it loses the chunk of 100 it
 * travelled with. Checked up front, before anything is created.
 */
const TRACK_URI = /^spotify:track:[A-Za-z0-9]{22}$/

/** Longest a 429 may ask us to wait before we give up and say so. */
const MAX_RETRY_WAIT_MS = 10_000

export type PlaylistRef = {
  id: string
  name: string
  /** `external_urls.spotify`. Null only if Spotify omits it, which it should not. */
  url: string | null
}

export type ExportResult = PlaylistRef & {
  /** Items Spotify accepted. The acceptance criterion is that this equals `requested`. */
  added: number
  requested: number
}

/**
 * The playlist exists but is short.
 *
 * Its own class because it is the one failure that must not be reported as either success
 * or a clean failure: something is sitting in the user's account, it is incomplete, and
 * pretending otherwise is exactly the silent drop this ticket forbids.
 */
export class PartialExportError extends Error {
  result: ExportResult
  constructor(result: ExportResult, detail: string) {
    super(detail)
    this.name = 'PartialExportError'
    this.result = result
  }
}

export type ExportInput = {
  name: string
  description?: string
  uris: string[]
  /** Private by default -- a PM call, and the safer default to get wrong. */
  isPublic?: boolean
  /** Called after every accepted chunk, for an honest in-flight count. */
  onProgress?: (added: number, total: number) => void
}

/**
 * Create a playlist and fill it.
 *
 * Deliberately holds no memory of playlists it has made. Two exports are two `POST
 * /me/playlists` calls and therefore two playlists; there is no id to accidentally reuse
 * and no code path that could mutate an earlier one.
 */
export async function exportPlaylist(
  config: SpotifyConfig,
  input: ExportInput,
): Promise<ExportResult> {
  const uris = input.uris
  if (uris.length === 0) throw new Error('nothing selected to export')

  const bad = uris.filter((u) => !TRACK_URI.test(u))
  if (bad.length > 0) {
    throw new Error(
      `${bad.length} of ${uris.length} selected tracks have a URI Spotify will not accept ` +
        `(first: ${bad[0]}). Refusing rather than creating a short playlist.`,
    )
  }

  const playlist = await createPlaylist(config, input)

  let added = 0
  // Sequential, not parallel. Items land in the order their requests arrive, so
  // concurrent chunks would scramble the nearest-to-centroid order the panel promises --
  // and a mid-flight failure would leave a playlist with holes rather than a short tail.
  for (const batch of chunk(uris, MAX_ITEMS_PER_ADD)) {
    try {
      await addItems(config, playlist.id, batch)
    } catch (err) {
      throw new PartialExportError(
        { ...playlist, added, requested: uris.length },
        err instanceof Error ? err.message : String(err),
      )
    }
    added += batch.length
    input.onProgress?.(added, uris.length)
  }

  return { ...playlist, added, requested: uris.length }
}

/** `POST /v1/me/playlists`. */
async function createPlaylist(config: SpotifyConfig, input: ExportInput): Promise<PlaylistRef> {
  const res = await send(config, '/me/playlists', {
    name: input.name,
    public: input.isPublic ?? false,
    ...(input.description ? { description: input.description } : {}),
  })
  if (!res.ok) throw await failure(res, 'could not create the playlist')

  const body = (await res.json()) as {
    id?: string
    name?: string
    external_urls?: { spotify?: string }
  }
  if (!body?.id) {
    throw new SpotifyApiError(res.status, 'Spotify created a playlist but returned no id')
  }
  return { id: body.id, name: body.name ?? input.name, url: body.external_urls?.spotify ?? null }
}

/** `POST /v1/playlists/{playlist_id}/items`. Max 100 uris, per the reference. */
async function addItems(config: SpotifyConfig, playlistId: string, uris: string[]): Promise<void> {
  const path = `/playlists/${encodeURIComponent(playlistId)}/items`

  // One retry, and only on 429. A 429 is documented as "not processed, come back later",
  // so replaying it cannot double-add. A 5xx carries no such promise: the items may
  // already be in the playlist, and retrying would break the count-equals-selection
  // criterion in the direction nobody would notice.
  let res = await send(config, path, { uris })
  if (res.status === 429) {
    const waitMs = retryAfterMs(res)
    if (waitMs > MAX_RETRY_WAIT_MS) {
      throw new SpotifyApiError(
        429,
        `Spotify is rate limiting this app and asked for ${Math.round(waitMs / 1000)}s. ` +
          'Try again shortly.',
      )
    }
    await sleep(waitMs)
    res = await send(config, path, { uris })
  }

  if (!res.ok) throw await failure(res, `could not add ${uris.length} tracks`)
}

function send(config: SpotifyConfig, path: string, body: unknown): Promise<Response> {
  return spotifyFetch(config, path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

/** Spotify's error envelope, turned into something a person can read. */
async function failure(res: Response, what: string): Promise<SpotifyApiError> {
  const body = (await res.json().catch(() => null)) as { error?: { message?: string } } | null
  const detail = body?.error?.message?.trim()

  if (res.status === 403) {
    return new SpotifyApiError(
      403,
      `${what}: Spotify refused it${detail ? ` — ${detail}` : ''}. If this persists, ` +
        'disconnect and reconnect to re-grant the playlist scopes.',
    )
  }
  return new SpotifyApiError(res.status, `${what}: ${detail ?? `Spotify returned ${res.status}`}`)
}

function retryAfterMs(res: Response): number {
  const seconds = Number(res.headers.get('Retry-After'))
  return Number.isFinite(seconds) && seconds > 0 ? seconds * 1000 : 1000
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/** Split into runs of at most `size`. Exported because the 100 cap is the thing most
 * likely to be got wrong, and it should be readable on its own. */
export function chunk<T>(items: T[], size: number): T[][] {
  const out: T[][] = []
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size))
  return out
}
