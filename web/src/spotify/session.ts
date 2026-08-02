/**
 * The Authorization Code + PKCE flow itself: start it, finish it, keep it alive, end it.
 *
 * Everything here talks directly to accounts.spotify.com. No backend is involved --
 * see docs/decisions/0003-spotify-auth.md for why, what that costs, and how to reverse it.
 */
import { AUTHORIZE_URL, SCOPES, TOKEN_URL } from './config'
import type { SpotifyConfig } from './config'
import { challengeFor, createState, createVerifier } from './pkce'
import { clearPending, readPending, sessionStore, writePending } from './storage'
import type { StoredSession } from './storage'

/** Refresh this far before the token actually expires, to absorb clock skew and latency. */
const REFRESH_MARGIN_MS = 60_000

/** A `state` older than this is stale rather than forged -- a tab left open for a day. */
const PENDING_MAX_AGE_MS = 15 * 60_000

export class NotConnectedError extends Error {
  constructor(message = 'not connected to Spotify') {
    super(message)
    this.name = 'NotConnectedError'
  }
}

// --- change notification -------------------------------------------------------------
// The session can change without anyone clicking anything: a refresh rewrites it, and a
// rejected refresh clears it. The header has to hear about the second one.

const listeners = new Set<() => void>()

export function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

function emit(): void {
  for (const listener of listeners) listener()
}

// --- reading and writing the session --------------------------------------------------

let cached: StoredSession | null | undefined

export function currentSession(): StoredSession | null {
  if (cached === undefined) cached = sessionStore.read()
  return cached
}

function persist(session: StoredSession): void {
  cached = session
  sessionStore.write(session)
  emit()
}

/** Forget the tokens. See `disconnect` for what this does and does not achieve. */
export function clearSession(): void {
  cached = null
  refreshPromise = null
  sessionStore.clear()
  emit()
}

/**
 * Drop the local tokens.
 *
 * Spotify publishes no revocation endpoint, so this is genuinely all that can be done
 * from here: after it the app holds no token and cannot call the API. The grant itself
 * remains on the account until the user removes it at spotify.com/account/apps. The UI
 * says so rather than implying an authority this function does not have.
 */
export function disconnect(): void {
  clearSession()
  clearPending()
}

// --- starting the flow ----------------------------------------------------------------

/**
 * Generate a verifier, stash it, and hand the browser to Spotify.
 *
 * Does not return in the normal case -- the page navigates away.
 */
export async function beginAuthorization(config: SpotifyConfig): Promise<void> {
  const verifier = createVerifier()
  const state = createState()
  const challenge = await challengeFor(verifier)

  writePending({ verifier, state, redirectUri: config.redirectUri, startedAt: Date.now() })

  const params = new URLSearchParams({
    client_id: config.clientId,
    response_type: 'code',
    redirect_uri: config.redirectUri,
    code_challenge_method: 'S256',
    code_challenge: challenge,
    // Documented as optional. Treated as required: without it the callback cannot be
    // distinguished from one a third party induced.
    state,
    scope: SCOPES.join(' '),
  })

  window.location.assign(`${AUTHORIZE_URL}?${params.toString()}`)
}

// --- finishing the flow ---------------------------------------------------------------

export type RedirectOutcome =
  | { kind: 'none' }
  | { kind: 'connected'; session: StoredSession }
  | { kind: 'cancelled' }
  | { kind: 'error'; detail: string }

let redirectPromise: Promise<RedirectOutcome> | null = null

/**
 * Handle a return from Spotify, if this page load is one.
 *
 * Memoised at module scope, which is load-bearing rather than an optimisation. An
 * authorization code is single-use, and React StrictMode runs effects twice in
 * development; without the memo the second run would present an already-redeemed code and
 * fail with `invalid_grant` on every single connect attempt in dev.
 */
export function consumeRedirect(config: SpotifyConfig): Promise<RedirectOutcome> {
  redirectPromise ??= runRedirect(config)
  return redirectPromise
}

/** Let a later connect attempt in the same page life run cleanly. */
export function resetRedirect(): void {
  redirectPromise = null
}

async function runRedirect(config: SpotifyConfig): Promise<RedirectOutcome> {
  // Everything up to the first await runs synchronously, so the URL is scrubbed before
  // any other code has a chance to observe -- or re-consume -- the code.
  const url = new URL(window.location.href)
  const code = url.searchParams.get('code')
  const state = url.searchParams.get('state')
  const error = url.searchParams.get('error')

  if (!code && !error) return { kind: 'none' }

  scrubAuthParams(url)

  const pending = readPending()
  clearPending()

  if (error) {
    return error === 'access_denied'
      ? { kind: 'cancelled' }
      : { kind: 'error', detail: `Spotify refused the authorization: ${error}` }
  }

  if (!pending) {
    return {
      kind: 'error',
      detail:
        'the PKCE verifier for this callback is missing. This usually means the ' +
        'authorization was started from a different origin or a different tab — check ' +
        'that the app is open at http://127.0.0.1:5173, not localhost.',
    }
  }
  if (pending.state !== state) {
    return { kind: 'error', detail: 'state mismatch on the Spotify callback — request discarded' }
  }
  if (Date.now() - pending.startedAt > PENDING_MAX_AGE_MS) {
    return { kind: 'error', detail: 'that authorization attempt expired — try connecting again' }
  }

  const body = new URLSearchParams({
    grant_type: 'authorization_code',
    code: code as string,
    // Sent again at exchange, and it must match the authorize request byte for byte.
    redirect_uri: pending.redirectUri,
    client_id: config.clientId,
    code_verifier: pending.verifier,
  })

  try {
    const token = await postToken(body)
    const session: StoredSession = {
      version: 1,
      accessToken: token.access_token,
      refreshToken: token.refresh_token ?? null,
      expiresAt: Date.now() + token.expires_in * 1000,
      scope: token.scope ?? '',
      userId: null,
      displayName: null,
    }
    persist(session)
    return { kind: 'connected', session }
  } catch (err) {
    return { kind: 'error', detail: describe(err) }
  }
}

/**
 * Remove the OAuth parameters from the address bar without touching anything else on it,
 * and without adding a history entry the back button would walk into.
 */
function scrubAuthParams(url: URL): void {
  for (const key of ['code', 'state', 'error']) url.searchParams.delete(key)
  const query = url.searchParams.toString()
  window.history.replaceState(null, '', `${url.pathname}${query ? `?${query}` : ''}${url.hash}`)
}

// --- keeping it alive -----------------------------------------------------------------

let refreshPromise: Promise<StoredSession | null> | null = null

/**
 * A usable access token, refreshing first if the current one is spent.
 *
 * Returns null when there is nothing to refresh from -- callers treat that as
 * "disconnected", not as an error.
 */
export async function getAccessToken(
  config: SpotifyConfig,
  options: { force?: boolean } = {},
): Promise<string | null> {
  const session = currentSession()
  if (!session) return null

  const stillGood = Date.now() < session.expiresAt - REFRESH_MARGIN_MS
  if (stillGood && !options.force) return session.accessToken

  const refreshed = await refreshSession(config)
  return refreshed?.accessToken ?? null
}

/**
 * Single-flighted on purpose. Spotify may rotate the refresh token, and two concurrent
 * refreshes presenting the same rotated token would race -- one of them losing, and
 * logging the user out for no visible reason.
 */
function refreshSession(config: SpotifyConfig): Promise<StoredSession | null> {
  refreshPromise ??= runRefresh(config).finally(() => {
    refreshPromise = null
  })
  return refreshPromise
}

async function runRefresh(config: SpotifyConfig): Promise<StoredSession | null> {
  const session = currentSession()
  if (!session?.refreshToken) {
    clearSession()
    return null
  }

  const body = new URLSearchParams({
    grant_type: 'refresh_token',
    refresh_token: session.refreshToken,
    // Required for PKCE specifically; the confidential flow sends it in a Basic header
    // instead, which is exactly the difference that makes PKCE viable without a backend.
    client_id: config.clientId,
  })

  let token: TokenResponse
  try {
    token = await postToken(body)
  } catch (err) {
    // `invalid_grant` means the refresh token is dead -- revoked, or past the six months
    // Spotify allows. Nothing to do but re-authorise, so clear rather than retry forever.
    if (err instanceof TokenError && err.code === 'invalid_grant') {
      clearSession()
      return null
    }
    throw err
  }

  const next: StoredSession = {
    ...session,
    accessToken: token.access_token,
    // Spotify documents the refresh token as not always reissued. When it is absent the
    // existing one stays valid, so overwriting with `?? null` here would be a logout bug.
    refreshToken: token.refresh_token ?? session.refreshToken,
    expiresAt: Date.now() + token.expires_in * 1000,
    scope: token.scope ?? session.scope,
  }
  persist(next)
  return next
}

/** Record who the token belongs to, once `/me` has said. */
export function rememberIdentity(userId: string, displayName: string | null): void {
  const session = currentSession()
  if (!session) return
  if (session.userId === userId && session.displayName === displayName) return
  persist({ ...session, userId, displayName })
}

// --- the token endpoint ----------------------------------------------------------------

type TokenResponse = {
  access_token: string
  token_type: string
  expires_in: number
  refresh_token?: string
  scope?: string
}

export class TokenError extends Error {
  code: string
  constructor(code: string, message: string) {
    super(message)
    this.name = 'TokenError'
    this.code = code
  }
}

async function postToken(body: URLSearchParams): Promise<TokenResponse> {
  const res = await fetch(TOKEN_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  })

  const payload: unknown = await res.json().catch(() => null)

  if (!res.ok) {
    const err = payload as { error?: string; error_description?: string } | null
    const code = err?.error ?? `http_${res.status}`
    throw new TokenError(code, err?.error_description ?? `Spotify token endpoint said ${code}`)
  }

  const token = payload as TokenResponse | null
  if (!token || typeof token.access_token !== 'string') {
    throw new TokenError('malformed_response', 'Spotify returned a token response without a token')
  }
  return token
}

export function describe(err: unknown): string {
  if (err instanceof TokenError) return `${err.message} (${err.code})`
  if (err instanceof Error) return err.message
  return String(err)
}
