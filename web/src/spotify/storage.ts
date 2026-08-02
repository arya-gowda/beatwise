/**
 * Where the session lives -- and the seam that makes 0003 cheap to reverse.
 *
 * The decision to keep the refresh token in the browser is recorded, argued and costed in
 * docs/decisions/0003-spotify-auth.md. It is also the decision most likely to be revisited
 * if Beatwise ever leaves Spotify's five-user development mode. So it is confined to one
 * implementation of one interface: swapping `sessionStore` for a cookie-authenticated
 * call to FastAPI changes this file and nothing else. Nothing outside web/src/spotify
 * knows a token exists at all.
 */

/**
 * `version` is not ceremony. A stored record whose shape has drifted must be *discarded*,
 * not fed to code expecting a different shape -- the failure mode of the latter is an
 * auth loop that looks like Spotify rejecting the user.
 */
export type StoredSession = {
  version: 1
  accessToken: string
  /** Absent only if Spotify declined to issue one, which it does not do for PKCE today. */
  refreshToken: string | null
  /** Epoch milliseconds. */
  expiresAt: number
  /** What Spotify actually granted, which can differ from what was asked for. */
  scope: string
  /**
   * The Spotify user id. Carried from day one even though there is exactly one user:
   * it is the tenant key the rest of the data model will need, and retrofitting it is
   * the classic personal-tool tax.
   */
  userId: string | null
  displayName: string | null
}

export interface SessionStore {
  read(): StoredSession | null
  write(session: StoredSession): void
  clear(): void
}

const SESSION_KEY = 'beatwise.spotify.session.v1'

/**
 * localStorage, not sessionStorage: "Connect Spotify" implies a connection that outlives
 * the tab. sessionStorage would survive a reload and silently drop on close, which reads
 * as the app forgetting for no reason.
 */
const localStorageStore: SessionStore = {
  read() {
    let raw: string | null
    try {
      raw = window.localStorage.getItem(SESSION_KEY)
    } catch {
      return null // private mode, or storage disabled
    }
    if (!raw) return null
    try {
      const parsed = JSON.parse(raw) as Partial<StoredSession>
      if (parsed?.version !== 1 || typeof parsed.accessToken !== 'string') {
        window.localStorage.removeItem(SESSION_KEY)
        return null
      }
      return parsed as StoredSession
    } catch {
      window.localStorage.removeItem(SESSION_KEY)
      return null
    }
  },
  write(session) {
    try {
      window.localStorage.setItem(SESSION_KEY, JSON.stringify(session))
    } catch {
      // Storage full or blocked. The in-memory session still works for this page life;
      // failing the whole connect over a persistence problem would be worse.
    }
  },
  clear() {
    try {
      window.localStorage.removeItem(SESSION_KEY)
    } catch {
      // nothing to do -- there is no state we could be leaving behind that we can reach
    }
  },
}

export const sessionStore: SessionStore = localStorageStore

// --- the transient half of the flow -------------------------------------------------
//
// The verifier and the CSRF state are alive for exactly one redirect. sessionStorage is
// the right home for them: tab-scoped, and gone when the tab is. They are deleted the
// moment the code is exchanged, successfully or not.

const PENDING_KEY = 'beatwise.spotify.pending.v1'

export type PendingAuthorization = {
  verifier: string
  state: string
  redirectUri: string
  startedAt: number
}

export function writePending(pending: PendingAuthorization): void {
  window.sessionStorage.setItem(PENDING_KEY, JSON.stringify(pending))
}

export function readPending(): PendingAuthorization | null {
  const raw = window.sessionStorage.getItem(PENDING_KEY)
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw) as Partial<PendingAuthorization>
    if (typeof parsed?.verifier !== 'string' || typeof parsed?.state !== 'string') return null
    return parsed as PendingAuthorization
  } catch {
    return null
  }
}

export function clearPending(): void {
  window.sessionStorage.removeItem(PENDING_KEY)
}
