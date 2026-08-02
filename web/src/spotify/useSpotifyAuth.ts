/**
 * The whole of what the rest of the app is allowed to know about Spotify auth.
 *
 * `{ status, profile, connect, disconnect }` and nothing else. No token, no expiry, no
 * hint of where either is kept. That is what makes 0003's browser-custody decision cheap
 * to reverse: moving the refresh token behind FastAPI changes web/src/spotify/storage.ts
 * and leaves this signature -- and therefore every consumer -- untouched.
 */
import { useCallback, useEffect, useState } from 'react'
import { resolveConfig } from './config'
import type { SpotifyConfig } from './config'
import { fetchProfile } from './client'
import type { SpotifyProfile } from './client'
import {
  NotConnectedError,
  consumeRedirect,
  currentSession,
  describe,
  beginAuthorization,
  disconnect as clearStoredSession,
  resetRedirect,
  subscribe,
} from './session'

export type AuthState =
  /** Missing or contradictory setup. Actionable by the developer, not by the user. */
  | { status: 'unconfigured'; detail: string }
  | { status: 'disconnected' }
  | { status: 'connecting' }
  | { status: 'connected'; profile: SpotifyProfile }
  | { status: 'error'; detail: string }

export type SpotifyAuth = AuthState & {
  connect: () => void
  disconnect: () => void
}

// Evaluated once: neither the environment nor the page origin changes under us.
const configResult = resolveConfig()

// Memoised at module scope for the same reason `consumeRedirect` is -- StrictMode mounts
// the tree twice in development, and this must not become two `/me` calls or, worse, two
// attempts to redeem one single-use authorization code.
let bootstrapPromise: Promise<AuthState> | null = null

function bootstrap(config: SpotifyConfig): Promise<AuthState> {
  bootstrapPromise ??= runBootstrap(config)
  return bootstrapPromise
}

async function runBootstrap(config: SpotifyConfig): Promise<AuthState> {
  const outcome = await consumeRedirect(config)
  if (outcome.kind === 'cancelled') return { status: 'disconnected' }
  if (outcome.kind === 'error') return { status: 'error', detail: outcome.detail }
  if (!currentSession()) return { status: 'disconnected' }

  try {
    // Goes through `getAccessToken`, so a session restored from storage with an expired
    // access token refreshes here -- silently, and without a second trip to Spotify's
    // consent screen. This call is where "persists across reload" is actually proven.
    const profile = await fetchProfile(config)
    return { status: 'connected', profile }
  } catch (err) {
    // A dead refresh token is not an error to report; it is simply being signed out.
    if (err instanceof NotConnectedError) return { status: 'disconnected' }
    return { status: 'error', detail: describe(err) }
  }
}

/**
 * First paint, before any await has resolved.
 *
 * A session that already knows who it belongs to renders the name immediately rather than
 * flashing "Connect Spotify" at someone who is connected. `/me` then confirms or corrects
 * it a moment later.
 */
function initialState(): AuthState {
  if (!configResult.ok) return { status: 'unconfigured', detail: configResult.detail }

  const params = new URL(window.location.href).searchParams
  if (params.has('code') || params.has('error')) return { status: 'connecting' }

  const session = currentSession()
  if (!session) return { status: 'disconnected' }
  if (session.userId) {
    return {
      status: 'connected',
      profile: {
        id: session.userId,
        displayName: session.displayName,
        imageUrl: null,
        profileUrl: null,
      },
    }
  }
  return { status: 'connecting' }
}

export function useSpotifyAuth(): SpotifyAuth {
  const [state, setState] = useState<AuthState>(initialState)

  useEffect(() => {
    if (!configResult.ok) return
    let alive = true
    bootstrap(configResult.config).then(
      (next) => alive && setState(next),
      (err: unknown) => alive && setState({ status: 'error', detail: describe(err) }),
    )
    return () => {
      alive = false
    }
  }, [])

  // A refresh that fails in the background clears the session without anyone clicking
  // anything. Without this the header would keep claiming a name the app can no longer
  // act as.
  useEffect(
    () =>
      subscribe(() => {
        if (!currentSession()) {
          setState((prev) => (prev.status === 'connected' ? { status: 'disconnected' } : prev))
        }
      }),
    [],
  )

  const connect = useCallback(() => {
    if (!configResult.ok) return
    setState({ status: 'connecting' })
    beginAuthorization(configResult.config).catch((err: unknown) => {
      setState({ status: 'error', detail: describe(err) })
    })
  }, [])

  const disconnect = useCallback(() => {
    clearStoredSession()
    resetRedirect()
    bootstrapPromise = null
    setState({ status: 'disconnected' })
  }, [])

  return { ...state, connect, disconnect }
}
