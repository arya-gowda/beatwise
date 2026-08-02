/**
 * The export, as the map is allowed to see it.
 *
 * Same contract as `useSpotifyAuth`: the caller gets a state machine and a function, and
 * learns nothing about tokens, config or the API. That is what keeps 0003's
 * browser-custody decision cheap to reverse -- moving the token behind FastAPI changes
 * storage.ts and leaves this signature, and therefore the map, untouched.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { resolveConfig } from './config'
import { NotConnectedError, describe } from './session'
import { PartialExportError, exportPlaylist } from './playlists'
import type { ExportInput, ExportResult } from './playlists'

// Pure and cheap; neither the environment nor the page origin changes under us.
const configResult = resolveConfig()

export type ExportState =
  | { phase: 'idle' }
  | { phase: 'working'; added: number; total: number }
  | { phase: 'done'; playlist: ExportResult }
  /** Created, but short. Carries the link anyway -- the thing exists and is findable. */
  | { phase: 'partial'; playlist: ExportResult; detail: string }
  | { phase: 'error'; detail: string; needsReconnect: boolean }

export type PlaylistExport = {
  state: ExportState
  /** Never rejects. Every outcome lands in `state`, so a failure cannot be a no-op. */
  run: (input: Omit<ExportInput, 'onProgress'>) => void
  /**
   * Back to idle, abandoning any in-flight run's result.
   *
   * Callers use this when the selection changes underneath them. It does NOT cancel the
   * request -- the playlist the user already asked for may still finish being created,
   * which is correct. What it prevents is a success link for the previous selection
   * appearing beneath the current one.
   */
  reset: () => void
}

export function usePlaylistExport(): PlaylistExport {
  const [state, setState] = useState<ExportState>({ phase: 'idle' })

  // The panel unmounts the moment the selection is cleared, which can happen while a
  // request is in flight. Nothing should be written to a dead component's state.
  const alive = useRef(true)
  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  // Bumped by `reset` and by every `run`. A resolving promise writes only if its epoch is
  // still current.
  const epoch = useRef(0)

  // Guards a double-fire from keyboard activation that beats the disabled attribute.
  // Two exports SHOULD make two playlists -- but only when the user asked twice.
  const running = useRef(false)

  const run = useCallback((input: Omit<ExportInput, 'onProgress'>) => {
    if (running.current) return
    if (!configResult.ok) {
      setState({ phase: 'error', detail: configResult.detail, needsReconnect: false })
      return
    }

    const mine = ++epoch.current
    const current = () => alive.current && epoch.current === mine

    running.current = true
    setState({ phase: 'working', added: 0, total: input.uris.length })

    exportPlaylist(configResult.config, {
      ...input,
      onProgress: (added, total) => {
        if (current()) setState({ phase: 'working', added, total })
      },
    })
      .then((playlist) => {
        if (current()) setState({ phase: 'done', playlist })
      })
      .catch((err: unknown) => {
        if (!current()) return
        if (err instanceof PartialExportError) {
          setState({ phase: 'partial', playlist: err.result, detail: err.message })
          return
        }
        // A dead or missing session surfaces here as NotConnectedError. It is not a bug
        // to report, it is a reconnect to offer -- and never a silent no-op.
        setState({
          phase: 'error',
          detail: describe(err),
          needsReconnect: err instanceof NotConnectedError,
        })
      })
      .finally(() => {
        running.current = false
      })
  }, [])

  const reset = useCallback(() => {
    epoch.current += 1
    setState({ phase: 'idle' })
  }, [])

  return { state, run, reset }
}
