/**
 * Endpoints Beatwise must never call.
 *
 * Spotify withdrew these from apps registered on or after 2024-11-27, and Beatwise is a
 * new app -- so they are not "discouraged", they are gone. This is the reason the
 * substrate is an uploaded CSV rather than a live fetch (0001, concept doc §5), and the
 * reason no fallback path exists against them anywhere in this codebase. Such a path
 * would be dead code the day it was written.
 *
 * There is a second, stronger reason to keep the guard rather than trust discipline. Two
 * of these names describe things a future contributor will genuinely want -- "just fetch
 * the audio features for the new tracks", "just ask for recommendations near this
 * region". Both would pull live API data into the layer that decides where points sit,
 * which is precisely the split the two-layer model exists to prevent.
 *
 * THIS FILE IS THE ONLY PLACE THESE NAMES MAY APPEAR IN web/src.
 * tests/test_no_deprecated_spotify_endpoints.py enforces that.
 *
 * A second list follows, for the February 2026 replacements. Same enforcement, different
 * reason -- see its own comment.
 */

export const DEPRECATED_PATHS = ['audio-features', 'audio-analysis', 'recommendations'] as const

/**
 * Endpoints Spotify replaced in February 2026, and what replaced them.
 *
 * A different failure from the list above, worth its own guard. These have a working
 * successor one word away, so the mistake is not "reaching for something gone" -- it is
 * writing the name everyone already has in their fingers and getting a deprecation, or a
 * 404, at runtime instead of at review. P1-07 depends on exactly these two paths, and the
 * old spellings are the ones a decade of tutorials teach.
 *
 * Verified against the live reference pages on 2026-08-02, not only the changelog.
 */
export const REPLACED_PATHS = [
  {
    pattern: /^\/playlists\/[^/?#]+\/tracks(?:[/?#]|$)/,
    use: 'POST /playlists/{playlist_id}/items',
  },
  {
    pattern: /^\/users\/[^/?#]+\/playlists(?:[/?#]|$)/,
    use: 'POST /me/playlists',
  },
] as const

/** Throws before the network call if `path` touches a withdrawn endpoint. */
export function assertNotDeprecated(path: string): void {
  const hit = DEPRECATED_PATHS.find((name) => path.includes(name))
  if (hit !== undefined) {
    throw new Error(
      `refusing to call "${hit}": withdrawn by Spotify for apps registered after ` +
        '2024-11-27, and audio features must come from the substrate, not the live layer. ' +
        'See docs/decisions/0003-spotify-auth.md.',
    )
  }

  const replaced = REPLACED_PATHS.find((entry) => entry.pattern.test(path))
  if (replaced !== undefined) {
    throw new Error(
      `refusing to call "${path}": replaced by Spotify in February 2026. ` +
        `Use ${replaced.use} instead.`,
    )
  }
}
