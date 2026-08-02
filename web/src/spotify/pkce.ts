/**
 * PKCE primitives (RFC 7636, as Spotify implements it).
 *
 * Deliberately dependency-free and pure: everything here is WebCrypto, which is available
 * because `http://127.0.0.1` counts as a secure context even over plain HTTP.
 */

/**
 * The RFC 7636 unreserved set: ALPHA / DIGIT / "-" / "." / "_" / "~".
 * Exactly 64 characters, which matters -- 256 is divisible by 64, so indexing random
 * bytes into it is uniform. A 62- or 66-character alphabet would quietly bias the
 * verifier toward its first few symbols.
 */
const UNRESERVED = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~'

/** A high-entropy code verifier. Spotify permits 43-128 characters; 96 sits comfortably. */
export function createVerifier(length = 96): string {
  if (length < 43 || length > 128) {
    throw new Error(`code verifier must be 43-128 characters, got ${length}`)
  }
  const bytes = new Uint8Array(length)
  crypto.getRandomValues(bytes)
  let out = ''
  for (const b of bytes) out += UNRESERVED[b % UNRESERVED.length]
  return out
}

/** S256 challenge: SHA-256 of the ASCII verifier, base64url, padding stripped. */
export async function challengeFor(verifier: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier))
  return base64url(digest)
}

/** Opaque CSRF value round-tripped through Spotify as `state`. */
export function createState(): string {
  const bytes = new Uint8Array(16)
  crypto.getRandomValues(bytes)
  return base64url(bytes.buffer)
}

function base64url(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (const b of bytes) binary += String.fromCharCode(b)
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}
