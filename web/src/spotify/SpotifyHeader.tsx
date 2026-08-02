/**
 * The header's account corner: connect, who you are, disconnect.
 *
 * Sits top-left. Top-right belongs to the selection panel (P1-05), and the two must not
 * fight over the same corner. Visual treatment beyond staying out of the way is the
 * ux-designer's call, not this file's.
 */
import { useState } from 'react'
import { useSpotifyAuth } from './useSpotifyAuth'
import './spotify.css'

const ACCOUNT_APPS_URL = 'https://www.spotify.com/account/apps/'

export default function SpotifyHeader() {
  const auth = useSpotifyAuth()
  const [showRevokeHint, setShowRevokeHint] = useState(false)

  if (auth.status === 'unconfigured' || auth.status === 'error') {
    return (
      <header className="account account--problem">
        <span className="account__label">
          {auth.status === 'unconfigured' ? 'Spotify not configured' : 'Spotify connection failed'}
        </span>
        <p className="account__detail">{auth.detail}</p>
        {auth.status === 'error' && (
          <button type="button" className="account__button" onClick={auth.connect}>
            Try again
          </button>
        )}
      </header>
    )
  }

  if (auth.status === 'connecting') {
    return (
      <header className="account">
        <span className="account__label">connecting to Spotify…</span>
      </header>
    )
  }

  if (auth.status === 'connected') {
    const { profile } = auth
    // Spotify permits a null display name. Falling back to the id is better than a blank
    // header claiming to have signed someone in.
    const name = profile.displayName?.trim() || profile.id
    return (
      <header className="account">
        {profile.imageUrl ? (
          <img className="account__avatar" src={profile.imageUrl} alt="" width={24} height={24} />
        ) : (
          <span className="account__avatar account__avatar--blank" aria-hidden="true" />
        )}
        <span className="account__name" title={profile.id}>
          {name}
        </span>
        <button
          type="button"
          className="account__button"
          onClick={() => {
            auth.disconnect()
            setShowRevokeHint(true)
          }}
        >
          Disconnect
        </button>
      </header>
    )
  }

  return (
    <header className="account">
      <button type="button" className="account__button account__button--primary" onClick={auth.connect}>
        Connect Spotify
      </button>
      {showRevokeHint && (
        // Honest about the limit of the button that was just pressed: the tokens are
        // gone from this browser, but Spotify publishes no revocation endpoint, so the
        // grant itself survives until the user removes it on their account page.
        <p className="account__detail">
          tokens cleared.{' '}
          <a href={ACCOUNT_APPS_URL} target="_blank" rel="noreferrer">
            remove the app from your account
          </a>{' '}
          to revoke the grant itself.
        </p>
      )}
    </header>
  )
}
