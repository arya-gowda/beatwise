"""P1-13 — is the genre gap empty AT SOURCE, or did the CSV export merely omit it?

    env/bin/python -m diagnostics.genre_gap_at_source

1,137 of 2,389 tracks carry no `Genres` value. §8 diagnoses that as Spotify genuinely
having no genres for those artists, and builds a tier 2-4 fetch cascade on top of the
diagnosis. If instead Spotify *does* hold genres and the export dropped them, the cascade
is solving a problem that does not exist. §8.1 calls this "worth one cheap test run to
confirm; not worth planning around" — so this is one cheap run, cached, and then read.

ONE-OFF, NOT A PIPELINE STAGE. Not imported by `api/`, not wired into the app, and it
writes nothing that any build reads. Genre labels decorate a point; they never move one.

THE TWO NUMBERS THAT MUST NOT BE CONFLATED
------------------------------------------
The CSV gives artist NAMES, not Spotify artist ids, so every artist needs a name→id
search before its genres can be read. A name search can return the wrong artist or none
at all, and "we could not find them" is NOT evidence that "Spotify has no genres for
them". So the sample splits three ways and is reported that way:

    resolution failures  — no search hit, or no exact name match among the hits
    resolved, empty      — we found the artist and Spotify's `genres` array is []
    resolved, populated  — we found the artist and Spotify HAS genres the CSV omitted

Only the third number answers the ticket. A large first number means the method was too
weak to answer, not that the gap is real.

ENDPOINTS — verified against the live reference pages on 2026-08-02, not from memory
------------------------------------------------------------------------------------
  * `POST https://accounts.spotify.com/api/token`, grant_type=client_credentials.
    App-only. Artist genres are public catalogue data and need no user context, so this
    never touches the browser's PKCE session (docs/decisions/0003-spotify-auth.md).
  * `GET /v1/search?type=artist` — live. `limit` maximum is **10**, default 5. Older docs
    and most tutorials say 50; that moved. Asking for more is a 400.
  * `GET /v1/artists/{id}` — live, no deprecation banner on the page.

  * `GET /v1/artists?ids=` is the batch call this would naturally use, and as of
    2026-08-02 its reference page carries an endpoint-level **Deprecated** tag. So the
    per-artist call is used instead: 100 requests rather than 2, which at this sample
    size is still nothing. Do not "optimise" it back.
  * The `genres` field itself is marked deprecated on the artist object (as are
    `followers` and `popularity`). It still exists and still answers this question, but
    it is on a path out — which is itself worth knowing before §8 plans four tiers of
    fetching around it. Nothing here depends on `followers` or `popularity`; when several
    artists share a name exactly, search relevance order breaks the tie and the ambiguity
    is recorded rather than hidden.

CREDENTIALS
-----------
Read from the REPO-ROOT `.env` (gitignored) — `SPOTIFY_CLIENT_ID` and
`SPOTIFY_CLIENT_SECRET`. `VITE_`-prefixed names are never read: Vite inlines those into
the browser bundle at build time, so a secret in one would be published. The reader below
is deliberately a whitelist rather than a general dotenv load, so a `VITE_SPOTIFY_SECRET`
sitting in the same file could not be picked up even by accident. The secret is never
printed and never written to the cache.
"""

import argparse
import base64
import hashlib
import json
import os
import random
import re
import ssl
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ml.data import split_artists
from pipeline import artifact, features, genres

REPO = Path(__file__).resolve().parent.parent
ENV_PATH = REPO / ".env"

# Bump when the cached record shape changes. Folded into the run id, so a cache written
# under different rules is a different directory rather than a silently mixed one.
# 2: added the positive control, without which the headline number is uninterpretable.
SCHEMA_VERSION = 2

# Pinned so the sample is the same 100 artists on every machine and every re-run. An
# unpinned sample would make two runs disagree for a reason nobody could reconstruct.
SEED = 42
SAMPLE_SIZE = 100

# Artists we KNOW have genres, fetched the same way. Ten is enough to tell "these
# particular artists have none" from "the field returns nothing for anybody"; a bigger
# control would only buy precision on a question that is not being asked.
CONTROL_SIZE = 10
# Below this share of the resolved control coming back populated, the field is not
# answering and the main number means nothing.
CONTROL_MIN_RATE = 0.7

# "Materially more than a handful" (ticket wording) resolved to a number, stated up front
# rather than eyeballed after the fact. Above this, §8's diagnosis is wrong.
HANDFUL = 5

TOKEN_URL = "https://accounts.spotify.com/api/token"
API = "https://api.spotify.com/v1"

# Verified 2026-08-02: maximum 10. Not 50.
SEARCH_LIMIT = 10

REQUIRED_ENV = ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET")

CACHE_ROOT = artifact.ARTIFACTS / "diagnostics" / "genre-gap"
RESULT = "result.json"
PARTIAL = "partial.jsonl"
CONTROL_PARTIAL = "control-partial.jsonl"

RESOLVED = "resolved"
NOT_FOUND = "not_found"
NO_EXACT_MATCH = "no_exact_match"

_WHITESPACE = re.compile(r"\s+")


# --- credentials ----------------------------------------------------------------------


class MissingCredentials(RuntimeError):
    pass


def read_env_file(path=ENV_PATH, wanted=REQUIRED_ENV):
    """The two names we need, and nothing else, from one `KEY=VALUE` file.

    A whitelist rather than a general loader on purpose: this function cannot return a
    `VITE_`-prefixed value however the file is edited, because it never looks at one.
    """
    values = {}
    if not Path(path).is_file():
        return values
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if key not in wanted:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            values[key] = value
    return values


def load_credentials(path=ENV_PATH, environ=None):
    """(client_id, client_secret), or a message naming exactly what to set and where.

    Half-running is the failure mode to avoid here: a partial fetch would produce a
    number that looks like an answer and is not one.
    """
    environ = os.environ if environ is None else environ
    values = read_env_file(path)
    for name in REQUIRED_ENV:
        if not values.get(name):
            values[name] = environ.get(name, "").strip()

    missing = [name for name in REQUIRED_ENV if not values.get(name)]
    if missing:
        raise MissingCredentials(
            f"missing {', '.join(missing)}.\n\n"
            f"Set them in {Path(path)} (gitignored), one per line:\n"
            "    SPOTIFY_CLIENT_ID=<your app's client id>\n"
            "    SPOTIFY_CLIENT_SECRET=<your app's client secret>\n\n"
            "Both are on the app's page at https://developer.spotify.com/dashboard "
            "(the secret is behind 'View client secret').\n\n"
            "Do NOT reuse a VITE_-prefixed name for either. Vite inlines those into the "
            "browser bundle, so a secret in one would be published. The browser app's "
            "PKCE flow has no secret and must keep it that way — "
            "docs/decisions/0003-spotify-auth.md."
        )
    return values["SPOTIFY_CLIENT_ID"], values["SPOTIFY_CLIENT_SECRET"]


# --- HTTP -----------------------------------------------------------------------------


class SpotifyError(RuntimeError):
    def __init__(self, status, detail):
        super().__init__(f"Spotify returned {status}: {detail}")
        self.status = status


def ssl_context():
    """Verified TLS, with a fallback for a framework Python that has no CA bundle.

    This venv's interpreter looks for its trust store under the python.org framework's
    own `etc/openssl`, which only exists if "Install Certificates.command" was ever run —
    it was not here, and every request fails CERTIFICATE_VERIFY_FAILED. macOS ships a
    bundle at /etc/ssl/cert.pem, so verification stays ON and points at that instead.

    The shortcut everyone reaches for at this error is an unverified context. On a request
    that carries a client secret in an Authorization header, that would hand the secret to
    anything that can intercept the connection. Not done, and asserted against in
    tests/test_genre_gap_diagnostic.py.
    """
    context = ssl.create_default_context()
    if context.cert_store_stats().get("x509_ca"):
        return context
    for candidate in ("/etc/ssl/cert.pem", "/usr/local/etc/openssl/cert.pem",
                      "/opt/homebrew/etc/openssl@3/cert.pem"):
        if Path(candidate).is_file():
            context.load_verify_locations(cafile=candidate)
            return context
    raise SpotifyError(
        "tls",
        "no CA bundle found, so certificates cannot be verified. On macOS run "
        "'Install Certificates.command' from your Python install, or point SSL_CERT_FILE "
        "at a bundle. Verification is not being disabled — this call carries a secret.",
    )


def _retry_after(headers, default=1.0):
    """Seconds to wait, from the documented `Retry-After` header.

    Spotify sends it on 429 and expects it honoured. Guessing a backoff instead is how an
    app earns a longer ban than the one it was given.
    """
    raw = headers.get("Retry-After") if headers else None
    try:
        return max(float(raw), 0.0) + 1.0     # +1s of slack; the header is whole seconds
    except (TypeError, ValueError):
        return default


def call(url, headers, data=None, attempts=6, sleep=time.sleep):
    """One JSON call, honouring 429 and retrying transient 5xx.

    Errors carry the status and Spotify's own error text only. The request headers — which
    hold the credential — are never included in an exception message, because exception
    messages end up in logs.
    """
    last = None
    context = ssl_context()
    for attempt in range(attempts):
        request = Request(url, data=data, headers=headers,
                          method="POST" if data is not None else "GET")
        try:
            with urlopen(request, timeout=30, context=context) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code == 429:
                wait = _retry_after(error.headers)
                print(f"    429 — waiting {wait:.0f}s as instructed", flush=True)
                sleep(wait)
                last = error
                continue
            if error.code in (500, 502, 503, 504):
                sleep(2 ** attempt)
                last = error
                continue
            raise SpotifyError(error.code, _error_text(error)) from None
        except URLError as error:
            # A TLS failure is a configuration problem, not a flaky link. Retrying it six
            # times only buys a minute of waiting before the same message.
            if isinstance(error.reason, ssl.SSLError):
                raise SpotifyError("tls", error.reason) from None
            last = error
            sleep(2 ** attempt)
    raise SpotifyError(getattr(last, "code", "network"), "gave up after retries")


def _error_text(error):
    try:
        body = json.loads(error.read().decode("utf-8"))
    except Exception:
        return error.reason
    inner = body.get("error", body)
    if isinstance(inner, dict):
        return inner.get("message") or inner.get("error_description") or str(inner)
    return str(body.get("error_description") or inner)


def app_token(client_id, client_secret):
    """Client Credentials flow. App-only; no user, no scopes, no refresh token."""
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    payload = call(
        TOKEN_URL,
        {"Authorization": f"Basic {basic}",
         "Content-Type": "application/x-www-form-urlencoded"},
        data=urlencode({"grant_type": "client_credentials"}).encode(),
    )
    token = payload.get("access_token")
    if not token:
        raise SpotifyError("token", "no access_token in the token response")
    return token


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


# --- the population -------------------------------------------------------------------


def match_key(name):
    """Comparison form for a name. NFKC-composed, whitespace-collapsed, casefolded.

    Composed for the same reason P1-11 curates against composed genre tokens: a decomposed
    accent is a different string with an identical glyph, and "Beyoncé" from the CSV
    would silently fail to match "Beyoncé" from the API.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", str(name))).strip().casefold()


def empty_artists(csv_path):
    """Artists the export attributes NO genre to, plus the counts that frame them.

    An artist is in the population when every row they are credited on has an empty
    `Genres` cell. The cell is the union over the credited artists, so an empty cell means
    each of them contributed nothing — but an artist credited on an empty row AND on a
    labelled one is ambiguous (the labels may belong entirely to their collaborator), and
    those are excluded rather than guessed at. That exclusion is the difference between
    718 and 635 below, and it is reported either way.

    Uses ml.data.split_artists for the SEMICOLON split and pipeline.genres.normalise_labels
    for the comma-separated genre cell. Two columns, two conventions, no third splitter.
    """
    scan = _scan(csv_path)
    population = sorted(scan["on_empty"] - scan["on_labelled"])
    counts = {
        "n_tracks": scan["n_tracks"],
        "n_dropped": scan["n_dropped"],
        "n_empty_genre_rows": scan["n_empty_rows"],
        "n_artists_total": len(scan["all_names"]),
        "n_artists_on_an_empty_row": len(scan["on_empty"]),
        "n_artists_on_both": len(scan["on_empty"] & scan["on_labelled"]),
        "n_population": len(population),
    }
    return population, counts


def control_artists(csv_path):
    """The POSITIVE CONTROL population: artists solely credited on a labelled row.

    Without this the headline number is uninterpretable. `genres` is a deprecated field on
    the artist object, so "Spotify returns [] for these 100 artists" and "Spotify has
    started returning [] for everyone" produce the identical measurement. These artists
    demonstrably have genres — their names sit alone on a row whose `Genres` cell is
    populated, so the labels cannot belong to a collaborator — and the API must return
    them. If it does not, the field has gone dark and the run answers nothing.
    """
    return sorted(_scan(csv_path)["solo_labelled"])


def _scan(csv_path):
    """One pass over the CSV, feeding both populations above."""
    df, dropped = features.load_features(csv_path)

    on_empty, on_labelled, all_names, solo_labelled = set(), set(), set(), set()
    empty_rows = 0
    for credits, cell in zip(df["Artist Name(s)"], df["Genres"], strict=True):
        names = split_artists(credits)
        all_names.update(names)
        if genres.normalise_labels(cell):
            on_labelled.update(names)
            if len(names) == 1:
                solo_labelled.add(names[0])
        else:
            empty_rows += 1
            on_empty.update(names)

    return {
        "n_tracks": len(df), "n_dropped": dropped, "n_empty_rows": empty_rows,
        "all_names": all_names, "on_empty": on_empty, "on_labelled": on_labelled,
        "solo_labelled": solo_labelled - on_empty,
    }


def sample(population, size=SAMPLE_SIZE, seed=SEED):
    """A deterministic subset. `population` is already sorted, so this is reproducible
    across processes — set iteration order is not, and would quietly give two runs two
    different samples."""
    if size >= len(population):
        return list(population)
    return sorted(random.Random(seed).sample(population, size))


# --- resolution -----------------------------------------------------------------------


def search_artists(token, name):
    query = urlencode({"q": name, "type": "artist", "limit": SEARCH_LIMIT})
    payload = call(f"{API}/search?{query}", _auth(token))
    return (payload.get("artists") or {}).get("items") or []


def resolve(name, candidates):
    """Classify one name→id attempt. Never invents a match.

    Exact name equality under match_key only. A fuzzy or first-hit-wins rule would
    manufacture resolutions, and every manufactured resolution lands in the "resolved,
    empty" bucket — the exact contamination that would make a real gap look bigger than
    it is.
    """
    if not candidates:
        return {"outcome": NOT_FOUND, "n_candidates": 0, "candidates": []}

    # `None` where the search hit carried no `genres` key at all -- see fetch_one on why
    # absent must not be flattened into empty.
    seen = [{"id": c.get("id"), "name": c.get("name"),
             "search_genres": _field(c, "genres")} for c in candidates[:5]]
    key = match_key(name)
    exact = [c for c in candidates if match_key(c.get("name", "")) == key]
    if not exact:
        return {"outcome": NO_EXACT_MATCH, "n_candidates": len(candidates),
                "candidates": seen}

    chosen = exact[0]     # search relevance order; ties recorded, not silently broken
    others_with_genres = sum(1 for c in exact[1:] if c.get("genres"))
    return {
        "outcome": RESOLVED,
        "artist_id": chosen.get("id"),
        "matched_name": chosen.get("name"),
        "n_candidates": len(candidates),
        "n_exact_matches": len(exact),
        "ambiguous": len(exact) > 1,
        # A same-named artist we did NOT pick that does have genres. Not counted as a hit
        # — it is probably a different act — but a run full of these would mean the
        # tie-break is doing real work and the answer needs a second look.
        "other_exact_matches_with_genres": others_with_genres,
        "search_genres": _field(chosen, "genres"),
        "candidates": seen,
    }


def _field(payload, name):
    """The list at `name`, or None when the key is absent. Never [] for a missing key."""
    return list(payload[name] or []) if name in payload else None


def fetch_one(token, name):
    """Search, then read the artist. Returns a cache-safe record — names, ids, genres.

    Deliberately field-by-field rather than dumping the raw response: it keeps the cache
    small and readable, and it makes "no auth material is written" a property of the code
    rather than something to re-check by eye after every run.
    """
    record = {"csv_name": name}
    record.update(resolve(name, search_artists(token, name)))

    if record["outcome"] != RESOLVED:
        record["genres_field_present"] = None
        record["genres"] = None
        record["has_genres"] = None
        return record

    payload = call(f"{API}/artists/{record['artist_id']}", _auth(token))
    record["artist_name"] = payload.get("name")

    # ABSENT IS NOT EMPTY, and collapsing the two is the same class of mistake as
    # collapsing "we could not find them" into "they have no genres". `payload.get(...)
    # or []` would turn a field Spotify no longer sends into a confident measurement of
    # zero genres — which reads as a clean confirmation of §8 while measuring nothing.
    present = "genres" in payload
    record["genres_field_present"] = present
    record["genres"] = list(payload.get("genres") or []) if present else None
    record["has_genres"] = bool(record["genres"]) if present else None

    # The artist object and the search hit both carry `genres` when it exists. A
    # disagreement would mean one of the two is a simplified object.
    record["search_field_present"] = record.get("search_genres") is not None
    record["search_agrees"] = (
        sorted(record.get("search_genres") or []) == sorted(record["genres"] or [])
        if present else None
    )
    return record


# --- cache ----------------------------------------------------------------------------


def run_id(csv_hash, size, seed):
    """Stable directory name. No timestamp, so a re-run FINDS the cache rather than
    writing a second copy beside it. Inputs that would change the answer are in the
    digest, so changing the sample size is a new run rather than a silent overwrite."""
    config = json.dumps({"schema": SCHEMA_VERSION, "csv": csv_hash,
                         "size": size, "seed": seed}, sort_keys=True)
    return hashlib.sha256(config.encode()).hexdigest()[:12]


def _read_partial(path):
    if not path.is_file():
        return {}
    done = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            done[record["csv_name"]] = record
    return done


def _fetch_all(token, names, partial):
    """Fetch each name, appending as it goes so an interrupted run resumes for free
    rather than re-spending the calls it already made."""
    done = _read_partial(partial)
    if done:
        print(f"  resuming — {len(done)} already fetched")
    with open(partial, "a", encoding="utf-8") as handle:
        for index, name in enumerate(names, 1):
            if name not in done:
                done[name] = fetch_one(token, name)
                handle.write(json.dumps(done[name], ensure_ascii=False) + "\n")
                handle.flush()
            if index % 10 == 0 or index == len(names):
                print(f"  {index}/{len(names)}", flush=True)
    return [done[name] for name in names]


# --- the count ------------------------------------------------------------------------


def tally(records):
    """The numbers, kept separate on purpose. See the module docstring.

    Three outcomes for a resolved artist, not two: Spotify said "no genres", Spotify said
    "these genres", or Spotify did not send the field at all. The third is not a small
    print — a run where it is the only outcome has measured nothing.
    """
    resolved = [r for r in records if r["outcome"] == RESOLVED]
    return {
        "sampled": len(records),
        "resolution_failures": len(records) - len(resolved),
        "not_found": sum(1 for r in records if r["outcome"] == NOT_FOUND),
        "no_exact_match": sum(1 for r in records if r["outcome"] == NO_EXACT_MATCH),
        "resolved": len(resolved),
        "genres_field_present": sum(1 for r in resolved if r.get("genres_field_present")),
        "resolved_with_genres": sum(1 for r in resolved if r["has_genres"] is True),
        "resolved_empty": sum(1 for r in resolved if r["has_genres"] is False),
        "resolved_field_absent": sum(1 for r in resolved if r["has_genres"] is None),
        "ambiguous_resolutions": sum(1 for r in resolved if r.get("ambiguous")),
        "unpicked_same_name_with_genres":
            sum(1 for r in resolved if r.get("other_exact_matches_with_genres")),
        "search_vs_artist_disagreements":
            sum(1 for r in resolved if r.get("search_agrees") is False),
    }


def control_holds(control):
    """Did the known-good artists come back with genres? If not, nothing else counts."""
    if not control or not control["resolved"]:
        return False
    return control["resolved_with_genres"] >= CONTROL_MIN_RATE * control["resolved"]


def verdict(counts, control=None):
    """A sentence and a flag, decided by the thresholds declared at the top of this file
    rather than by whoever reads the output."""
    populated = counts["resolved_with_genres"]
    resolved = counts["resolved"]

    # Checked first because it is the most specific and the most actionable: if Spotify
    # is not sending the field, no threshold below is measuring anything.
    if resolved and not counts.get("genres_field_present"):
        return ("field_removed",
                f"the artist object came back with NO `genres` key at all for all "
                f"{resolved} resolved artists, the positive control included. The field "
                "is not empty — it is gone for this application. Spotify cannot answer "
                "whether it holds genres for these artists, so §8's premise is neither "
                "confirmed nor refuted, and tier 2 (fetch artist genres from Spotify) is "
                "IMPOSSIBLE rather than unnecessary")

    if control is not None and not control_holds(control):
        return ("method_void",
                f"the positive control failed — only {control['resolved_with_genres']} of "
                f"{control['resolved']} artists the CSV DOES have genres for came back "
                "populated. `genres` is a deprecated field; if it has gone dark, an empty "
                "array proves nothing about any artist and this run answers nothing")

    if resolved == 0:
        return ("inconclusive",
                "nothing resolved — the method failed, and this says nothing at all "
                "about whether Spotify has genres for these artists")
    if populated == 0:
        return ("gap_is_real",
                f"0 of {resolved} resolved artists have genres at Spotify. The gap is "
                "empty at source; §8's diagnosis holds and the tier 2-4 cascade is "
                "addressing a real absence")
    if populated <= HANDFUL:
        return ("gap_is_real",
                f"{populated} of {resolved} resolved artists have genres at Spotify — "
                f"at or under the {HANDFUL}-artist threshold §8.1 calls a handful. The "
                "gap is substantially real; §8's diagnosis holds")
    return ("diagnosis_wrong",
            f"{populated} of {resolved} resolved artists DO have genres at Spotify, "
            f"above the {HANDFUL}-artist threshold. The CSV export dropped them. §8's "
            "premise that the gap is empty at source is WRONG, and a plain artist-genre "
            "fetch may replace most of the tier 2-4 cascade")


# --- run ------------------------------------------------------------------------------


def run(csv_path="Liked_Songs.csv", size=SAMPLE_SIZE, seed=SEED, refresh=False,
        user_id="local"):
    population, counts = empty_artists(csv_path)
    csv_hash = features.source_hash(csv_path)
    identifier = run_id(csv_hash, size, seed)
    out = CACHE_ROOT / identifier
    result_path = out / RESULT

    if result_path.is_file() and not refresh:
        print(f"cached — {result_path} (0 API calls)")
        return json.loads(result_path.read_text(encoding="utf-8"))

    # Before anything is created or fetched. A missing secret must leave no trace and no
    # half-run: a partially populated cache would read as an answer on the next run.
    client_id, client_secret = load_credentials()

    chosen = sample(population, size, seed)
    print(f"{counts['n_empty_genre_rows']} rows with no genre; "
          f"{counts['n_population']} artists never attributed one "
          f"({counts['n_artists_on_both']} more are ambiguous and excluded)")
    print(f"sampling {len(chosen)} of them (seed {seed})")

    controls = sample(control_artists(csv_path), CONTROL_SIZE, seed)

    out.mkdir(parents=True, exist_ok=True)
    token = app_token(client_id, client_secret)

    print(f"positive control: {len(controls)} artists the CSV DOES have genres for")
    control_records = _fetch_all(token, controls, out / CONTROL_PARTIAL)
    control_counts = tally(control_records)
    print(f"  control: {control_counts['resolved_with_genres']}/"
          f"{control_counts['resolved']} resolved came back with genres")

    records = _fetch_all(token, chosen, out / PARTIAL)
    tallied = tally(records)
    flag, sentence = verdict(tallied, control_counts)

    result = {
        "diagnostic": "p1-13-genre-gap-at-source",
        "schema_version": SCHEMA_VERSION,
        "run_id": identifier,
        "user_id": user_id,          # one user today; carried so multi-user is not a rewrite
        "source_csv": str(csv_path),
        "source_sha256": csv_hash,
        "seed": seed,
        "sample_size": len(chosen),
        "handful_threshold": HANDFUL,
        "population": counts,
        "endpoints": {
            "token": TOKEN_URL,
            "resolve": f"{API}/search?type=artist (limit max {SEARCH_LIMIT})",
            "genres": f"{API}/artists/{{id}}",
            "not_used": "GET /v1/artists?ids= — endpoint-level deprecated 2026-08-02",
        },
        "counts": tallied,
        "control": {
            "size": len(controls),
            "min_rate": CONTROL_MIN_RATE,
            "counts": control_counts,
            "holds": control_holds(control_counts),
            "artists": control_records,
        },
        "verdict": flag,
        "verdict_text": sentence,
        "artists": records,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    (out / PARTIAL).unlink(missing_ok=True)
    (out / CONTROL_PARTIAL).unlink(missing_ok=True)
    print(f"wrote {result_path}")
    return result


def report(result):
    counts = result["counts"]
    print()
    print(f"  sampled                {counts['sampled']:4d}")
    print(f"  resolution failures    {counts['resolution_failures']:4d}"
          f"   ({counts['not_found']} no search hit, "
          f"{counts['no_exact_match']} no exact name match)")
    print(f"  resolved               {counts['resolved']:4d}")
    print(f"    -> no genres         {counts['resolved_empty']:4d}"
          "   genuinely empty at Spotify")
    print(f"    -> HAS genres        {counts['resolved_with_genres']:4d}"
          "   the CSV omitted them")
    print(f"    -> field absent      {counts['resolved_field_absent']:4d}"
          "   Spotify sent no `genres` key — unmeasurable")
    print(f"  ambiguous resolutions  {counts['ambiguous_resolutions']:4d}")
    print(f"  search/artist disagree {counts['search_vs_artist_disagreements']:4d}")

    control = result.get("control")
    if control:
        held = control["counts"]
        print()
        print(f"  positive control       {held['resolved_with_genres']:4d}"
              f" of {held['resolved']} known-genre artists came back populated"
              f"   [{'OK' if control['holds'] else 'FAILED'}]")
    print()

    if result["verdict"] in ("diagnosis_wrong", "method_void", "field_removed"):
        # Loud on purpose. Every one of these invalidates a premise later tickets are
        # built on, and the failure mode the ticket names is that it gets buried in a log.
        bar = "!" * 78
        headline = {
            "diagnosis_wrong": "STOP — §8's DIAGNOSIS IS WRONG",
            "field_removed": "STOP — SPOTIFY NO LONGER SENDS ARTIST GENRES",
            "method_void": "STOP — THIS RUN MEASURED NOTHING",
        }[result["verdict"]]
        print(bar)
        print(f"!! {headline}")
        print(f"!! {result['verdict_text']}")
        print("!! Re-open §8 before P1 plans any tier 2-4 work.")
        print(bar)
    else:
        print(f"  {result['verdict_text']}.")

    example = next((r for r in result["artists"] if r.get("has_genres")), None)
    if example:
        print(f"  e.g. {example['csv_name']} -> {example['genres']}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="P1-13: does Spotify hold genres for the artists our CSV left blank?")
    parser.add_argument("source", nargs="?", default="Liked_Songs.csv")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--refresh", action="store_true",
                        help="ignore the cache and spend the API calls again")
    parser.add_argument("--user-id", default="local")
    args = parser.parse_args(argv)

    try:
        result = run(args.source, size=args.sample_size, seed=args.seed,
                     refresh=args.refresh, user_id=args.user_id)
    except MissingCredentials as error:
        print(f"cannot run: {error}", file=sys.stderr)
        return 2
    except SpotifyError as error:
        print(f"cannot run: {error}", file=sys.stderr)
        return 3

    report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
