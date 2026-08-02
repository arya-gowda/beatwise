"""Beatwise API.

Serves the map artifact to the web app. Deliberately read-only with respect to the
embedding: see the module docstring in `pipeline/` for why this service must never fit
a reducer. Building an embedding is a CLI operation; this process only reads what the
CLI produced and projects new tracks through the persisted transform.
"""

from fastapi import FastAPI

app = FastAPI(title="Beatwise API", version="0.1.0")


@app.get("/health")
def health():
    """Liveness probe. The web app calls this on load to prove the seam works."""
    return {"status": "ok", "service": "beatwise-api"}
