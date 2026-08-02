"""Beatwise API.

Serves the map artifact to the web app. Deliberately read-only with respect to the
embedding: this process imports `pipeline.artifact` and `pipeline.project`, neither of
which can fit a reducer, and `tests/test_api_cannot_fit.py` asserts that umap never
enters this module's import graph.

The reason is concept doc §5, seam item 3. A live service is a standing invitation to
re-fit on request, and a re-fit silently relocates every existing point — scrambling
territories the user has learned and breaking saved routes, with nothing reporting an
error. Building an embedding is a deliberate offline act: `python -m pipeline.build`.
"""

from fastapi import FastAPI, HTTPException

from pipeline import artifact

app = FastAPI(title="Beatwise API", version="0.1.0")


@app.get("/health")
def health():
    """Liveness probe. The web app calls this on load to prove the seam works."""
    return {"status": "ok", "service": "beatwise-api"}


@app.get("/map")
def get_map(version: str | None = None):
    """The full point cloud plus its manifest.

    2,389 points is small enough to send in one response; at Phase 4's corpus scale this
    becomes a tiled or binary endpoint, which is why the version is explicit in the
    payload rather than implied.
    """
    version = version or artifact.latest_version()
    if version is None:
        raise HTTPException(
            status_code=503,
            detail="no embedding artifact found — run: python -m pipeline.build",
        )
    try:
        return {
            "version": version,
            "manifest": artifact.load_manifest(version),
            "points": artifact.load_points(version),
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"unknown artifact version {version}")
