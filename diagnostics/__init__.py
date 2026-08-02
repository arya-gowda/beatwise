"""One-off diagnostics.

Nothing here is a pipeline stage, nothing here is imported by `api/` or `web/`, and
nothing here can move a point on the map. These modules answer a question once, cache the
answer, and are then read rather than re-run. Keeping them out of `pipeline/` is the
point: a diagnostic that quietly becomes a build step is how live-layer data gets into the
substrate.
"""
