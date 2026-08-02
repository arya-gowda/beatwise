"""Render an artifact so a human can look at it.

    python -m pipeline.preview

Reads the artifact rather than recomputing, so what you see is exactly what the web app
will receive -- if the preview and the browser ever disagree, the bug is in the browser.
"""

import argparse

import pandas as pd
import plotly.express as px

from . import artifact


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=None)
    ap.add_argument("--color", default="popularity")
    args = ap.parse_args()

    version = args.version or artifact.latest_version()
    if version is None:
        raise SystemExit("no artifact found -- run: python -m pipeline.build")

    manifest = artifact.load_manifest(version)
    df = pd.DataFrame(artifact.load_points(version))

    fig = px.scatter(
        df, x="x", y="y", color=args.color,
        color_continuous_scale="Viridis",
        hover_data=["name", "artists", "tempo", "popularity"],
        title=f"Beatwise map — {version} ({manifest['n_tracks']} tracks, "
              f"key {manifest['key_encoding']})",
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0)))
    fig.update_layout(template="plotly_dark", height=800)
    fig.show()


if __name__ == "__main__":
    main()
