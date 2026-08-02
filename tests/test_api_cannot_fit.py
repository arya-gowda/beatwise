"""The API must not be able to fit an embedding.

This is the guardrail the FastAPI decision made load-bearing (docs/decisions/0001).
A live service is a standing invitation to re-fit on request, and a re-fit silently
relocates every existing point -- scrambling learned territories and breaking saved
routes, with no error raised anywhere.

Enforced structurally rather than by convention: fitting lives only in
pipeline.build, and umap is imported only there. If someone wires the build path into
the API, umap appears in the API's import graph and this test fails.
"""

import subprocess
import sys

PROBE = """
import sys
import api.main  # noqa: F401

leaked = sorted(m for m in ("umap", "pipeline.build") if m in sys.modules)
print(",".join(leaked))
"""


def _probe():
    """Import the API in a clean interpreter and report forbidden modules.

    A subprocess is required: pytest itself may have imported umap for other tests,
    which would make an in-process check pass regardless of what the API does.
    """
    out = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True, text=True, cwd=".",
    )
    assert out.returncode == 0, f"importing api.main failed:\n{out.stderr}"
    return [m for m in out.stdout.strip().split(",") if m]


def test_api_does_not_import_the_fit_path():
    leaked = _probe()
    assert not leaked, (
        f"api.main pulled in {leaked}. Fitting must stay in the pipeline.build CLI -- "
        "the API may read an artifact and project through the persisted reducer, "
        "nothing more."
    )


def test_projection_is_importable_without_umap():
    """The read side must work without dragging in the fit machinery, otherwise the
    separation above is accidental rather than designed."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import pipeline.project; "
         "print('umap' in sys.modules)"],
        capture_output=True, text=True, cwd=".",
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "pipeline.project must not import umap"
