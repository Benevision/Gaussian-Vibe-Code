import os
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent
for p in (str(TOOLS), str(TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session")
def synth_small(tmp_path_factory):
    """Small orbit dataset with GT (shared across tests)."""
    from splatprep.synth import generate

    out = tmp_path_factory.mktemp("synth") / "orbit"
    generate(out, "orbit", num_views=8, width=200, height=150, seed=3, supersample=1, photometric_strength=0.3)
    return out


@pytest.fixture
def child_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(TOOLS), str(TESTS)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env
