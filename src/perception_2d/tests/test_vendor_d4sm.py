"""The vendored DAM4SAM model class must stay byte-identical to the authors'."""

import hashlib
from pathlib import Path

import wato_perception_2d._vendor.d4sm as d4sm

# _vendor/d4sm/README.md: sha256 of sam2/modeling/sam2_base.py at the pinned
# commit. A changed file is no longer the authors' memory policy.
UPSTREAM_SAM2_BASE_SHA256 = (
    "3ec3600badfb8465471276e4e9fe0adb9ccfce920cb328dae4da8c0bf014afe6"
)


def test_sam2_base_is_the_authors_file():
    path = Path(d4sm.__file__).parent / "sam2_base.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == UPSTREAM_SAM2_BASE_SHA256
    readme = (Path(d4sm.__file__).parent / "README.md").read_text()
    assert UPSTREAM_SAM2_BASE_SHA256 in readme
    assert d4sm.COMMIT in readme  # one pin, stated where the source is
