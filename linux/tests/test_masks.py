"""Mask files in dataset.json reach the trainer as ignore masks (until 1.3.14 the worker dropped them)."""
import json

import numpy as np
from PIL import Image

from conftest import make_dataset
from splatworker.dataset import load_view, parse_dataset


def test_mask_file_is_loaded_and_resized(tmp_path):
    root = make_dataset(tmp_path / "ds", n_cams=3, width=64, height=48, depth=False)
    doc = json.loads((root / "dataset.json").read_text())
    (root / "masks").mkdir()
    m = np.full((48, 64), 255, np.uint8)
    m[:, :32] = 0                                          # left half: a person to ignore
    Image.fromarray(m).save(root / "masks" / "000000.png")
    doc["cameras"][0]["mask"] = "masks/000000.png"
    doc["cameras"][0]["sky"] = "masks/000000.png"
    (root / "dataset.json").write_text(json.dumps(doc))
    ds = parse_dataset(root)
    assert ds.cameras[0].mask == "masks/000000.png" and ds.cameras[0].sky == "masks/000000.png"
    v = load_view(ds, ds.cameras[0], 32)                  # downscaled 64×48 → 32×24
    assert v.mask is not None and v.mask.shape == (24, 32)
    assert v.mask[:, :14].max() == 0 and v.mask[:, 18:].min() == 255
    assert v.alpha is None                                 # an ignore mask, not a cut-out
    assert load_view(ds, ds.cameras[1], 32).mask is None
