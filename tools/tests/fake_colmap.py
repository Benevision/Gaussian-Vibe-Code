"""A fake ``colmap`` executable for orchestration tests.

* records every invocation (JSON line) in $FAKE_COLMAP_LOG,
* ``feature_extractor`` creates a database with the listed images,
* matchers are no-ops,
* ``point_triangulator`` / ``bundle_adjuster`` copy the input model (poses unchanged),
* ``mapper`` checks the database (needs >= 1 verified two-view geometry unless
  $FAKE_COLMAP_NO_CHECK) and writes the ground-truth poses from $FAKE_COLMAP_GT
  (a dataset.json) as a text model in <output_path>/0, using the DB image ids.
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

OPTIONS = {
    "feature_extractor": "--database_path --image_path --image_list_path --ImageReader.camera_model --ImageReader.single_camera "
    "--ImageReader.mask_path --FeatureExtraction.use_gpu --FeatureExtraction.max_image_size --SiftExtraction.max_num_features",
    "exhaustive_matcher": "--database_path --FeatureMatching.use_gpu",
    "sequential_matcher": "--database_path --SequentialMatching.overlap --FeatureMatching.use_gpu",
    "matches_importer": "--database_path --match_list_path --match_type --FeatureMatching.use_gpu",
    "mapper": "--database_path --image_path --output_path",
    "point_triangulator": "--database_path --image_path --input_path --output_path --clear_points",
    "bundle_adjuster": "--input_path --output_path --BundleAdjustment.refine_focal_length",
}


def opt(args, name):
    return args[args.index("--" + name) + 1]


def main():
    args = sys.argv[1:]
    log = os.environ.get("FAKE_COLMAP_LOG")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(args) + "\n")
    if not args or args[0] in ("help", "-h"):
        print("COLMAP fake\nUsage:\n  colmap [command] [options]\nAvailable commands:\n" + "\n".join(f"  {c}" for c in OPTIONS))
        return 0
    cmd = args[0]
    if "-h" in args or "--help" in args:
        print(OPTIONS.get(cmd, ""))
        return 0
    if cmd == "feature_extractor":
        from splatprep.colmap_db import COLMAPDatabase

        names = Path(opt(args, "image_list_path")).read_text().split()
        db = COLMAPDatabase.create(Path(opt(args, "database_path")))
        cid = db.add_camera(2, 100, 100, [100, 50, 50, 0])
        for k, n in enumerate(names):
            db.add_image(n, cid)
            print(f"Processed file [{k + 1}/{len(names)}]")
        db.close()
        return 0
    if cmd in ("exhaustive_matcher", "sequential_matcher", "matches_importer", "vocab_tree_matcher"):
        return 0
    if cmd == "mapper":
        from splatprep.colmap_model import Camera, Image, Reconstruction, write_text_model

        con = sqlite3.connect(opt(args, "database_path"))
        n_tvg = con.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0").fetchone()[0]
        if n_tvg == 0 and not os.environ.get("FAKE_COLMAP_NO_CHECK"):
            print("no verified pairs", file=sys.stderr)
            return 1
        ids = {name: iid for iid, name in con.execute("SELECT image_id, name FROM images")}
        gt = json.loads(Path(os.environ["FAKE_COLMAP_GT"]).read_text())
        rec = Reconstruction()
        c0 = gt["cameras"][0]
        rec.cameras[1] = Camera(1, "SIMPLE_RADIAL", c0["width"], c0["height"], np.array([c0["fx"], c0["cx"], c0["cy"], 0.0]))
        for c in gt["cameras"]:
            name = c["image"].split("/", 1)[1]
            if name in ids:
                im = Image(ids[name], np.array([1.0, 0, 0, 0]), np.zeros(3), 1, name)
                im.set_c2w(np.array(c["cameraToWorld"]))
                rec.images[ids[name]] = im
                print(f"Registering image #{ids[name]}")
        write_text_model(rec, Path(opt(args, "output_path")) / "0")
        return 0
    if cmd in ("point_triangulator", "bundle_adjuster"):
        # poses pass through unchanged (prior refinement of exact priors is a no-op)
        import shutil

        src, dst = Path(opt(args, "input_path")), Path(opt(args, "output_path"))
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy(f, dst / f.name)
        return 0
    print(f"unknown command {cmd}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
