"""Mock backend for protocol tests: returns the ground truth under a known Sim(3)."""

import json
from pathlib import Path

import numpy as np

from splatprep.backends.base import Availability, Backend, SfMOutput
from splatprep.backends.dense_common import build_reconstruction
from splatprep.geometry import Sim3, qvec_to_rotmat
from splatprep.images import image_size
from splatprep.ply import read_points_ply

TRUE_SIM3 = Sim3(2.5, qvec_to_rotmat([0.9, 0.1, -0.3, 0.2]), np.array([1.0, -2.0, 0.5]))


class MockBackend(Backend):
    id = "mock"
    display_name = "Mock (GT under Sim3)"
    kind = "dense"

    def check(self):
        return Availability(True, True)

    def run(self, ctx):
        print("stray stdout noise from a third-party library")  # must not corrupt the JSONL stream
        gt_path = Path(ctx.request.get("mockGroundTruth") or ctx.image_dir.parent / "dataset.json")
        gt = json.loads(gt_path.read_text())
        by_name = {c["image"].split("/", 1)[1]: c for c in gt["cameras"]}
        names = [n for n in ctx.names if n in by_name]
        drop = int(ctx.request.get("mockDrop", 0))
        names = names[: len(names) - drop] if drop else names
        inv = Sim3(1 / TRUE_SIM3.s, TRUE_SIM3.R.T, -TRUE_SIM3.R.T @ TRUE_SIM3.t / TRUE_SIM3.s)
        c2ws, Ks = [], []
        for k, n in enumerate(names):
            c = by_name[n]
            c2ws.append(inv.apply_c2w(np.asarray(c["cameraToWorld"], float)))
            Ks.append([c["fx"], c["fy"], c["cx"], c["cy"]])
            ctx.progress.update("features", (k + 1) / len(names), f"mock {n}")
        xyz, rgb = read_points_ply(gt_path.parent / "points3d.ply")
        rec = build_reconstruction(names, [image_size(ctx.image_dir / n) for n in names], np.stack(c2ws), np.array(Ks), inv.apply_points(xyz), rgb)
        ctx.progress.update("mapping", 1.0, "mock mapping done")
        return SfMOutput(rec, None, {"backend": self.id, "mapper": "mock"})


BACKENDS = [MockBackend()]
