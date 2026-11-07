"""``colmap_sift`` — COLMAP CLI end to end.

feature_extractor -> sequential / exhaustive / vocab_tree matcher (or our own
pair list via ``matches_importer`` for retrieval / prior-pose pairs) -> mapper
(GLOMAP if present, else ``colmap mapper``).

Option names changed across COLMAP versions (e.g. ``SiftExtraction.use_gpu`` ->
``FeatureExtraction.use_gpu`` in 3.12); they are probed from ``colmap <cmd> -h``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

from ..mapper import find_executable, pick_option, run_logged, run_mapper
from ..pairs import select_pairs, write_pairs_txt
from .base import Availability, Backend, SfMContext, SfMOutput, camera_model_name


def gps_fraction(paths, sample: int = 20) -> float:
    """Share of (a sample of) images with GPS latitude/longitude in EXIF."""
    try:
        from PIL import Image
    except Exception:
        return 0.0
    paths = list(paths)
    if not paths:
        return 0.0
    step = max(1, len(paths) // sample)
    picked = paths[::step][:sample]
    hits = 0
    for p in picked:
        try:
            with Image.open(p) as im:
                g = im.getexif().get_ifd(0x8825)
            hits += 1 if (g.get(2) and g.get(4)) else 0
        except Exception:
            pass
    return hits / len(picked)


class ColmapSiftBackend(Backend):
    id = "colmap_sift"
    display_name = "COLMAP SIFT"
    kind = "sparse"
    install_hint = "brew install colmap glomap  (Linux: apt install colmap, or build with CUDA)"

    def check(self) -> Availability:
        if not find_executable("colmap"):
            return Availability(False, False, "colmap executable not found — " + self.install_hint)
        return Availability(True, True)

    def run(self, ctx: SfMContext) -> SfMOutput:
        self.require(ctx)
        colmap = find_executable("colmap")
        assert colmap
        db = ctx.work_dir / "database.db"
        if db.exists():
            db.unlink()
        list_path = ctx.work_dir / "image_list.txt"
        list_path.write_text("\n".join(ctx.names) + "\n")
        use_gpu = "1" if (ctx.device == "cuda" and sys.platform != "darwin") else "0"

        def opt(cmd: str, *names: str):
            return pick_option(colmap, cmd, *names)

        cmd: List[str] = [colmap, "feature_extractor", "--database_path", str(db), "--image_path", str(ctx.image_dir)]
        cmd += ["--image_list_path", str(list_path)]
        cmd += ["--ImageReader.camera_model", camera_model_name(ctx.request)]
        cmd += ["--ImageReader.single_camera", "1" if ctx.opt("singleCamera", True) else "0"]
        if ctx.masks_dir is not None:
            cmd += ["--ImageReader.mask_path", str(ctx.masks_dir)]
        for names, val in (
            (("FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"), use_gpu),
            (("FeatureExtraction.max_image_size", "SiftExtraction.max_image_size"), str(int(ctx.opt("resizeMax", 1600)))),
            (("SiftExtraction.max_num_features",), str(int(ctx.opt("maxKeypoints", 8192)))),
        ):
            o = opt("feature_extractor", *names)
            if o:
                cmd += [f"--{o}", val]
        ctx.progress.update("features", 0.05, "COLMAP feature extraction")
        n = len(ctx.names)
        counter = [0]

        def on_feat(line: str):
            if line.startswith("Processed file") or "Features:" in line:
                counter[0] += 1 if line.startswith("Processed file") else 0
                if counter[0]:
                    ctx.progress.update("features", min(1.0, counter[0] / n), f"extracted {counter[0]}/{n}")

        run_logged(cmd, ctx.log_path, on_feat)
        ctx.progress.update("features", 1.0, "features done")

        strategy = (ctx.opt("matching", "auto") or "auto").lower()
        prior_c2w = {i: ctx.priors[nm].c2w for i, nm in enumerate(ctx.names) if nm in ctx.priors}
        from ..prior_refine import lookat_pairs, refine_poses, set_prior_cameras, use_prior_refine

        if use_prior_refine(ctx.names, ctx.priors, ctx.request):
            # Known poses (iPhone capture): refine them rather than solve from scratch — far fewer pairs,
            # no mapper. See prior_refine.py.
            ncam = set_prior_cameras(db, ctx.names, ctx.priors)
            pairs = lookat_pairs(prior_c2w, k=int(ctx.opt("refineNeighbors", 10)), window=int(ctx.opt("refineWindow", 4)))
            pairs_txt = ctx.work_dir / "pairs.txt"
            write_pairs_txt(pairs_txt, ctx.names, pairs)
            ctx.progress.update("pairs", 1.0, f"{len(pairs)} pairs from the capture poses ({ncam} cameras with capture intrinsics)")
            mcmd = [colmap, "matches_importer", "--database_path", str(db), "--match_list_path", str(pairs_txt), "--match_type", "pairs"]
            g = opt("matches_importer", "FeatureMatching.use_gpu", "SiftMatching.use_gpu")
            if g:
                mcmd += [f"--{g}", use_gpu]
            ctx.progress.update("matching", 0.05, "COLMAP matches_importer")
            run_logged(mcmd, ctx.log_path)
            ctx.progress.update("matching", 1.0, "matching done")
            ctx.progress.update("verification", 1.0, "verified by COLMAP")
            rec, info = refine_poses(colmap, db, ctx.image_dir, ctx.priors, ctx.work_dir / "refine", ctx.log_path,
                                     ctx.progress.sub("mapping"))
            return SfMOutput(rec, None, {"backend": self.id, "mapper": "prior-refine", "matcher": "matches_importer",
                                         "pairs": {"strategy": "lookat", "numPairs": len(pairs)}, "priorRefine": info})
        has_ts = bool(ctx.priors) and all(ctx.priors.get(nm) is not None and ctx.priors[nm].timestamp is not None for nm in ctx.names)
        vocab = os.environ.get("SPLATPREP_VOCAB_TREE")
        gpu_opt = lambda c: opt(c, "FeatureMatching.use_gpu", "SiftMatching.use_gpu")
        pairs, pinfo = select_pairs(
            ctx.names, ctx.paths, strategy, int(ctx.opt("sequentialOverlap", 10)), int(ctx.opt("retrievalTopK", 30)), prior_c2w or None, has_ts, ctx.work_dir
        )
        resolved = pinfo["resolved"]
        # Aerial / drone sets: photos with GPS in EXIF are matched with their spatial neighbours (COLMAP's
        # spatial_matcher reads the GPS priors feature_extractor stored), which beats retrieval for surveys.
        gps = (strategy in ("auto", "spatial") and not prior_c2w and (strategy == "spatial" or n > 60)
               and gps_fraction(ctx.paths) >= 0.9)
        if gps:
            resolved = "spatial"
            pinfo["resolved"] = "spatial"
        ctx.progress.update("pairs", 1.0, f"{len(pairs)} pairs ({resolved})" if not gps else "spatial (GPS) matching")
        if gps:
            mcmd = [colmap, "spatial_matcher", "--database_path", str(db), "--SpatialMatching.ignore_z", "0",
                    "--SpatialMatching.max_num_neighbors", str(int(ctx.opt("spatialNeighbors", 40))),
                    "--SpatialMatching.max_distance", str(float(ctx.opt("spatialMaxDistance", 200)))]
            c = "spatial_matcher"
        elif resolved == "exhaustive" and not prior_c2w:
            mcmd = [colmap, "exhaustive_matcher", "--database_path", str(db)]
            c = "exhaustive_matcher"
        elif resolved == "sequential" and not prior_c2w and strategy == "sequential":
            mcmd = [colmap, "sequential_matcher", "--database_path", str(db), "--SequentialMatching.overlap", str(int(ctx.opt("sequentialOverlap", 10)))]
            c = "sequential_matcher"
        elif resolved == "retrieval" and vocab and Path(vocab).exists() and not prior_c2w:
            mcmd = [colmap, "vocab_tree_matcher", "--database_path", str(db), "--VocabTreeMatching.vocab_tree_path", vocab]
            mcmd += ["--VocabTreeMatching.num_images", str(int(ctx.opt("retrievalTopK", 30)))]
            c = "vocab_tree_matcher"
        else:
            pairs_txt = ctx.work_dir / "pairs.txt"
            write_pairs_txt(pairs_txt, ctx.names, pairs)
            mcmd = [colmap, "matches_importer", "--database_path", str(db), "--match_list_path", str(pairs_txt), "--match_type", "pairs"]
            c = "matches_importer"
        g = gpu_opt(c)
        if g:
            mcmd += [f"--{g}", use_gpu]
        ctx.progress.update("matching", 0.05, f"COLMAP {c}")
        run_logged(mcmd, ctx.log_path)
        ctx.progress.update("matching", 1.0, "matching done")
        ctx.progress.update("verification", 1.0, "verified by COLMAP")
        rec, mapper = run_mapper(db, ctx.image_dir, ctx.work_dir / "sparse", ctx.opt("mapper", "auto"), ctx.log_path, ctx.progress.sub("mapping"), n)
        return SfMOutput(rec, None, {"backend": self.id, "mapper": mapper, "matcher": c, "pairs": pinfo})
