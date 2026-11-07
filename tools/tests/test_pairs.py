import numpy as np
from PIL import Image

from splatprep.geometry import look_at
from splatprep.images import looks_like_video
from splatprep.pairs import (
    exhaustive_pairs,
    proximity_pairs,
    resolve_strategy,
    select_pairs,
    sequential_pairs,
    thumbnail_descriptors,
)


def test_sequential_and_exhaustive():
    assert sequential_pairs(5, 2) == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    assert len(exhaustive_pairs(10)) == 45
    assert all(i < j for i, j in exhaustive_pairs(7))


def test_video_like_detection():
    assert looks_like_video([f"frame_{i:05d}.jpg" for i in range(100)])
    assert looks_like_video([f"f{i * 5}.png" for i in range(100)])  # constant step
    assert not looks_like_video(["IMG_1234.jpg", "DSC_0001.jpg", "IMG_9999.jpg", "beach.jpg"])
    assert not looks_like_video([f"img_{i}.jpg" for i in (1, 7, 30, 31, 200, 999)])


def test_auto_strategy():
    small = [f"x{i}.jpg" for i in range(40)]
    video = [f"frame_{i:05d}.jpg" for i in range(300)]
    photos = [f"IMG_{i}.jpg" for i in np.random.default_rng(0).choice(9000, 300, replace=False)]
    assert resolve_strategy("auto", small) == "exhaustive"
    assert resolve_strategy("auto", video) == "sequential"
    assert resolve_strategy("auto", photos) == "retrieval"
    assert resolve_strategy("sequential", small) == "sequential"


def _make_images(tmp_path, n=12):
    """Two groups of visually distinct images; similar images within a group."""
    rng = np.random.default_rng(0)
    bases = [rng.random((8, 8, 3)), rng.random((8, 8, 3))]
    paths = []
    for i in range(n):
        g = i % 2
        img = np.kron(bases[g], np.ones((16, 16, 1))) + rng.normal(0, 0.02, (128, 128, 3))
        p = tmp_path / f"im_{i:02d}.png"
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(p)
        paths.append(p)
    return paths


def test_retrieval_prefers_similar_images(tmp_path):
    paths = _make_images(tmp_path)
    D = thumbnail_descriptors(paths)
    assert D.shape[0] == len(paths)
    pairs, info = select_pairs([p.name for p in paths], paths, "retrieval", top_k=3, work_dir=None)
    assert info["descriptor"] == "thumbnail"
    same_group = [((i % 2) == (j % 2)) for i, j in pairs]
    assert np.mean(same_group) > 0.9


def test_prior_proximity_pairs():
    # cameras on a line, all looking +Z: nearest neighbours only
    c2ws = {i: look_at([i * 1.0, 0, 0], [i * 1.0, 0, 10]) for i in range(10)}
    pairs = proximity_pairs(c2ws, k=2)
    assert all(abs(i - j) <= 2 for i, j in pairs)
    # opposite-facing camera is not paired even when close
    c2ws[10] = look_at([0.1, 0, 0], [0.1, 0, -10])
    pairs = proximity_pairs(c2ws, k=2)
    assert not any(10 in p and 0 in p for p in pairs)


def test_select_pairs_merges_priors(tmp_path):
    paths = _make_images(tmp_path, 6)
    names = [p.name for p in paths]
    priors = {i: look_at([i * 1.0, 0, 0], [i * 1.0, 0, 10]) for i in range(6)}
    pairs, info = select_pairs(names, paths, "sequential", overlap=1, top_k=3, prior_c2w=priors)
    assert set(sequential_pairs(6, 1)) <= set(pairs)
    assert (0, 2) in pairs  # added by proximity
    assert info["priorPairs"] > 0
