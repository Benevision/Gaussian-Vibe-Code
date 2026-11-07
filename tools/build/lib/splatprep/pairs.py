"""Image pair selection: sequential / exhaustive / retrieval / auto (+ prior-pose proximity).

``auto``:
  * N <= ``EXHAUSTIVE_MAX`` (60): exhaustive (cheap and most robust),
  * video-like names (shared prefix + consecutive frame numbers) or timestamped
    priors: sequential (overlap) + a few retrieval pairs for loop closure,
  * otherwise retrieval (global descriptor top-K).
If prior poses are given, the K spatially nearest cameras (with compatible
viewing direction) are always added.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .images import load_rgb, looks_like_video

EXHAUSTIVE_MAX = 60
Pair = Tuple[int, int]


def _norm(i: int, j: int) -> Pair:
    return (i, j) if i < j else (j, i)


def sequential_pairs(n: int, overlap: int = 10) -> List[Pair]:
    out = set()
    for i in range(n):
        for k in range(1, overlap + 1):
            if i + k < n:
                out.add((i, i + k))
    return sorted(out)


def exhaustive_pairs(n: int) -> List[Pair]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def topk_pairs_from_similarity(sim: np.ndarray, k: int, min_sim: float = -np.inf) -> List[Pair]:
    n = sim.shape[0]
    sim = sim.copy()
    np.fill_diagonal(sim, -np.inf)
    k = min(k, n - 1)
    out = set()
    for i in range(n):
        if k <= 0:
            break
        idx = np.argpartition(-sim[i], k - 1)[:k]
        for j in idx:
            if sim[i, j] > min_sim:
                out.add(_norm(i, int(j)))
    return sorted(out)


def proximity_pairs(c2ws: Dict[int, np.ndarray], k: int, max_angle_deg: float = 100.0) -> List[Pair]:
    """K nearest cameras by centre distance whose optical axes differ by < max_angle."""
    ids = sorted(c2ws)
    if len(ids) < 2:
        return []
    C = np.stack([c2ws[i][:3, 3] for i in ids])
    Z = np.stack([c2ws[i][:3, 2] for i in ids])
    D = np.linalg.norm(C[:, None] - C[None], axis=2)
    cosang = Z @ Z.T
    D[cosang < np.cos(np.radians(max_angle_deg))] = np.inf
    np.fill_diagonal(D, np.inf)
    out = set()
    kk = min(k, len(ids) - 1)
    for a in range(len(ids)):
        for b in np.argsort(D[a])[:kk]:
            if np.isfinite(D[a, b]):
                out.add(_norm(ids[a], ids[int(b)]))
    return sorted(out)


# ---------------------------------------------------------------------------
# global descriptors


def thumbnail_descriptors(paths: Sequence[Path], size: int = 32) -> np.ndarray:
    """Dependency-free global descriptor: tiny gray thumbnail + 4x4x8 gradient-orientation histogram."""
    from PIL import Image

    descs = []
    for p in paths:
        rgb, _ = load_rgb(p, resize_max=128)
        g = rgb.mean(2).astype(np.float32)
        th = np.asarray(Image.fromarray(g).resize((size, size), Image.BILINEAR), np.float32)
        th = th - th.mean()
        th /= np.linalg.norm(th) + 1e-6
        gy, gx = np.gradient(np.asarray(Image.fromarray(g).resize((64, 64), Image.BILINEAR), np.float32))
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) % np.pi) / np.pi * 8
        hist = np.zeros((4, 4, 8), np.float32)
        b = np.minimum(ang.astype(int), 7)
        for cy in range(4):
            for cx in range(4):
                sl = (slice(cy * 16, cy * 16 + 16), slice(cx * 16, cx * 16 + 16))
                hist[cy, cx] = np.bincount(b[sl].ravel(), weights=mag[sl].ravel(), minlength=8)
        hist = np.sqrt(hist.ravel())
        hist /= np.linalg.norm(hist) + 1e-6
        col = np.asarray(Image.fromarray(rgb).resize((4, 4), Image.BILINEAR), np.float32).ravel()
        col = col - col.mean()
        col /= np.linalg.norm(col) + 1e-6
        d = np.concatenate([th.ravel(), hist, 0.5 * col])
        descs.append(d / (np.linalg.norm(d) + 1e-6))
    return np.stack(descs) if descs else np.zeros((0, size * size + 128 + 48))


def netvlad_descriptors(paths: Sequence[Path], work_dir: Path) -> Optional[np.ndarray]:
    """NetVLAD via hloc if installed (weights are fetched by hloc on first use)."""
    try:
        from hloc import extract_features  # type: ignore
        import h5py  # type: ignore
    except Exception:
        return None
    try:
        image_dir = Path(paths[0]).parent
        names = [Path(p).relative_to(image_dir).as_posix() for p in paths]
        conf = extract_features.confs["netvlad"]
        out = Path(work_dir) / "global-netvlad.h5"
        extract_features.main(conf, image_dir, image_list=names, feature_path=out)
        with h5py.File(out, "r") as f:
            D = np.stack([f[n]["global_descriptor"][()] for n in names])
        return D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    except Exception:
        return None


def global_descriptors(paths: Sequence[Path], work_dir: Optional[Path] = None, method: str = "auto") -> Tuple[np.ndarray, str]:
    if method in ("auto", "netvlad") and work_dir is not None:
        D = netvlad_descriptors(paths, work_dir)
        if D is not None:
            return D, "netvlad"
    return thumbnail_descriptors(paths), "thumbnail"


# ---------------------------------------------------------------------------


def resolve_strategy(strategy: str, names: Sequence[str], has_timestamps: bool = False) -> str:
    strategy = (strategy or "auto").lower()
    if strategy != "auto":
        return strategy
    n = len(names)
    if n <= EXHAUSTIVE_MAX:
        return "exhaustive"
    if has_timestamps or looks_like_video(list(names)):
        return "sequential"
    return "retrieval"


def select_pairs(
    names: Sequence[str],
    paths: Sequence[Path],
    strategy: str = "auto",
    overlap: int = 10,
    top_k: int = 30,
    prior_c2w: Optional[Dict[int, np.ndarray]] = None,
    has_timestamps: bool = False,
    work_dir: Optional[Path] = None,
    descriptor_fn: Optional[Callable[[Sequence[Path]], np.ndarray]] = None,
) -> Tuple[List[Pair], dict]:
    """Returns (sorted unique (i<j) pairs, info dict)."""
    n = len(names)
    resolved = resolve_strategy(strategy, names, has_timestamps)
    info = {"strategy": strategy, "resolved": resolved}
    pairs: Set[Pair] = set()
    if resolved == "exhaustive":
        pairs.update(exhaustive_pairs(n))
    elif resolved == "sequential":
        pairs.update(sequential_pairs(n, overlap))
        if n > 2 * overlap and strategy == "auto":  # loop closure candidates
            D = descriptor_fn(paths) if descriptor_fn else global_descriptors(paths, work_dir)[0]
            sim = D @ D.T
            for i in range(n):
                sim[i, max(0, i - overlap) : i + overlap + 1] = -np.inf
            pairs.update(topk_pairs_from_similarity(sim, k=min(5, max(1, top_k // 6))))
    elif resolved == "retrieval":
        if descriptor_fn:
            D, how = descriptor_fn(paths), "custom"
        else:
            D, how = global_descriptors(paths, work_dir)
        info["descriptor"] = how
        pairs.update(topk_pairs_from_similarity(D @ D.T, top_k))
    else:
        raise ValueError(f"unknown matching strategy: {strategy}")
    if prior_c2w:
        prox = proximity_pairs(prior_c2w, k=max(top_k, overlap))
        info["priorPairs"] = len(prox)
        pairs.update(prox)
    out = sorted(p for p in pairs if p[0] != p[1])
    info["numPairs"] = len(out)
    return out, info


def write_pairs_txt(path: Path, names: Sequence[str], pairs: Sequence[Pair]) -> None:
    with open(path, "w") as f:
        for i, j in pairs:
            f.write(f"{names[i]} {names[j]}\n")
