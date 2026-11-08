"""Large training sets must not be held in RAM twice (pinned copies froze a 32 GB PC on a 9,824-view 360° set)."""
import numpy as np

from splatworker.viewstore import GPU_BUDGET, ViewStore, choose_mode


class V:
    def __init__(self, i):
        self.i, self.image, self.alpha, self.depth = i, None, None, None


class Full:
    def __init__(self, i):
        self.image = np.full((100, 100, 3), i % 255, np.uint8)
        self.alpha = np.full((100, 100), 255, np.uint8)
        self.depth = None


def test_modes():
    gib = 1024 ** 3
    assert choose_mode(GPU_BUDGET - 1, 64 * gib) == "gpu"
    assert choose_mode(10 * gib, 64 * gib) == "ram"
    assert choose_mode(56 * gib, 30 * gib) == "disk"     # the 360° set on the PC


def test_disk_cache_is_bounded_and_correct():
    loads = []
    store = ViewStore(lambda v: (loads.append(v.i), Full(v.i))[1], [], "disk", budget=0)
    store.budget = 5 * 40_000          # room for 5 views (30 KB image + 10 KB alpha each)
    views = [V(i) for i in range(20)]
    store.prefetch(views[:3])
    for v in views:
        im, al, _, _, _ = store.get(v)
        assert im[0, 0, 0] == v.i and al.shape == (100, 100)
    assert store.bytes <= store.budget and len(store.cache) <= 5
    # recently used views are served from the cache
    n = len(loads)
    store.get(views[-1])
    assert len(loads) == n
    store.close()


def test_views_with_pixels_bypass_the_cache():
    store = ViewStore(lambda v: None, [], "ram", budget=0)
    v = V(1)
    v.image = np.zeros((2, 2, 3), np.uint8)
    assert store.get(v)[0] is v.image
