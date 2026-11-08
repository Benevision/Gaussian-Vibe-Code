import numpy as np

from splatworker.studio import autoclean
from splatworker.studio.presets import PRESETS, ProjectOptions, training_config


def test_autoclean_rules_match_mac():
    n = 1000
    peak = np.full(n, 0.5, np.float32)
    views = np.full(n, 40, np.uint32)
    peak[:100] = 0.01                  # invisible
    views[100:150] = 5                 # floaters: < 25 % of 40
    p = autoclean.plan(peak, views, np.full(n, 3.0))
    assert p.invisible == 100 and p.few_views == 50 and p.typical_views == 40
    assert p.remove[:150].all() and not p.remove[150:].any()
    q = autoclean.plan(peak, views, np.full(n, 3.0), include_few_views=False)
    assert q.few_views == 0 and q.total == 100


def test_autoclean_small_capture_skips_few_views():
    peak = np.full(100, 0.5, np.float32)
    views = np.full(100, 8, np.uint32)
    views[:10] = 1
    p = autoclean.plan(peak, views, np.zeros(100))
    assert p.typical_views == 0 and p.few_views == 0


def test_presets_and_requests():
    assert set(PRESETS) == {"preview", "balanced", "high", "ultra"}
    cfg = training_config("ultra", {"seed": 7})
    assert cfg["iterations"] == 40_000 and cfg["maxGaussians"] == 3_000_000 and cfg["seed"] == 7
    o = ProjectOptions(name="x", source_kind="video360", source="/v.mp4", views_360=8, target_frames=200)
    r = o.frames_request("v.mp4")
    assert r["equirect"]["rings"][0]["count"] == 8 and r["videoOptions"]["targetFrameCount"] == 200
    s = o.sfm_request(True, True, True)
    assert s["masksDir"] == "masks" and s["priorMode"] == "auto"
    assert ProjectOptions.from_dict({**o.to_dict(), "unknown": 1}) == o
    from splatworker.config import TrainingConfig

    TrainingConfig.from_dict(o.train_config())       # every key is valid for the worker


def test_catalog_in_sync_with_mac():
    import importlib.util
    from pathlib import Path

    import pytest

    from splatworker.studio import catalog

    root = Path(__file__).resolve().parents[2]
    swift = root / "mac/Sources/SplatStudio/Core/SettingsCatalog.swift"
    if not swift.is_file():
        pytest.skip("Mac sources not present")
    spec = importlib.util.spec_from_file_location("gen_catalog", root / "linux/scripts/gen_catalog.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert gen.parse(swift.read_text(encoding="utf-8")) == catalog.load(), \
        "catalog.json is stale: run python3 linux/scripts/gen_catalog.py"
    assert catalog.doc("viewer.displayMode")["group"] == "Viewer"
    assert any(s["id"] == "edit.autoClean" for s in catalog.search("floaters"))
    # every Studio training control is documented
    from splatworker.studio.presets import TRAIN_FIELDS

    missing = [cid for cid, *_ in TRAIN_FIELDS if catalog.doc(cid) is None]
    assert not missing, missing


def test_prefs_lenient(tmp_path):
    import json

    from splatworker.studio.prefs import StudioPrefs

    p = tmp_path / "s.json"
    p.write_text(json.dumps({"fov": 75, "invert_vertical": "yes", "background": [1, 1], "x": 1}))
    s = StudioPrefs.load(p)
    assert s.fov == 75.0 and s.invert_vertical is False and s.background == [0.06, 0.06, 0.07]
    s.save(p)
    assert StudioPrefs.load(p) == s


def test_capture_receiver(tmp_path):
    import io
    import json
    import urllib.request
    import zipfile

    import pytest

    from splatworker.studio.inbox import CaptureReceiver, list_captures

    got = []
    r = CaptureReceiver(tmp_path / "inbox", "Test PC", port=17879, on_received=got.append, mdns=False)
    assert r.start()
    try:
        base = f"http://127.0.0.1:{r.port}"
        hello = json.loads(urllib.request.urlopen(base + "/v1/hello").read())
        assert hello == {"name": "Test PC", "app": "Splat Studio", "version": 1}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("Kitchen.splatcapture/manifest.json", json.dumps({"format": "splatcapture", "frames": [{}]}))
            z.writestr("Kitchen.splatcapture/images/000000.jpg", b"jpeg")
        body = buf.getvalue()

        def post(code):
            req = urllib.request.Request(base + "/v1/captures", data=body, method="POST", headers={
                "Content-Type": "application/zip", "X-Capture-Name": "Kitchen%20one", "X-Pairing-Code": code})
            return urllib.request.urlopen(req)
        with pytest.raises(urllib.error.HTTPError) as e:
            post("000000" if r.code != "000000" else "111111")
        assert e.value.code == 403
        res = json.loads(post(r.code[:3] + " " + r.code[3:]).read())
        assert res == {"ok": True, "savedAs": "Kitchen one.splatcapture"}
        caps = list_captures(tmp_path / "inbox")
        assert len(caps) == 1 and caps[0].frames == 1 and got and got[0].name == "Kitchen one"
        assert (caps[0].path / "images" / "000000.jpg").read_bytes() == b"jpeg"
    finally:
        r.stop()
