"""Other compute devices: discovery parsing, the device store / registry, the Mac's throughput and memory rules,
block partition / distribution / merge, and training on a second in-process worker (one device and
block-parallel) through the Studio pipeline."""
import json
import os
import socket
import stat
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from splatworker.ply import write_gaussian_ply
from splatworker.studio import blocks as B
from splatworker.studio import devices as D


# ============================================================================ discovery parsing

def test_txt_and_addresses():
    txt = D.decode_txt({b"version": b"1", b"gpu": b"NVIDIA GeForce RTX 4080", b"auth": b"1", b"kind": None,
                        b"addrs": b"192.168.0.113:ethernet,192.168.0.120:wifi"})
    assert txt == {"version": "1", "gpu": "NVIDIA GeForce RTX 4080", "auth": "1", "kind": "",
                   "addrs": "192.168.0.113:ethernet,192.168.0.120:wifi"}
    alt = D.parse_addrs(txt["addrs"])
    assert alt == {"192.168.0.113": "ethernet", "192.168.0.120": "wifi"}
    assert D.parse_addrs("") == {} and D.parse_addrs("junk,:x,1.2.3.4:wifi") == {"1.2.3.4": "wifi"}
    assert D.preferred_address("192.168.0.120", alt) == "192.168.0.113"
    assert D.preferred_address("192.168.0.113", alt) is None
    assert D.preferred_address("10.0.0.1", {"10.0.0.1": "wifi"}) is None
    assert D.instance_name("Ben's MacBook Pro._splatworker._tcp.local.") == "Ben's MacBook Pro"
    assert D.instance_name("pc-4080._splatworker._tcp.local.", "_splatworker._tcp.local.") == "pc-4080"


def test_parse_address():
    assert D.parse_address("192.168.0.113") == ("192.168.0.113", 7878)
    assert D.parse_address(" http://192.168.0.113:7879/ ") == ("192.168.0.113", 7879)
    assert D.parse_address("pc.local:9000") == ("pc.local", 9000)
    assert D.parse_address("[fe80::1]:7878") == ("fe80::1", 7878)
    assert D.parse_address("fe80::1") == ("fe80::1", 7878)
    for bad in ("", "host:0", "host:99999", "host:abc"):
        with pytest.raises(ValueError):
            D.parse_address(bad)


def test_device_kind_and_summary():
    mac = D.Device("mdns:Mac", "Mac", "10.0.0.2", 7879, txt={"kind": "mac", "gpu": "Apple M4 Pro"})
    assert mac.kind == "mac" and not mac.has_sky and mac.slots == 1
    phone = D.Device("mdns:iPhone", "iPhone", "10.0.0.3", 7879, info={"workerVersion": "ios-1.0", "gpus": []})
    assert phone.kind == "iphone"
    pc = D.Device("manual:pc:7878", "pc", "pc", 7878, status="online",
                  info={"gpus": [{"name": "NVIDIA GeForce RTX 4080", "memoryMB": 16376}] * 2, "queueLength": 2,
                        "capabilities": ["train", "process", "chain"], "busy": True})
    assert pc.kind == "linux" and pc.slots == 2 and pc.can_chain and pc.has_sky
    assert pc.gpu_name == "2 × NVIDIA GeForce RTX 4080"
    assert pc.summary() == "SplatWork for Linux · pc:7878 · 2 × NVIDIA GeForce RTX 4080 · busy · 2 queued"
    assert D.Device("x", "v6", "fe80::1", 7878).base_url == "http://[fe80::1]:7878"


def test_throughput_table():
    assert D.estimate_throughput([{"name": "NVIDIA GeForce RTX 4080", "memoryMB": 16376}]) == 3.8
    assert D.estimate_throughput([{"name": "NVIDIA GeForce RTX 4090"}, {"name": "NVIDIA GeForce RTX 4090"}]) == 10
    assert D.estimate_throughput([{"name": "Apple A18 Pro GPU"}]) == 0.22          # not the CUDA "A10"
    assert D.estimate_throughput([{"name": "Apple M4 Pro"}]) == 1
    assert D.estimate_throughput([{"name": "Mystery GPU", "memoryMB": 24000}]) == 3.0
    assert D.estimate_throughput([]) == 1.0
    assert D.estimate_throughput([{"name": "AMD Radeon RX 7900 XTX"}]) == 3.5
    assert D.estimate_throughput([{"name": "AMD Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M"}]) == 3.5
    assert D.estimate_throughput([{"name": "Intel Arc A770"}]) == 1.4     # not the CUDA "A10"
    assert D.estimate_throughput([{"name": "AMD Instinct MI300X"}]) == 7


def _cams(n, w=1920, h=1440):
    return [{"id": i, "width": w, "height": h} for i in range(n)]


def test_fit_problem_gpu_ram_and_apple_budget():
    pc = D.Device("p", "PC", "h", 1, info={"gpus": [{"name": "RTX 4080", "memoryMB": 16376}],
                                           "capabilities": ["train", "stream"]})
    assert D.fit_problem(pc, _cams(200), {"maxGaussians": 3_000_000, "maxImageSize": 1600}) is None
    p = D.fit_problem(pc, _cams(200), {"maxGaussians": 20_000_000, "maxImageSize": 1600})
    assert p and "GPU memory" in p
    old = D.Device("o", "Old", "h", 1, info={"gpus": [], "capabilities": ["train"], "memory": {"availableMB": 8000}})
    assert "in memory" in D.fit_problem(old, _cams(2000, 4000, 3000), {"maxImageSize": 4000})
    phone = D.Device("i", "iPhone", "h", 1, info={"kind": "iphone", "memory": {"availableMB": 3000}})
    assert D.fit_problem(phone, _cams(100), {"maxImageSize": 1600, "maxGaussians": 1_500_000}) is None
    big = D.fit_problem(phone, _cams(6000), {"maxImageSize": 1600, "bilateralGrid": True})
    assert big and "can't hold all 6000 photos" in big
    unknown = D.Device("u", "U", "h", 1, info={"kind": "mac"})
    assert D.fit_problem(unknown, _cams(10_000), {}) is None


# ============================================================================ store / registry

def test_store_and_registry(tmp_path):
    path = tmp_path / "cfg" / "devices.json"
    reg = D.DeviceRegistry(path, own=lambda: [("192.168.0.50", 7878)], own_names=lambda: ["this-pc"])
    d = reg.add_manual("10.1.1.1", 7878, token="secret", check=False)
    assert d.id == "manual:10.1.1.1:7878" and d.token == "secret"
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert reg.found("this-pc", "192.168.0.50", 7878, {}) is None              # our own worker by address
    reg.check = lambda *_a, **_k: None
    reg.found("this-pc", "192.168.0.50", 7900, {})                              # … and by name
    reg.found("Ben's Mac", "192.168.0.60", 7879, {"kind": "mac", "addrs": "192.168.0.61:ethernet"})
    names = [x.name for x in reg.devices()]
    assert names == ["10.1.1.1", "Ben's Mac"]
    mac = reg.get("mdns:Ben's Mac")
    assert mac.kind == "mac" and mac.alternates == {"192.168.0.61": "ethernet"}
    reg.remove(mac.id)                                                          # hidden, not forgotten
    assert [x.name for x in reg.devices()] == ["10.1.1.1"]
    reg2 = D.DeviceRegistry(path)
    assert [x.id for x in reg2.devices()] == ["manual:10.1.1.1:7878"]
    assert reg2.get("manual:10.1.1.1:7878").token == "secret"
    assert reg2.store.hidden == ["mdns:Ben's Mac"]
    reg2.unhide_all()
    reg2.remove("manual:10.1.1.1:7878")
    reg3 = D.DeviceRegistry(path)
    assert reg3.devices() == [] and reg3.store.tokens == {} and reg3.store.hidden == []
    reg.lost("Ben's Mac")
    assert reg.get("mdns:Ben's Mac") is None


def test_store_is_lenient(tmp_path):
    p = tmp_path / "devices.json"
    p.write_text('{"manual": [{"host": 3}, {"host": "a", "port": 1}], "tokens": {"x": 5}, "hidden": "no"}')
    s = D.DeviceStore.load(p)
    assert s.manual == [{"host": "a", "port": 1, "name": "a"}] and s.tokens == {} and s.hidden == []
    p.write_text("not json")
    assert D.DeviceStore.load(p).manual == []


def test_health_check_against_a_fake_http_worker(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Authorization") != "Bearer tok":
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b'{"error":"unauthorized"}')
                return
            body = json.dumps({"name": "iPhone 16 Pro", "workerVersion": "ios-1.0", "kind": "iphone",
                               "gpus": [{"name": "Apple A18 Pro GPU", "memoryMB": 8000}], "queueLength": 0,
                               "capabilities": ["train"], "memory": {"availableMB": 3000}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        reg = D.DeviceRegistry(tmp_path / "devices.json")
        d = reg.add_manual("127.0.0.1", srv.server_address[1], check=False)
        assert reg.check(d.id).status == "unauthorized"
        reg.set_token(d.id, "tok")
        d = reg.get(d.id)
        assert d.status == "online" and d.kind == "iphone" and d.name == "iPhone 16 Pro"
        assert d.throughput == 0.22 and not d.can_chain
        # iPhones no longer act as workers: an older app that still answers is left out of the device list
        assert reg.devices() == []
        srv.shutdown()
        srv.server_close()
        assert reg.check(d.id).status == "offline"
    finally:
        srv.server_close()


# ============================================================================ blocks

def _look_at(pos, target=(0, 0, 0)):
    z = np.asarray(target, float) - pos
    z /= np.linalg.norm(z)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, pos
    return m


def street_doc(n=40, length=20.0):
    """Cameras walking along +X looking sideways at a wall of points (a scene worth splitting)."""
    cams = []
    for i in range(n):
        pos = np.array([-length / 2 + length * i / (n - 1), 0.0, -3.0])
        cams.append({"id": i, "image": f"images/{i:06d}.jpg", "width": 32, "height": 24, "fx": 20.0,
                     "fy": 20.0, "cx": 16.0, "cy": 12.0, "cameraToWorld": _look_at(pos, pos + [0, 0, 3]).tolist(),
                     "holdout": i % 10 == 9})
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(-length / 2, length / 2, 5000), rng.uniform(-2, 2, 5000), rng.uniform(-0.5, 0.5, 5000)]
    return {"format": "splatdataset", "version": 1, "upVector": [0, -1, 0], "cameras": cams}, pts


def test_partition_covers_the_scene_and_assigns_cameras():
    doc, pts = street_doc()
    blocks = B.partition(doc, pts, 4)
    assert len(blocks) == 4
    owned = np.stack([b.owns(pts) for b in blocks])
    assert (owned.sum(0) == 1).all()                     # every point belongs to exactly one block
    far = np.array([[1e6, 0, 0], [-1e6, 5e5, 0], [0, -1e6, 1e6]])
    assert (np.stack([b.owns(far) for b in blocks]).sum(0) == 1).all()   # open outer faces
    for b in blocks:
        assert b.open_min[1] and b.open_max[1]          # never split along the vertical (Y) axis
        assert b.open_min[2] and b.open_max[2]          # the wall is long in X: all cuts are along X
        assert b.camera_ids and all(i % 10 != 9 for i in b.camera_ids)      # no holdout cameras
        assert len(b.camera_ids) < 36                    # a block sees only part of the street
    xs = [b.min[0] for b in blocks]
    assert xs == sorted(xs)
    assert sorted(blocks[0].camera_ids)[0] == 0 and 38 in blocks[-1].camera_ids
    # the weighted split balances points + cameras
    w = [b.weight for b in blocks]
    assert max(w) / min(w) < 1.6


def test_partition_without_points_uses_box_coverage():
    doc, _ = street_doc()
    blocks = B.partition(doc, None, 2)
    assert len(blocks) == 2 and all(b.camera_ids for b in blocks)


def test_block_config_and_distribute():
    doc, pts = street_doc()
    blocks = B.partition(doc, pts, 4)
    cfg = B.block_config({"iterations": 100, "maxGaussians": 1_000_000}, blocks[1], blocks)
    assert cfg["region"]["min"] == blocks[1].min and cfg["region"]["margin"] == 0.15
    assert cfg["cameraIds"] == blocks[1].camera_ids and cfg["iterations"] == 100
    assert 100_000 <= cfg["maxGaussians"] <= 1_000_000
    plan = B.distribute(blocks, {"local": 3.8, "mac": 1.0})
    assert sorted(b.index for bs in plan.values() for b in bs) == [0, 1, 2, 3]
    assert len(plan["local"]) >= len(plan.get("mac", []))
    only_local = B.distribute(blocks, {"local": 1.0, "mac": 5.0}, fits=lambda d, b: d == "local")
    assert list(only_local) == ["local"]
    nowhere = B.distribute(blocks, {"a": 1.0, "b": 2.0}, fits=lambda d, b: False)
    assert list(nowhere) == ["b"]                        # fits nowhere: the fastest device
    assert B.distribute(blocks, {}) == {}


def test_plan_blocks_grows_until_blocks_fit():
    doc, pts = street_doc(n=120, length=80)
    plan = B.plan_blocks(doc, pts, 1, lambda b, bs: len(b.camera_ids) <= 30)
    assert len(plan) > 2 and all(len(b.camera_ids) <= 30 for b in plan)
    assert len(B.plan_blocks(doc, pts, 3, lambda b, bs: True)) == 6


def _block_ply(path, xyz, degree=3):
    n = len(xyz)
    rng = np.random.default_rng(len(xyz))
    if degree == 3:          # the trainer's layout (45 f_rest columns)
        write_gaussian_ply(path, xyz, rng.normal(size=(n, 3)), rng.normal(size=(n, 15, 3)), rng.normal(size=n),
                           rng.normal(size=(n, 3)), rng.normal(size=(n, 4)), up=np.array([0, -1, 0]))
        return
    from splatworker.studio.formats import write_ply
    from splatworker.studio.splats import GaussianCloud

    k = (degree + 1) ** 2
    write_ply(GaussianCloud(xyz.astype(np.float32), rng.normal(size=(n, 3)).astype(np.float32),
                            rng.normal(size=(n, 4)).astype(np.float32), rng.normal(size=n).astype(np.float32),
                            rng.normal(size=(n, k, 3)).astype(np.float32)), path)


def test_merge_keeps_owned_splats_streamed(tmp_path):
    from splatworker.studio import formats

    doc, pts = street_doc()
    blocks = B.partition(doc, pts, 3)
    rng = np.random.default_rng(3)
    parts, expect = [], 0
    for b in blocks:
        lo, hi = b.expanded
        xyz = rng.uniform(lo, hi, size=(500, 3))          # each block's result spills into its margins
        p = tmp_path / f"b{b.index}.ply"
        _block_ply(p, xyz, degree=3 if b.index != 1 else 1)          # a lower-SH block (e.g. from a Mac)
        parts.append((b, p))
        expect += int(b.owns(xyz).sum())
    out = tmp_path / "merged.ply"
    n = B.merge_plys(parts, out, chunk=64)
    assert n == expect and 0 < n < 1500
    g = formats.load(out)
    assert g.count == n and g.sh_degree == 3
    assert np.allclose(g.up, [0, -1, 0])
    owners = np.stack([b.owns(g.positions) for b in blocks]).sum(0)
    assert (owners == 1).all()
    with pytest.raises(B.MergeError):
        B.merge_plys([], out)


# ============================================================================ training on other workers

pytest.importorskip("uvicorn")
from conftest import FakeBackend, wait_for  # noqa: E402
from splatworker.config import ServerSettings  # noqa: E402
from splatworker.server import create_app  # noqa: E402
from splatworker.studio import compute, formats  # noqa: E402
from splatworker.studio.client import Client  # noqa: E402
from splatworker.studio.pipeline import LocalPipeline  # noqa: E402
from splatworker.studio.presets import ProjectOptions  # noqa: E402
from splatworker.studio.projects import Project  # noqa: E402


class RecordingBackend(FakeBackend):
    """Remembers the files each training job saw (the dataset is deleted after the run)."""

    def __init__(self):
        super().__init__()
        self.files = []

    def run(self, ctx):
        if ctx.kind == "train":
            self.files.append({f.relative_to(ctx.dataset_dir).as_posix() for f in ctx.dataset_dir.rglob("*")
                               if f.is_file()})
        return super().run(ctx)


class AppleLikeBackend(FakeBackend):
    """A worker without chain / compose (like the Mac's and iPhone's)."""

    def capabilities(self):
        return ["train"]


def _serve(tmp_path, name, backend, token, local):
    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    settings = ServerSettings(data_dir=tmp_path / name, token=token, name=name, mdns=False, host="127.0.0.1",
                              port=port, allow_local_datasets=local, local_secret=("k" * 32) if local else None)
    app = create_app(settings, backend=backend, devices=[None])
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(300):
        if srv.started:
            break
        time.sleep(0.02)
    return srv, th, port, settings


@pytest.fixture
def workers(tmp_path, monkeypatch):
    monkeypatch.setattr(compute.ComputeRun, "poll_seconds", 0.02)
    started = []

    def make(name, backend=None, token=None, local=False):
        backend = backend or FakeBackend()
        srv, th, port, settings = _serve(tmp_path, name, backend, token, local)
        started.append((srv, th))
        return port, backend, settings
    yield make
    for srv, th in started:
        srv.should_exit = True
        th.join(5)


def make_project(tmp_path, train_on, n=24):
    from PIL import Image

    doc, pts = street_doc(n=n, length=12)
    opts = ProjectOptions(name="Street", source_kind="dataset", source="", preset="preview", train_on=train_on,
                          split_above=0)
    proj = Project.create(tmp_path / "projects", "Street", opts.to_dict())
    ds = proj.dataset_dir
    (ds / "images").mkdir(parents=True)
    (ds / "masks").mkdir()
    for c in doc["cameras"]:
        Image.fromarray(np.full((24, 32, 3), (c["id"] * 9) % 256, np.uint8)).save(ds / c["image"])
        if c["id"] % 2 == 0:
            Image.fromarray(np.full((24, 32), 255, np.uint8)).save(ds / f"masks/{c['id']:06d}.png")
            c["mask"] = f"masks/{c['id']:06d}.png"
    (ds / "splatprep-masks").mkdir()                       # intermediates never uploaded
    (ds / "splatprep-masks" / "huge.bin").write_bytes(b"x" * 1000)
    write_gaussian_ply(ds / "points3d.ply", pts, np.zeros((len(pts), 3)), None, np.zeros(len(pts)),
                       np.zeros((len(pts), 3)), np.tile([1.0, 0, 0, 0], (len(pts), 1)))
    doc["points"] = "points3d.ply"
    (ds / "dataset.json").write_text(json.dumps(doc))
    return proj, opts


def pipeline(local_port, proj, opts, registry):
    events = []
    p = LocalPipeline(Client(f"http://127.0.0.1:{local_port}", local_secret="k" * 32), proj, opts,
                      lambda k, d: events.append((k, d)), train=True)
    p.poll_seconds = 0.02
    p.preview_seconds = 0.0
    p.devices = registry
    return p, events


def _run(p):
    p._titles = ["Training"]
    try:
        return p.train()
    finally:
        for d in p._registered:
            p.client.delete_dataset(d)


def test_train_on_a_chosen_device_reuses_its_photos(tmp_path, workers):
    lport, lback, _ = workers("local", local=True)
    rport, rback, rset = workers("remote", backend=RecordingBackend(), token="r")
    reg = D.DeviceRegistry(tmp_path / "devices.json")
    dev = reg.add_manual("127.0.0.1", rport, token="r")
    assert dev.status == "online" and dev.can_chain
    proj, opts = make_project(tmp_path, dev.id)
    p, events = pipeline(lport, proj, opts, reg)
    out = _run(p)
    assert out.parent == proj.outputs_dir and formats.load(out).count == 10
    assert not [c for c in lback.contexts if c.kind == "train"]              # nothing trained here
    files = rback.files[-1]
    assert "images/000003.jpg" in files and "masks/000002.png" in files and "points3d.ply" in files
    assert not any(f.startswith("splatprep-masks") for f in files)
    assert any(k == "job" and d.get("step") for k, d in events)
    rec = json.loads((proj.dataset_dir / compute.UPLOADS_FILE).read_text())
    assert list(rec) == [dev.base_url]
    remote = Client(dev.base_url, "r")
    assert [d["datasetId"] for d in remote.datasets()] == [rec[dev.base_url]["datasetId"]]   # only the photos stay
    assert remote.jobs() == []                                               # the job was cleaned up
    assert json.loads((proj.root / "project.json").read_text())["lastJob"]["device"] == dev.name

    # second run: only dataset.json / masks / points travel
    time.sleep(1.1)
    p2, _ = pipeline(lport, proj, opts, reg)
    out2 = _run(p2)
    assert out2 != out and out2.is_file()
    log = (proj.root / "prep.log").read_text()
    assert "still has the photos" in log
    # the photos are gone from the worker: compose fails, everything is sent again
    remote.delete_dataset(rec[dev.base_url]["datasetId"])
    time.sleep(1.1)
    p3, ev3 = pipeline(lport, proj, opts, reg)
    assert _run(p3).is_file(), ev3


def test_apple_like_worker_gets_one_full_upload(tmp_path, workers):
    lport, _, _ = workers("local", local=True)
    rport, rback, _ = workers("mac", backend=AppleLikeBackend())
    reg = D.DeviceRegistry(tmp_path / "devices.json")
    dev = reg.add_manual("127.0.0.1", rport)
    assert not dev.can_chain
    proj, opts = make_project(tmp_path, dev.id)
    p, _ = pipeline(lport, proj, opts, reg)
    assert _run(p).is_file()
    assert not (proj.dataset_dir / compute.UPLOADS_FILE).exists()
    assert Client(dev.base_url).datasets() == []


def test_offline_device_falls_back_to_this_pc(tmp_path, workers):
    lport, lback, _ = workers("local", local=True)
    reg = D.DeviceRegistry(tmp_path / "devices.json")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    dev = reg.add_manual("127.0.0.1", dead, check=False)
    proj, opts = make_project(tmp_path, dev.id)
    p, _ = pipeline(lport, proj, opts, reg)
    out = _run(p)
    assert out.is_file() and [c for c in lback.contexts if c.kind == "train"]
    assert "training on this PC instead" in (proj.root / "prep.log").read_text()


def test_block_parallel_on_this_pc_and_another_worker(tmp_path, workers):
    lport, lback, _ = workers("local", local=True)
    rport, rback, _ = workers("remote", backend=RecordingBackend(), token="r")
    reg = D.DeviceRegistry(tmp_path / "devices.json")
    reg.add_manual("127.0.0.1", rport, token="r")
    proj, opts = make_project(tmp_path, "all", n=30)
    p, events = pipeline(lport, proj, opts, reg)
    out = _run(p)
    lt = [c for c in lback.contexts if c.kind == "train"]
    rt = [c for c in rback.contexts if c.kind == "train"]
    assert lt and rt and len(lt) + len(rt) == 4, (proj.root / "prep.log").read_text()   # 2 blocks a device
    for c in lt + rt:
        assert c.spec["region"]["margin"] == 0.15 and c.spec["cameraIds"]
    # the remote got only the photos its blocks need
    need = set().union(*[set(c.spec["cameraIds"]) for c in rt])
    sent = {int(Path(f).stem) for f in rback.files[0] if f.startswith("images/")}
    assert sent == need
    g = formats.load(out)
    assert 0 < g.count <= 40
    jobs = [d for k, d in events if k == "job" and d.get("jobId") == "blocks"]
    assert jobs and jobs[-1]["step"] == jobs[-1]["totalSteps"]
    assert "merged 4 blocks" in (proj.root / "prep.log").read_text()
    assert not list(proj.outputs_dir.glob(".blocks-*")) and not list(proj.root.glob(".upload-*"))


def test_failed_remote_block_is_retried_here(tmp_path, workers):
    lport, lback, _ = workers("local", local=True)
    bad = FakeBackend()
    bad.mode = "fail"
    rport, _, _ = workers("remote", backend=bad)
    reg = D.DeviceRegistry(tmp_path / "devices.json")
    reg.add_manual("127.0.0.1", rport)
    proj, opts = make_project(tmp_path, "all", n=30)
    p, _ = pipeline(lport, proj, opts, reg)
    out = _run(p)
    assert out.is_file()
    assert len([c for c in lback.contexts if c.kind == "train"]) == 4
    assert "retrying it on this PC" in (proj.root / "prep.log").read_text()


def test_small_scene_trains_whole_on_this_pc(tmp_path, workers):
    lport, lback, _ = workers("local", local=True)
    rport, rback, _ = workers("remote")
    reg = D.DeviceRegistry(tmp_path / "devices.json")
    reg.add_manual("127.0.0.1", rport)
    proj, opts = make_project(tmp_path, "all")
    opts.split_above = 400
    p, _ = pipeline(lport, proj, opts, reg)
    assert _run(p).is_file()
    ctx = [c for c in lback.contexts if c.kind == "train"]
    assert len(ctx) == 1 and "region" not in ctx[0].spec
    assert not [c for c in rback.contexts if c.kind == "train"]


def test_cancel_cancels_the_remote_job(tmp_path, workers):
    lport, _, _ = workers("local", local=True)
    slow = FakeBackend()
    slow.mode = "block"
    rport, _, _ = workers("remote", backend=slow)
    reg = D.DeviceRegistry(tmp_path / "devices.json")
    dev = reg.add_manual("127.0.0.1", rport)
    proj, opts = make_project(tmp_path, dev.id)
    p, events = pipeline(lport, proj, opts, reg)
    p._titles = ["Training"]
    t = threading.Thread(target=p.run, daemon=True)
    # run() would redo the dataset: call train via run's machinery by patching _run
    p._run = p.train
    t.start()
    assert slow.started.wait(10)
    assert wait_for(lambda: p.remote_jobs)          # tracked, so quit can cancel it without the poll thread
    remote = Client(dev.base_url)
    p.cancel()                                       # cancels the remote job itself, synchronously
    t.join(10)
    assert ("cancelled", None) in events and not p.remote_jobs
    assert remote.jobs() and wait_for(lambda: all(j["state"] == "cancelled" for j in remote.jobs()))


def test_measured_benchmark_overrides_gpu_table(tmp_path):
    from splatworker.studio import benchmark
    from splatworker.studio.devices import Device, apply_benchmarks, local_device

    rec = {"name": "x", "date": "d", "trainStepsPerSecond": 60.0, "splats": 1, "megapixels": 1.0}
    benchmark.save_benchmark(tmp_path, {**rec, "deviceId": "local"})
    benchmark.save_benchmark(tmp_path, {**rec, "deviceId": "http://10.0.0.7:7878", "trainStepsPerSecond": 10.0})
    me = local_device({"gpus": [{"name": "NVIDIA GeForce RTX 4080"}]})
    other = Device(id="manual:10.0.0.7:7878", name="o", host="10.0.0.7", port=7878,
                   info={"gpus": [{"name": "NVIDIA GeForce RTX 4090"}]})
    plain = Device(id="manual:10.0.0.8:7878", name="p", host="10.0.0.8", port=7878,
                   info={"gpus": [{"name": "NVIDIA GeForce RTX 4090"}]})
    apply_benchmarks([me, other, plain], tmp_path)
    assert me.throughput == 3.0 and other.throughput == 0.5
    assert plain.measured is None and plain.throughput > 0.5
