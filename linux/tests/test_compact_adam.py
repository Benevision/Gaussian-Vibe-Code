"""Reduced-precision Adam (optimizerPrecision) and its compatibility with gsplat's state surgery."""
import pytest

# ------------------------------------------------------------------ optimizer

torch = pytest.importorskip("torch")

from splatworker.compact_adam import (CompactAdam, dequantize_m, dequantize_v,  # noqa: E402
                                      estimate_bytes_per_splat, moments_for, quantize_m, quantize_v,
                                      ratio_bound)


def _quadratic(seed=0, n=64, d=45):
    """Badly conditioned separable quadratic 0.5·Σ a·(x − c)², a spanning 4 decades per row."""
    g = torch.Generator().manual_seed(seed)
    a = 10 ** (torch.rand(n, d, generator=g) * 4 - 2)
    c = torch.randn(n, d, generator=g)
    x0 = torch.randn(n, d, generator=g)
    return a, c, x0


def _run(opt_factory, steps=1500, seed=0):
    a, c, x0 = _quadratic(seed)
    x = torch.nn.Parameter(x0.clone())
    opt = opt_factory([x])
    for _ in range(steps):
        loss = 0.5 * (a * (x - c) ** 2).sum()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    return x.detach(), c, a


def test_fp32_matches_torch_adam():
    ref, c, _ = _run(lambda p: torch.optim.Adam(p, lr=1e-2, eps=1e-15), steps=300)
    ours, _, _ = _run(lambda p: CompactAdam(p, lr=1e-2, eps=1e-15, moments="fp32"), steps=300)
    assert torch.allclose(ours, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("moments,tol", [("bf16", 0.01), ("int8", 0.02)])
def test_compact_modes_converge(moments, tol):
    ref, c, a = _run(lambda p: torch.optim.Adam(p, lr=1e-2, eps=1e-15))
    ours, _, _ = _run(lambda p: CompactAdam(p, lr=1e-2, eps=1e-15, moments=moments))
    err_ref = (ref - c).abs().max().item()
    err = (ours - c).abs().max().item()
    assert err < tol, (moments, err, err_ref)
    # loss relative to the start
    x0 = _quadratic(0)[2]
    l0 = (a * (x0 - c) ** 2).sum()
    assert (a * (ours - c) ** 2).sum() / l0 < 1e-4


@pytest.mark.parametrize("moments", ["bf16", "int8"])
def test_chunked_update_converges(moments, monkeypatch):
    import splatworker.compact_adam as ca
    monkeypatch.setattr(ca, "_CHUNK_VALUES", 100)   # 64 rows of 45 values → 2 rows per chunk
    ours, c, _ = _run(lambda p: CompactAdam(p, lr=1e-2, eps=1e-15, moments=moments))
    assert (ours - c).abs().max().item() < 0.02


def test_compact_tracks_adam_trajectory_early():
    """Over the first steps (moments far from converged) compact steps stay close to exact Adam."""
    ref, _, _ = _run(lambda p: torch.optim.Adam(p, lr=1e-3, eps=1e-15), steps=50)
    for moments in ("bf16", "int8"):
        ours, _, _ = _run(lambda p: CompactAdam(p, lr=1e-3, eps=1e-15, moments=moments), steps=50)
        # total movement is ≤ 50·lr = 0.05; deviation must be a small fraction of it
        assert (ours - ref).abs().max().item() < 0.01, moments


def test_quantizers_unbiased_and_zero_rows():
    torch.manual_seed(1)
    # each row spans 8 decades of gradient magnitude (SH coefficients of one splat differ a lot)
    g0 = torch.randn(8, 15, 3) * 10 ** (torch.rand(8, 15, 3) * 8 - 6)
    g0[3] = 0
    v = g0 * g0
    m = 0.5 * g0          # |m| / sqrt(v) = 0.5 ≤ R
    R = ratio_bound(0.9, 0.999)
    assert 7.0 < R < 7.6
    g = torch.Generator().manual_seed(0)
    acc_m = torch.zeros_like(m)
    acc_v = torch.zeros_like(v)
    k = 400
    for _ in range(k):
        cv, sv = quantize_v(v, g)
        vd = dequantize_v(cv, sv)
        cm = quantize_m(m, vd, R, g)
        assert cm.dtype == torch.int8 and cv.dtype == torch.uint8 and sv.shape == (8, 1, 1)
        acc_m += dequantize_m(cm, vd, R)
        acc_v += vd
    # unbiased: the mean of many encodings approaches the exact value, *relative to each element* (log grid)
    nz = v > 0
    assert ((acc_v / k - v).abs()[nz] / v[nz]).max() < 0.05
    assert ((acc_m / k - m).abs()[nz] / m.abs()[nz]).max() < 0.1
    # a single decode is within one level (19 %) of v
    assert ((vd[nz] / v[nz]) - 1).abs().max() < 0.2
    assert torch.all(acc_m[3] == 0) and torch.all(acc_v[3] == 0)
    # zero v codes or zero v scale decode to zero v and m; zero m codes decode to zero m
    for vz in (dequantize_v(torch.zeros_like(cv), sv), dequantize_v(cv, torch.zeros_like(sv))):
        assert torch.all(vz == 0) and torch.all(dequantize_m(cm, vz, R) == 0)
    assert torch.all(dequantize_m(torch.zeros_like(cm), vd, R) == 0)


# ------------------------------------------------------------------ gsplat-style state surgery

def _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers, names=None):
    """Same loop as gsplat/strategy/ops.py."""
    for name in names or list(params.keys()):
        param = params[name]
        new_param = param_fn(name, param)
        params[name] = new_param
        optimizer = optimizers[name]
        for i in range(len(optimizer.param_groups)):
            param_state = optimizer.state[param]
            del optimizer.state[param]
            for key in param_state.keys():
                if key != "step":
                    param_state[key] = optimizer_fn(key, param_state[key])
            optimizer.param_groups[i]["params"] = [new_param]
            optimizer.state[new_param] = param_state


def _decoded(opt, p):
    st = opt.state[p]
    if "exp_avg_sq_scale" in st:
        v = dequantize_v(st["exp_avg_sq"], st["exp_avg_sq_scale"])
        return dequantize_m(st["exp_avg"], v, ratio_bound(0.9, 0.999)), v
    return st["exp_avg"].float(), st["exp_avg_sq"].float()


@pytest.mark.parametrize("precision", ["fp32", "compact"])
def test_gsplat_row_ops_on_state(precision):
    torch.manual_seed(0)
    n = 32
    target = {"means": torch.randn(n, 3), "opacities": torch.randn(n), "shN": torch.randn(n, 15, 3)}
    params = {k: torch.nn.Parameter(torch.zeros_like(v)) for k, v in target.items()}
    opts = {k: CompactAdam([params[k]], lr=1e-2, eps=1e-15, moments=moments_for(k, precision))
            for k in params}

    def train(steps):
        for _ in range(steps):
            loss = sum(((params[k] - target[k]) ** 2).sum() for k in params)
            loss.backward()
            for o in opts.values():
                o.step()
                o.zero_grad(set_to_none=True)

    train(20)
    # sample_add / duplicate: cat fp32 zeros rows (promotes compact state to fp32)
    add = torch.tensor([1, 5, 5, 7])
    _update_param_with_optimizer(
        lambda name, p: torch.nn.Parameter(torch.cat([p, p[add]])),
        lambda key, v: torch.cat([v, torch.zeros((len(add), *v.shape[1:]))]), params, opts)
    for k, o in opts.items():
        p = params[k]
        o.restore_dtypes()
        m, v = _decoded(o, p)
        assert m.shape == p.shape and torch.all(m[n:] == 0) and torch.all(v[n:] == 0)
        if precision == "compact":
            assert o.state[p]["exp_avg"].dtype in (torch.int8, torch.bfloat16)
    # relocate: zero some rows in place
    dead = torch.tensor([0, 2, 3])
    def zero_rows(key, v):
        v[dead] = 0
        return v
    _update_param_with_optimizer(lambda name, p: torch.nn.Parameter(p), zero_rows, params, opts)
    for k, o in opts.items():
        m, v = _decoded(o, params[k])
        assert torch.all(m[dead] == 0) and torch.all(v[dead] == 0)
        assert m[1].abs().sum() > 0   # untouched rows kept their moments
    # remove: index-select surviving rows
    keep = torch.arange(params["means"].shape[0]) % 3 != 1
    _update_param_with_optimizer(lambda name, p: torch.nn.Parameter(p[keep]),
                                 lambda key, v: v[keep], params, opts)
    # reset_opa: zeros_like on one parameter's state
    _update_param_with_optimizer(lambda name, p: p, lambda key, v: torch.zeros_like(v), params, opts,
                                 names=["opacities"])
    m, v = _decoded(opts["opacities"], params["opacities"])
    assert torch.all(m == 0) and torch.all(v == 0)
    # split-style: cat after an index-select, without calling restore_dtypes (step() must cope)
    _update_param_with_optimizer(
        lambda name, p: torch.nn.Parameter(torch.cat([p, p[:2]])),
        lambda key, v: torch.cat([v, torch.zeros((2, *v.shape[1:]))]), params, opts)

    # the optimizer keeps working: fit new targets of the new shape (fresh rows included)
    for k in params:
        target[k] = torch.randn_like(params[k])
    train(1500)
    for k in params:
        assert torch.isfinite(params[k]).all()
        assert (params[k] - target[k]).abs().max() < 0.05, k
    for k, o in opts.items():
        m, v = _decoded(o, params[k])
        assert m.shape == params[k].shape and v.shape == params[k].shape
        if precision == "compact":
            want = torch.int8 if k == "shN" else torch.bfloat16
            assert o.state[params[k]]["exp_avg"].dtype == want


def test_bytes_per_splat_estimate_sh3():
    fp32 = estimate_bytes_per_splat(3, "fp32")
    compact = estimate_bytes_per_splat(3, "compact")
    assert fp32["optimizerState"] == 472 and fp32["total"] == 944
    assert compact["optimizerState"] == 150 and compact["total"] == 622


def test_sh_degree_zero_empty_shn():
    p = torch.nn.Parameter(torch.zeros(10, 0, 3))
    opt = CompactAdam([p], lr=1e-2, moments="int8")
    (p.sum() * 0).backward()
    opt.step()   # nothing to do, must not crash


@pytest.mark.parametrize("moments", ["bf16", "int8"])
def test_state_survives_save_and_load(moments, tmp_path):
    """Pause / resume: the trainer saves optimizer state with torch.save and reloads it with weights_only=True;
    load_state_dict casts the compact moments to fp32, restore_dtypes() must bring them back losslessly."""
    x = torch.nn.Parameter(torch.randn(16, 45))
    opt = CompactAdam([{"params": [x], "lr": 1e-2, "name": "shN"}], moments=moments)
    for _ in range(3):
        (x ** 2).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.save({"opt": opt.state_dict()}, tmp_path / "s.pt")
    y = torch.nn.Parameter(x.detach().clone())
    opt2 = CompactAdam([{"params": [y], "lr": 1e-2, "name": "shN"}], moments=moments)
    opt2.load_state_dict(torch.load(tmp_path / "s.pt", map_location="cpu", weights_only=True)["opt"])
    opt2.restore_dtypes()
    a, b = opt.state[x], opt2.state[y]
    for k in a:
        assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k
