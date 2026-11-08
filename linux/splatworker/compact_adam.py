"""Adam with reduced-precision moments, so more splats fit in GPU memory (``optimizerPrecision: "compact"``).

Plain fp32 Adam keeps two fp32 moments per parameter: 8 bytes per value, i.e. 472 bytes of optimizer state per
splat at SH degree 3 — twice the splat itself. ``CompactAdam`` has the same update rule as ``torch.optim.Adam``
(betas, eps, bias correction; no weight decay / amsgrad) but stores the moments as

* ``"fp32"`` — fp32 (reference; matches ``torch.optim.Adam``),
* ``"bf16"`` — bfloat16 per value, written with stochastic rounding,
* ``"int8"`` — 8-bit codes with one fp32 scale per splat (row).

State layout and gsplat
-----------------------
gsplat's densification ops (``gsplat/strategy/ops.py`` ``_update_param_with_optimizer``: duplicate / split /
remove / relocate / sample_add / reset_opa) edit every tensor in ``optimizer.state[param]`` except ``"step"``
row by row, assuming dim 0 is the splat index: ``v[sel]``, ``v[idx] = 0``, ``zeros_like(v)`` and
``torch.cat([v, torch.zeros(k, *v.shape[1:])])``. So every state tensor here is per-splat: codes of the
parameter's shape and scales of shape ``(N, 1, …)``. No block layout spans splats. A row of zero codes decodes to
zero moments whatever its scale, and a zero scale decodes to zero whatever its codes, so new / relocated splats
start from fresh moments exactly like with fp32 Adam.

``torch.cat`` with gsplat's fp32 zeros promotes int8 / uint8 / bf16 state to fp32. The values stay exact
(integer codes, bf16 values), so ``restore_dtypes()`` casts them back losslessly; it runs at the start of every
``step()`` and the trainer also calls it right after each densification pass so the promoted copies don't sit in
GPU memory during the next forward / backward.

Quantisation (int8 mode)
------------------------
Requantising an EMA every step with round-to-nearest stalls it: with beta2 = 0.999 the per-step change is far
below one code, so values would never decay. All rounding is therefore *stochastic* and unbiased (the expected
decoded value equals the exact value), so the moments stay unbiased EMAs, with some noise.

* exp_avg_sq (v ≥ 0): uint8 code c on a *logarithmic* grid below one fp32 scale per splat s = row max of v:
  ``v = s · 2^(−(255 − c) / 4)`` for c ≥ 1, ``v = 0`` for c = 0. Levels are 19 % apart in v (9 % in sqrt(v), the
  quantity that divides the step) and span 19 decades of v / 9.6 of sqrt(v) within one splat — gradients of one
  splat's 45 SH coefficients differ by many orders of magnitude, and a linear or power-law grid rounded the
  small ones to v = 0, turning m / sqrt(v) into a huge step. Rounding is stochastic between neighbouring levels
  in the linear domain, so E[v_decoded] = v.
* exp_avg (m): stored *relative to the element's own decoded sqrt(v)*: int8 code c with
  ``m = sign(c) · (c / 127)² · R · sqrt(v)``. For exact Adam from zero state, Cauchy–Schwarz on the two EMAs bounds
  ``|m| / sqrt(v) ≤ R = √((1−β1)² / ((1−β2)(1 − β1²/β2)))`` (≈ 7.3 for 0.9 / 0.999), so this ratio is a
  bounded, per-element-scaled quantity: m inherits v's dynamic range, needs no scale of its own, and the
  update ratio can never blow up even when v is tiny. The square-law grid puts resolution where the ratio
  usually is (|m|/sqrt(v) ≲ 1, and → 0 near convergence: steps of 0.013 at ratio 0.1, 0.04 at 1); stochastic
  rounding between neighbouring levels in the linear domain → E[m_decoded | v] = m.
  (If β1² ≥ β2 the bound doesn't exist; then R = 16 and the ratio is clamped.)

A splat with v codes (or v scale) of zero decodes to v = 0 and m = 0; a row of zero m codes decodes to m = 0.

bf16 mode keeps fp32's exponent range (fp16 would underflow g² of small position gradients) and its 8-bit
mantissa is made unbiased with stochastic rounding (random low 16 bits added before truncation).

The update itself is done in fp32 from the dequantised moments (decode → update → step → encode), fully
vectorised per parameter. Stochastic rounding draws from a private generator, so the trainer's global RNG
stream (backgrounds, MCMC noise) is unchanged by the choice of precision.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

MOMENTS = ("fp32", "bf16", "int8")

_V_Q, _V_K = 255, 4.0       # exp_avg_sq: uint8 log-grid codes, _V_K levels per octave of v
_CHUNK_VALUES = 1 << 24       # values per chunk of a compact update (bounds fp32 temporaries)
_M_Q = 127                  # exp_avg: int8 square-law codes of m / (R · sqrt(v))


def _row_shape(x: torch.Tensor):
    return (x.shape[0],) + (1,) * (x.dim() - 1)


def _row_max(a: torch.Tensor) -> torch.Tensor:
    """Per-row max of a non-negative tensor, shaped ``(N, 1, …)`` (or ``(N,)`` for 1-D)."""
    if a.dim() == 1:
        return a.clone()
    return a.reshape(a.shape[0], -1).amax(dim=1).reshape(_row_shape(a))


def _rand(like: torch.Tensor, gen: Optional[torch.Generator]) -> torch.Tensor:
    return torch.rand(like.shape, device=like.device, dtype=torch.float32, generator=gen)


def ratio_bound(b1: float, b2: float) -> float:
    """Largest |m| / sqrt(v) exact Adam can reach from zero state (Cauchy–Schwarz on the two EMAs)."""
    r = b1 * b1 / b2 if b2 > 0 else math.inf
    if r >= 1:
        return 16.0
    return math.sqrt((1 - b1) ** 2 / ((1 - b2) * (1 - r)))


def _v_level(c: torch.Tensor) -> torch.Tensor:
    """Normalised value of float code c (0 → 0)."""
    return torch.where(c > 0, torch.exp2((c - _V_Q) / _V_K), torch.zeros_like(c))


def quantize_v(v: torch.Tensor, gen=None):
    """Non-negative moment → (uint8 log-grid codes, fp32 per-row scale). Unbiased stochastic rounding."""
    s = _row_max(v)
    safe = torch.where(s > 0, s, torch.ones_like(s))
    a = (v / safe).clamp_(0, 1)
    t = torch.log2(a.clamp_min(1e-38)).mul_(_V_K).add_(_V_Q)
    lo = t.floor_().clamp_(0, _V_Q - 1)
    vlo, vhi = _v_level(lo), _v_level(lo + 1)
    p = ((a - vlo) / (vhi - vlo)).clamp_(0, 1)
    c = lo.add_((_rand(a, gen) < p).to(lo.dtype))
    return c.to(torch.uint8), s


def dequantize_v(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return _v_level(codes.float()) * scale.float()


def quantize_m(m: torch.Tensor, v_decoded: torch.Tensor, bound: float, gen=None) -> torch.Tensor:
    """m → int8 codes c of r = m / (bound · sqrt(v_decoded)) on the square-law grid r = sign(c)·(|c|/127)²,
    unbiased stochastic rounding in the linear domain (0 where v = 0)."""
    u = v_decoded.sqrt()
    r = torch.where(u > 0, m / (u * bound).clamp_min(1e-38), torch.zeros_like(m)).clamp_(-1, 1)
    a = r.abs()
    lo = a.sqrt().mul_(_M_Q).floor_().clamp_(0, _M_Q - 1)
    vlo = (lo / _M_Q).square_()
    vhi = ((lo + 1) / _M_Q).square_()
    p = ((a - vlo) / (vhi - vlo)).clamp_(0, 1)
    c = lo.add_((_rand(a, gen) < p).to(lo.dtype)).mul_(torch.sign(r))
    return c.to(torch.int8)


def dequantize_m(codes: torch.Tensor, v_decoded: torch.Tensor, bound: float) -> torch.Tensor:
    c = codes.float()
    return c * c.abs() * (bound / (_M_Q * _M_Q)) * v_decoded.sqrt()


def to_bf16_stochastic(x: torch.Tensor, gen=None) -> torch.Tensor:
    """fp32 → bf16 with unbiased stochastic rounding (add random low 16 bits, then truncate)."""
    bits = x.float().contiguous().view(torch.int32)
    r = torch.randint(0, 1 << 16, x.shape, device=x.device, dtype=torch.int32, generator=gen)
    return ((bits + r) & -65536).view(torch.float32).to(torch.bfloat16)


# state key -> dtype the optimizer expects, per mode
_STATE_DTYPES = {
    "fp32": {"exp_avg": torch.float32, "exp_avg_sq": torch.float32},
    "bf16": {"exp_avg": torch.bfloat16, "exp_avg_sq": torch.bfloat16},
    "int8": {"exp_avg": torch.int8, "exp_avg_sq": torch.uint8, "exp_avg_sq_scale": torch.float32},
}


def state_bytes_per_row(row_values: int, moments: str) -> int:
    """Optimizer-state bytes per splat for a parameter with ``row_values`` values per splat."""
    if row_values == 0:
        return 0
    if moments == "fp32":
        return 8 * row_values
    if moments == "bf16":
        return 4 * row_values
    if moments == "int8":
        return 2 * row_values + 4      # two 1-byte codes per value + one fp32 row scale
    raise ValueError(moments)


# values per splat of each trained parameter (trainer.py layout)
def param_row_values(sh_degree: int) -> Dict[str, int]:
    return {"means": 3, "scales": 3, "quats": 4, "opacities": 1, "sh0": 3,
            "shN": ((sh_degree + 1) ** 2 - 1) * 3}


def moments_for(name: str, precision: str) -> str:
    """Moment storage for trainer parameter ``name`` under ``optimizerPrecision``."""
    if precision == "fp32":
        return "fp32"
    if precision == "compact":
        return "int8" if name == "shN" else "bf16"
    raise ValueError(f"unknown optimizerPrecision {precision!r}")


def estimate_bytes_per_splat(sh_degree: int, precision: str) -> Dict[str, int]:
    """Training memory per splat: fp32 params + fp32 grads + optimizer state (bytes)."""
    rows = param_row_values(sh_degree)
    params = 4 * sum(rows.values())
    state = sum(state_bytes_per_row(n, moments_for(k, precision)) for k, n in rows.items())
    return {"params": params, "grads": params, "optimizerState": state, "total": 2 * params + state}


class CompactAdam(torch.optim.Optimizer):
    """``torch.optim.Adam`` (no weight decay / amsgrad) with fp32, bf16 or int8 moments. See module docstring."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 moments: str = "fp32", seed: int = 0):
        if moments not in MOMENTS:
            raise ValueError(f"moments must be one of {MOMENTS}")
        if lr < 0 or eps < 0 or not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError("invalid lr / eps / betas")
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, moments=moments))
        self._seed = int(seed)
        self._gens: Dict[str, torch.Generator] = {}

    # ------------------------------------------------------------ helpers
    def _gen(self, device: torch.device) -> torch.Generator:
        key = str(device)
        g = self._gens.get(key)
        if g is None:
            g = torch.Generator(device=device)
            g.manual_seed(self._seed)
            self._gens[key] = g
        return g

    @torch.no_grad()
    def restore_dtypes(self) -> None:
        """Cast state promoted by gsplat's ``torch.cat`` (int8/uint8/bf16 + fp32 zeros → fp32) back. Lossless."""
        for group in self.param_groups:
            want = _STATE_DTYPES[group["moments"]]
            for p in group["params"]:
                st = self.state.get(p)
                if not st:
                    continue
                for k, dt in want.items():
                    v = st.get(k)
                    if v is not None and v.dtype != dt:
                        if dt in (torch.int8, torch.uint8):
                            v = v.round()
                        st[k] = v.to(dt)

    def _init_state(self, p: torch.Tensor, moments: str) -> dict:
        st = {"step": torch.zeros((), dtype=torch.float32)}
        if moments == "int8":
            rs = _row_shape(p)
            st["exp_avg"] = torch.zeros(p.shape, dtype=torch.int8, device=p.device)
            st["exp_avg_sq"] = torch.zeros(p.shape, dtype=torch.uint8, device=p.device)
            st["exp_avg_sq_scale"] = torch.zeros(rs, dtype=torch.float32, device=p.device)
        else:
            dt = _STATE_DTYPES[moments]["exp_avg"]
            st["exp_avg"] = torch.zeros(p.shape, dtype=dt, device=p.device)
            st["exp_avg_sq"] = torch.zeros(p.shape, dtype=dt, device=p.device)
        return st

    def _update(self, p, g, st, sl, moments, b1, b2, eps, lr, bc1, bc2_sqrt) -> None:
        """Adam update of rows ``sl`` of ``p`` (decode → update → step → encode, in place in ``st``)."""
        g = g[sl].float()
        bound = ratio_bound(b1, b2)
        if moments == "int8":
            v = dequantize_v(st["exp_avg_sq"][sl], st["exp_avg_sq_scale"][sl])
            m = dequantize_m(st["exp_avg"][sl], v, bound)
        elif moments == "bf16":
            m = st["exp_avg"][sl].float()
            v = st["exp_avg_sq"][sl].float()
        else:   # fp32: update in place, like torch.optim.Adam
            m, v = st["exp_avg"][sl], st["exp_avg_sq"][sl]
        m.mul_(b1).add_(g, alpha=1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)
        denom = (v.sqrt() / bc2_sqrt).add_(eps)
        p.data[sl].addcdiv_(m.to(p.dtype), denom.to(p.dtype), value=-lr / bc1)
        if moments == "int8":
            gen = self._gen(p.device)
            cv, sv = quantize_v(v, gen)
            st["exp_avg_sq"][sl] = cv
            st["exp_avg_sq_scale"][sl] = sv
            st["exp_avg"][sl] = quantize_m(m, dequantize_v(cv, sv), bound, gen)
        elif moments == "bf16":
            gen = self._gen(p.device)
            st["exp_avg"][sl] = to_bf16_stochastic(m, gen)
            st["exp_avg_sq"][sl] = to_bf16_stochastic(v, gen)

    # ------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.restore_dtypes()
        for group in self.param_groups:
            lr, (b1, b2), eps, moments = group["lr"], group["betas"], group["eps"], group["moments"]
            for p in group["params"]:
                g = p.grad
                if g is None or p.numel() == 0:
                    continue
                if g.is_sparse:
                    raise RuntimeError("CompactAdam does not support sparse gradients")
                st = self.state[p]
                if not st or st["exp_avg"].shape != p.shape:
                    st.clear()
                    st.update(self._init_state(p, moments))
                st["step"] += 1
                t = float(st["step"])
                bc1 = 1 - b1 ** t
                bc2_sqrt = math.sqrt(1 - b2 ** t)
                if moments == "fp32" or p.dim() == 0:
                    self._update(p, g, st, ..., moments, b1, b2, eps, lr, bc1, bc2_sqrt)
                    continue
                # Compact modes decode to fp32 temporaries (~10 per value): process rows in chunks so the
                # transient memory stays bounded (< 1 GB) instead of costing more than the state saves.
                n = p.shape[0]
                rows = max(1, _CHUNK_VALUES // max(1, p[0].numel()))
                for r0 in range(0, n, rows):
                    self._update(p, g, st, slice(r0, min(n, r0 + rows)), moments, b1, b2, eps, lr, bc1, bc2_sqrt)
        return loss

    def state_bytes(self) -> int:
        """Bytes currently held in optimizer state tensors (excluding the scalar step)."""
        total = 0
        for st in self.state.values():
            for k, v in st.items():
                if k != "step" and torch.is_tensor(v):
                    total += v.numel() * v.element_size()
        return total
