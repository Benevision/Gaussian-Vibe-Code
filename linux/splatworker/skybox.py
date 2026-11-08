"""Learned skybox: an equirectangular panorama at infinity, drawn behind the splats.

Splats can't represent the sky (kilometres away, seen from every camera at once): they paint it as haze with
shards floating in front of the scene. The skybox gives every pixel a background colour looked up by its viewing
direction alone — ``pred = splats + (1 − splat alpha) · sky(direction)`` — and, where a photo's sky mask says
"sky", a small penalty keeps splats out (their alpha → 0), so the sky is explained by the panorama.

Orientation: latitude from the dataset's up vector, longitude from a fixed horizontal axis, written next to the
PLY as ``<name>.sky.png`` (8-bit sRGB, 2:1) with ``<name>.sky.json`` (up / forward / right vectors) so viewers
can draw it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def basis(up) -> tuple:
    """(up, forward, right) unit vectors: forward ⟂ up, as close to world +Z as possible (else +X)."""
    u = np.asarray(up, np.float64)
    u = u / (np.linalg.norm(u) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    f = ref - u * float(ref @ u)
    f /= np.linalg.norm(f) + 1e-12
    r = np.cross(u, f)
    return u, f, r


class Skybox:
    def __init__(self, torch, up, height: int = 256, init_rgb=(0.6, 0.65, 0.7), lr: float = 0.01, device="cuda"):
        self.torch = torch
        u, f, r = basis(up)
        self.up, self.forward, self.right = u, f, r
        self.B = torch.tensor(np.stack([r, u, f]), dtype=torch.float32, device=device)     # world → (x, y, z) sky
        h = int(height)
        init = torch.tensor(init_rgb, dtype=torch.float32, device=device).clamp(0.02, 0.98)
        logit = torch.log(init / (1 - init))
        self.tex = torch.nn.Parameter(logit.view(1, 3, 1, 1).repeat(1, 1, h, 2 * h).contiguous())
        self.opt = torch.optim.Adam([self.tex], lr=lr, eps=1e-15)
        self._grid_cache = {}

    def directions(self, K, w2c, width: int, height: int):
        """World-space unit view directions per pixel, (H, W, 3)."""
        torch = self.torch
        key = (width, height, str(K.device))
        if key not in self._grid_cache:
            ys, xs = torch.meshgrid(torch.arange(height, device=K.device, dtype=torch.float32) + 0.5,
                                    torch.arange(width, device=K.device, dtype=torch.float32) + 0.5, indexing="ij")
            self._grid_cache = {key: (xs, ys)}
        xs, ys = self._grid_cache[key]
        d = torch.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], torch.ones_like(xs)], -1)
        d = d @ w2c[:3, :3]                     # camera → world: Rᵀ·d as a row vector
        return d / d.norm(dim=-1, keepdim=True)

    def sample(self, dirs):
        """Sky colour (H, W, 3) in [0, 1] for world directions (H, W, 3)."""
        torch = self.torch
        s = dirs @ self.B.T                      # (x: right, y: up, z: forward)
        lon = torch.atan2(s[..., 0], s[..., 2])                  # −π … π
        lat = torch.asin(s[..., 1].clamp(-1, 1))                 # −π/2 … π/2
        grid = torch.stack([lon / math.pi, -lat / (math.pi / 2)], -1)[None]   # top row = straight up
        rgb = torch.nn.functional.grid_sample(torch.sigmoid(self.tex), grid, mode="bilinear",
                                              padding_mode="border", align_corners=False)
        return rgb[0].permute(1, 2, 0)

    def step(self):
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    def save(self, ply_path: Path) -> Path:
        from PIL import Image

        torch = self.torch
        with torch.no_grad():
            img = (torch.sigmoid(self.tex)[0].permute(1, 2, 0).clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()
        png = Path(ply_path).with_suffix(".sky.png")
        Image.fromarray(img).save(png)
        Path(ply_path).with_suffix(".sky.json").write_text(json.dumps({
            "format": "equirectangular", "image": png.name,
            "up": self.up.tolist(), "forward": self.forward.tolist(), "right": self.right.tolist(),
            "longitude": "atan2(dir·right, dir·forward), left edge = −180°", "latitude": "asin(dir·up), top = +90°"}))
        return png
