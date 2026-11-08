"""OpenGL 4.3 splat renderer (the viewer's engine; see docs/LINUX_APP.md "Renderer choice").

Same pipeline and math as the Mac's Metal viewer (``RenderShaders.swift``):

1. ``project`` compute shader: per splat, 3D covariance → screen ellipse (EWA, 0.3 px dilation, optional
   Mip-Splatting opacity compensation), opacity-aware extent (< 1/255 beyond), SH colour (degree ≤ 3) or a
   display mode (depth / centres / ellipses / opacity), selection highlight;
2. instanced quads in back-to-front order (an SSBO filled by :class:`~.sorting.BackgroundSorter`),
   premultiplied alpha blending;
3. the learned skybox, a full-screen triangle drawn first (``sky_fragment``);
4. lines (training cameras, crop box) on top.

Everything renders into an offscreen framebuffer at ``render_scale`` × the target size and is blitted to the
target (the GtkGLArea's framebuffer, or a screenshot read back with :meth:`read_rgba`).
Must be used with a current GL ≥ 4.3 core context.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from . import gl
from .splats import GaussianCloud, Skybox

MODES = {"color": 0, "depth": 1, "centers": 2, "ellipses": 3, "opacity": 4}

_COMMON = r"""
#version 430 core
struct SplatVertex { vec4 axes; vec2 center; uvec2 color; };
"""

PROJECT_SRC = _COMMON + r"""
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer Pos { vec4 posop[]; };
layout(std430, binding = 1) readonly buffer Cov { vec4 cov[]; };
layout(std430, binding = 2) readonly buffer SH { uint sh[]; };
layout(std430, binding = 3) readonly buffer Flags { uint flags[]; };
layout(std430, binding = 4) writeonly buffer Verts { SplatVertex verts[]; };
uniform mat4 uView;            // world → camera (OpenCV)
uniform vec4 uIntr;            // fx fy cx cy (render pixels)
uniform vec2 uViewport;        // render pixels
uniform vec3 uCamPos;
uniform uint uCount;
uniform uint uMode;
uniform int uShDegree;
uniform uint uShStride;        // halves per splat
uniform float uSplatScale;
uniform float uAlphaMin;
uniform float uNear;
uniform vec2 uDepthRange;      // log-depth colouring range
uniform uint uAntialias;
uniform uint uHighlight;

vec3 turbo(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec4 kR = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kG = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kB = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kR2 = vec2(-152.94239396, 59.28637943);
    const vec2 kG2 = vec2(4.27729857, 2.82956604);
    const vec2 kB2 = vec2(-89.90310912, 27.34824973);
    vec4 v4 = vec4(1.0, t, t * t, t * t * t);
    vec2 v2 = v4.zw * v4.z;
    return vec3(dot(v4, kR) + dot(v2, kR2), dot(v4, kG) + dot(v2, kG2), dot(v4, kB) + dot(v2, kB2));
}

vec3 shc(uint base, int k) {
    uint h = base + uint(k) * 3u;
    vec3 r;
    for (int c = 0; c < 3; c++) {
        uint hh = h + uint(c);
        vec2 two = unpackHalf2x16(sh[hh >> 1]);
        r[c] = (hh & 1u) == 0u ? two.x : two.y;
    }
    return r;
}

vec3 eval_sh(uint i, vec3 d) {
    uint base = i * uShStride;
    vec3 rgb = 0.28209479177387814 * shc(base, 0);
    if (uShDegree > 0) {
        float x = d.x, y = d.y, z = d.z;
        rgb += -0.4886025119029199 * y * shc(base, 1) + 0.4886025119029199 * z * shc(base, 2)
               - 0.4886025119029199 * x * shc(base, 3);
        if (uShDegree > 1) {
            float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
            rgb += 1.0925484305920792 * xy * shc(base, 4) - 1.0925484305920792 * yz * shc(base, 5)
                   + 0.31539156525252005 * (2.0 * zz - xx - yy) * shc(base, 6)
                   - 1.0925484305920792 * xz * shc(base, 7) + 0.5462742152960396 * (xx - yy) * shc(base, 8);
            if (uShDegree > 2) {
                rgb += -0.5900435899266435 * y * (3.0 * xx - yy) * shc(base, 9)
                       + 2.890611442640554 * xy * z * shc(base, 10)
                       - 0.4570457994644658 * y * (4.0 * zz - xx - yy) * shc(base, 11)
                       + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shc(base, 12)
                       - 0.4570457994644658 * x * (4.0 * zz - xx - yy) * shc(base, 13)
                       + 1.445305721320277 * z * (xx - yy) * shc(base, 14)
                       - 0.5900435899266435 * x * (xx - 3.0 * yy) * shc(base, 15);
            }
        }
    }
    return max(rgb + 0.5, 0.0);
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uCount) return;
    SplatVertex v;
    v.axes = vec4(0.0);
    v.center = vec2(-1e9);
    v.color = uvec2(0u);
    uint fl = flags[i];
    vec4 po = posop[i];
    vec3 t = (uView * vec4(po.xyz, 1.0)).xyz;
    if ((fl & 1u) != 0u || t.z < uNear) { verts[i] = v; return; }
    float s2 = uSplatScale * uSplatScale;
    vec4 c0 = cov[2u * i], c1 = cov[2u * i + 1u];
    mat3 S = mat3(c0.x, c0.y, c0.z, c0.y, c0.w, c1.x, c0.z, c1.x, c1.y) * s2;
    mat3 W = mat3(uView);
    float fx = uIntr.x, fy = uIntr.y;
    float limx = 1.3 * (0.5 * uViewport.x / fx), limy = 1.3 * (0.5 * uViewport.y / fy);
    float tx = clamp(t.x / t.z, -limx, limx) * t.z;
    float ty = clamp(t.y / t.z, -limy, limy) * t.z;
    mat3 J = mat3(fx / t.z, 0.0, 0.0,  0.0, fy / t.z, 0.0,  -fx * tx / (t.z * t.z), -fy * ty / (t.z * t.z), 0.0);
    mat3 T = J * W;
    mat3 C = T * S * transpose(T);
    float a = C[0][0], b = C[0][1], c = C[1][1];
    float det0 = a * c - b * b;
    a += 0.3; c += 0.3;
    float det = a * c - b * b;
    if (det <= 0.0) { verts[i] = v; return; }
    float opacity = po.w;
    if (uAntialias != 0u) opacity *= sqrt(max(det0, 0.0) / det);
    if (opacity < uAlphaMin) { verts[i] = v; return; }
    vec2 center = vec2(fx * t.x / t.z + uIntr.z, fy * t.y / t.z + uIntr.w);
    float mid = 0.5 * (a + c);
    float disc = sqrt(max(0.1, mid * mid - det));
    float l1 = mid + disc, l2 = max(mid - disc, 0.1);
    vec2 v1 = abs(b) > 1e-7 ? normalize(vec2(b, l1 - a)) : (a >= c ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
    vec2 v2 = vec2(-v1.y, v1.x);
    float k = min(3.0, sqrt(max(2.0 * log(max(255.0 * opacity, 1.0001)), 0.0)));
    float r1 = k * sqrt(l1), r2 = k * sqrt(l2);
    if (uMode == 2u) { r1 = 2.0; r2 = 2.0; v1 = vec2(1.0, 0.0); v2 = vec2(0.0, 1.0); }
    if (r1 < 0.3) { verts[i] = v; return; }
    if (center.x + r1 < 0.0 || center.y + r1 < 0.0 || center.x - r1 > uViewport.x || center.y - r1 > uViewport.y) {
        verts[i] = v; return;
    }
    vec3 rgb;
    if (uMode == 1u) {
        float dn = (log(t.z) - uDepthRange.x) / max(uDepthRange.y - uDepthRange.x, 1e-6);
        rgb = turbo(1.0 - dn);
    } else if (uMode == 4u) {
        rgb = turbo(po.w);
    } else {
        rgb = eval_sh(i, normalize(po.xyz - uCamPos));
    }
    if ((fl & 2u) != 0u && uHighlight != 0u) rgb = mix(rgb, vec3(1.0, 0.45, 0.1), 0.6);
    if (uMode == 2u) opacity = 1.0;
    v.axes = vec4(v1 * r1, v2 * r2);
    v.center = center;
    v.color = uvec2(packHalf2x16(rgb.rg), packHalf2x16(vec2(rgb.b, opacity)));
    verts[i] = v;
}
"""

SPLAT_VS = _COMMON + r"""
layout(std430, binding = 4) readonly buffer Verts { SplatVertex verts[]; };
layout(std430, binding = 5) readonly buffer Order { uint order[]; };
uniform vec2 uViewport;
out vec2 vUV;
out vec4 vColor;
flat out float vExt;
void main() {
    SplatVertex v = verts[order[gl_InstanceID]];
    vec2 corner = vec2((gl_VertexID & 1) != 0 ? 1.0 : -1.0, (gl_VertexID & 2) != 0 ? 1.0 : -1.0);
    vec2 p = v.center + corner.x * v.axes.xy + corner.y * v.axes.zw;
    gl_Position = vec4(p.x / uViewport.x * 2.0 - 1.0, 1.0 - p.y / uViewport.y * 2.0, 0.0, 1.0);
    vec4 col = vec4(unpackHalf2x16(v.color.x), unpackHalf2x16(v.color.y));
    float k = min(3.0, sqrt(max(2.0 * log(max(255.0 * col.a, 1.0001)), 0.0)));
    vUV = corner * k;
    vExt = k;
    vColor = col;
}
"""

SPLAT_FS = r"""
#version 430 core
in vec2 vUV;
in vec4 vColor;
flat in float vExt;
uniform uint uMode;
out vec4 fragColor;
void main() {
    float r2 = dot(vUV, vUV);
    float a;
    if (uMode == 2u) {
        a = r2 < vExt * vExt ? 1.0 : 0.0;
    } else if (uMode == 3u) {
        float r = sqrt(r2);
        a = (r > 0.85 * vExt && r < vExt) ? 1.0 : 0.0;
    } else {
        a = vColor.a * exp(-0.5 * r2);
    }
    if (a < 1.0 / 255.0) discard;
    a = min(a, 0.99);
    fragColor = vec4(vColor.rgb * a, a);
}
"""

SKY_VS = r"""
#version 430 core
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

SKY_FS = r"""
#version 430 core
uniform mat4 uView;
uniform vec4 uIntr;
uniform vec2 uViewport;
uniform mat3 uBasis;           // rows: right, up, forward
uniform sampler2D uTex;
out vec4 fragColor;
void main() {
    vec2 px = vec2(gl_FragCoord.x, uViewport.y - gl_FragCoord.y);
    vec3 dc = vec3((px.x - uIntr.z) / uIntr.x, (px.y - uIntr.w) / uIntr.y, 1.0);
    mat3 R = mat3(uView);
    vec3 d = normalize(transpose(R) * dc);
    vec3 k = uBasis * d;
    float lon = atan(k.x, k.z), lat = asin(clamp(k.y, -1.0, 1.0));
    vec2 uv = vec2(lon / (2.0 * 3.14159265358979) + 0.5, 0.5 - lat / 3.14159265358979);
    fragColor = vec4(texture(uTex, uv).rgb, 1.0);
}
"""

LINE_VS = r"""
#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
uniform mat4 uViewProj;
out vec4 vColor;
void main() { gl_Position = uViewProj * vec4(aPos, 1.0); vColor = aColor; }
"""

LINE_FS = r"""
#version 430 core
in vec4 vColor;
out vec4 fragColor;
void main() { fragColor = vec4(vColor.rgb * vColor.a, vColor.a); }
"""


@dataclass
class ViewSettings:
    mode: str = "color"
    sh_degree: int = 3
    splat_scale: float = 1.0
    background: Tuple[float, float, float] = (0.06, 0.06, 0.07)
    alpha_min: float = 1 / 255
    antialias: str = "auto"        # auto (as trained) | on | off
    render_scale: float = 1.0
    highlight_selection: bool = True
    show_sky: bool = True


def pack_cloud(g: GaussianCloud) -> dict:
    """CPU-side buffers in the shader's layout."""
    n = g.count
    posop = np.empty((n, 4), np.float32)
    posop[:, :3] = g.positions
    posop[:, 3] = g.opacities
    c = g.covariances()
    cov = np.zeros((n, 8), np.float32)
    cov[:, :6] = c
    k = g.sh.shape[1]
    halves = k * 3 + (k * 3) % 2
    sh = np.zeros((n, halves), np.float16)
    sh[:, :k * 3] = np.clip(g.sh.reshape(n, k * 3), -65000, 65000).astype(np.float16)
    return {"posop": posop, "cov": cov, "sh": sh.view(np.uint32), "stride": halves, "degree": g.sh_degree,
            "count": n}


def frustum_lines(c2ws: np.ndarray, size: float, color=(0.3, 0.75, 1.0, 0.9), aspect: float = 1.5,
                  fov_y_deg: float = 50.0) -> np.ndarray:
    """Line vertices (M, 7: xyz rgba) drawing a small pyramid per camera (row-major OpenCV c2w)."""
    import math

    h = math.tan(math.radians(fov_y_deg) / 2) * size
    w = h * aspect
    corners = np.array([[-w, -h, size], [w, -h, size], [w, h, size], [-w, h, size]])
    out = []
    for m in np.asarray(c2ws, np.float64).reshape(-1, 4, 4):
        o = m[:3, 3]
        cs = corners @ m[:3, :3].T + o
        for i in range(4):
            out += [o, cs[i], cs[i], cs[(i + 1) % 4]]
        out += [cs[0] + (cs[1] - cs[0]) * 0.3, (cs[0] + cs[1]) / 2 - m[:3, 1] * h * 0.4,
                (cs[0] + cs[1]) / 2 - m[:3, 1] * h * 0.4, cs[0] + (cs[1] - cs[0]) * 0.7]   # "up" tick
    if not out:
        return np.zeros((0, 7), np.float32)
    p = np.array(out, np.float32)
    return np.concatenate([p, np.tile(np.array(color, np.float32), (len(p), 1))], 1)


def box_lines(lo, hi, color=(1.0, 0.8, 0.2, 1.0)) -> np.ndarray:
    lo, hi = np.asarray(lo, np.float32), np.asarray(hi, np.float32)
    c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    p = np.array([c[i] for e in edges for i in e], np.float32)
    return np.concatenate([p, np.tile(np.array(color, np.float32), (len(p), 1))], 1)


class SplatRenderer:
    def __init__(self):
        self.ready = False
        self.count = 0
        self.degree = 0
        self.stride = 2
        self.antialiased_scene = False
        self.n_draw = 0
        self.depth_range = (0.0, 1.0)
        self.has_sky = False
        self.sky_basis = np.eye(3, dtype=np.float32)
        self._bufs = {}
        self._fbo = 0
        self._fbo_tex = 0
        self._fbo_size = (0, 0)
        self._line_count = 0
        self.gl_version = ""
        self.gl_renderer = ""

    # --------------------------------------------------------------- setup
    def init_gl(self) -> None:
        major, minor = gl.version()
        self.gl_version = gl.get_string(gl.VERSION)
        self.gl_renderer = gl.get_string(gl.RENDERER)
        if (major, minor) < (4, 3):
            raise gl.GLError(f"OpenGL 4.3 is needed for the viewer; this context is {major}.{minor} "
                             f"({self.gl_renderer})")
        self.p_project = gl.link_program(gl.compile_shader(gl.COMPUTE_SHADER, PROJECT_SRC))
        self.p_splat = gl.link_program(gl.compile_shader(gl.VERTEX_SHADER, SPLAT_VS),
                                       gl.compile_shader(gl.FRAGMENT_SHADER, SPLAT_FS))
        self.p_sky = gl.link_program(gl.compile_shader(gl.VERTEX_SHADER, SKY_VS),
                                     gl.compile_shader(gl.FRAGMENT_SHADER, SKY_FS))
        self.p_line = gl.link_program(gl.compile_shader(gl.VERTEX_SHADER, LINE_VS),
                                      gl.compile_shader(gl.FRAGMENT_SHADER, LINE_FS))
        names = ["posop", "cov", "sh", "flags", "verts", "order", "lines"]
        for name, b in zip(names, gl.gen("glGenBuffers", len(names))):
            self._bufs[name] = b
        self.vao_empty, self.vao_lines = gl.gen("glGenVertexArrays", 2)
        gl.glBindVertexArray(self.vao_lines)
        gl.glBindBuffer(gl.ARRAY_BUFFER, self._bufs["lines"])
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.FLOAT, 0, 28, None)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 4, gl.FLOAT, 0, 28, ctypes.c_void_p(12))
        gl.glBindVertexArray(0)
        self.sky_tex = gl.gen("glGenTextures")[0]
        self._uniforms = {}
        # empty buffers so binding never fails
        for name in ("posop", "cov", "sh", "flags", "verts", "order"):
            self._upload(name, np.zeros(16, np.uint32))
        self.ready = True
        gl.check("init")

    def destroy(self) -> None:
        if not self.ready:
            return
        gl.delete("glDeleteBuffers", self._bufs.values())
        gl.delete("glDeleteVertexArrays", [self.vao_empty, self.vao_lines])
        gl.delete("glDeleteTextures", [self.sky_tex, self._fbo_tex])
        gl.delete("glDeleteFramebuffers", [self._fbo])
        for p in (self.p_project, self.p_splat, self.p_sky, self.p_line):
            gl.glDeleteProgram(p)
        self.ready = False

    def _u(self, prog: int, name: str) -> int:
        key = (prog, name)
        loc = self._uniforms.get(key)
        if loc is None:
            loc = gl.glGetUniformLocation(prog, name.encode())
            self._uniforms[key] = loc
        return loc

    def _upload(self, name: str, arr: np.ndarray, target=gl.SHADER_STORAGE_BUFFER) -> None:
        arr = np.ascontiguousarray(arr)
        gl.glBindBuffer(target, self._bufs[name])
        size = max(arr.nbytes, 16)
        gl.glBufferData(target, size, arr.ctypes.data_as(ctypes.c_void_p) if arr.nbytes else None, gl.STATIC_DRAW)
        gl.glBindBuffer(target, 0)

    # --------------------------------------------------------------- data
    def set_cloud(self, g: Optional[GaussianCloud], packed: Optional[dict] = None) -> None:
        if g is None:
            self.count = 0
            self.n_draw = 0
            return
        p = packed or pack_cloud(g)
        self._upload("posop", p["posop"])
        self._upload("cov", p["cov"])
        self._upload("sh", p["sh"])
        self.count, self.degree, self.stride = p["count"], p["degree"], p["stride"]
        self.antialiased_scene = bool(g.antialiased)
        gl.glBindBuffer(gl.SHADER_STORAGE_BUFFER, self._bufs["verts"])
        gl.glBufferData(gl.SHADER_STORAGE_BUFFER, max(self.count * 32, 32), None, gl.DYNAMIC_DRAW)
        gl.glBindBuffer(gl.SHADER_STORAGE_BUFFER, 0)
        self.set_flags(np.zeros(self.count, np.uint32))
        self.set_order(np.arange(self.count, dtype=np.uint32), 0)
        self.set_sky(g.skybox)

    def set_flags(self, flags: np.ndarray) -> None:
        """uint32 per splat: bit 0 hidden (deleted / cropped), bit 1 selected."""
        self._upload("flags", flags.astype(np.uint32))

    @staticmethod
    def make_flags(hidden: Optional[np.ndarray], selected: Optional[np.ndarray], n: int) -> np.ndarray:
        f = np.zeros(n, np.uint32)
        if hidden is not None:
            f |= hidden.astype(np.uint32)
        if selected is not None:
            f |= selected.astype(np.uint32) << 1
        return f

    def set_order(self, order: np.ndarray, n_draw: int) -> None:
        self._upload("order", order.astype(np.uint32))
        self.n_draw = int(n_draw)

    def set_sky(self, sky: Optional[Skybox]) -> None:
        self.has_sky = sky is not None
        if sky is None:
            return
        img = np.ascontiguousarray(sky.image[..., :3], np.uint8)
        gl.glBindTexture(gl.TEXTURE_2D, self.sky_tex)
        gl.glPixelStorei(gl.UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.TEXTURE_2D, 0, gl.RGB8, img.shape[1], img.shape[0], 0, gl.RGB, gl.UNSIGNED_BYTE,
                        img.ctypes.data_as(ctypes.c_void_p))
        gl.glTexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
        gl.glTexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
        gl.glTexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT)
        gl.glTexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
        gl.glBindTexture(gl.TEXTURE_2D, 0)
        self.sky_basis = sky.basis

    def set_lines(self, verts: Optional[np.ndarray]) -> None:
        v = np.zeros((0, 7), np.float32) if verts is None else np.ascontiguousarray(verts, np.float32)
        self._upload("lines", v if len(v) else np.zeros(7, np.float32), gl.ARRAY_BUFFER)
        self._line_count = len(v)

    # --------------------------------------------------------------- frame
    def _ensure_fbo(self, w: int, h: int) -> None:
        if self._fbo and self._fbo_size == (w, h):
            return
        if not self._fbo:
            self._fbo = gl.gen("glGenFramebuffers")[0]
            self._fbo_tex = gl.gen("glGenTextures")[0]
        gl.glBindTexture(gl.TEXTURE_2D, self._fbo_tex)
        gl.glTexImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
        gl.glTexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
        gl.glBindTexture(gl.TEXTURE_2D, 0)
        gl.glBindFramebuffer(gl.FRAMEBUFFER, self._fbo)
        gl.glFramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, self._fbo_tex, 0)
        status = gl.glCheckFramebufferStatus(gl.FRAMEBUFFER)
        gl.glBindFramebuffer(gl.FRAMEBUFFER, 0)
        if status != gl.FRAMEBUFFER_COMPLETE:
            raise gl.GLError(f"offscreen framebuffer incomplete (0x{status:x})")
        self._fbo_size = (w, h)

    def render(self, camera, width: int, height: int, settings: ViewSettings, target_fbo: Optional[int] = None,
               blit: bool = True) -> Tuple[int, int]:
        """Draw one frame for ``camera`` into a ``width``×``height`` target. Returns the render size used."""
        if target_fbo is None:
            target_fbo = gl.get_int(gl.FRAMEBUFFER_BINDING)
        rs = max(0.1, min(1.0, float(settings.render_scale)))
        rw, rh = max(1, int(round(width * rs))), max(1, int(round(height * rs)))
        self._ensure_fbo(rw, rh)
        fx, fy, cx, cy = camera.intrinsics(rw, rh)
        view = np.ascontiguousarray(camera.view_matrix(), np.float32)
        pos = camera.position()
        bg = settings.background

        gl.glBindFramebuffer(gl.FRAMEBUFFER, self._fbo)
        gl.glViewport(0, 0, rw, rh)
        gl.glDisable(gl.DEPTH_TEST)
        gl.glDisable(gl.CULL_FACE)
        gl.glClearColor(bg[0], bg[1], bg[2], 1.0)
        gl.glClear(gl.COLOR_BUFFER_BIT)

        mode = MODES.get(settings.mode, 0)
        if self.has_sky and settings.show_sky and mode == 0:
            gl.glUseProgram(self.p_sky)
            gl.glUniformMatrix4fv(self._u(self.p_sky, "uView"), 1, 1, view.ctypes.data_as(ctypes.c_void_p))
            gl.glUniform4f(self._u(self.p_sky, "uIntr"), fx, fy, cx, cy)
            gl.glUniform2f(self._u(self.p_sky, "uViewport"), rw, rh)
            basis = np.ascontiguousarray(self.sky_basis, np.float32)
            gl.glUniformMatrix3fv(self._u(self.p_sky, "uBasis"), 1, 1, basis.ctypes.data_as(ctypes.c_void_p))
            gl.glActiveTexture(gl.TEXTURE0)
            gl.glBindTexture(gl.TEXTURE_2D, self.sky_tex)
            gl.glUniform1i(self._u(self.p_sky, "uTex"), 0)
            gl.glBindVertexArray(self.vao_empty)
            gl.glDrawArrays(gl.TRIANGLES, 0, 3)

        if self.count and self.n_draw:
            p = self.p_project
            gl.glUseProgram(p)
            for binding, name in enumerate(("posop", "cov", "sh", "flags", "verts")):
                gl.glBindBufferBase(gl.SHADER_STORAGE_BUFFER, binding, self._bufs[name])
            gl.glUniformMatrix4fv(self._u(p, "uView"), 1, 1, view.ctypes.data_as(ctypes.c_void_p))
            gl.glUniform4f(self._u(p, "uIntr"), fx, fy, cx, cy)
            gl.glUniform2f(self._u(p, "uViewport"), rw, rh)
            gl.glUniform3f(self._u(p, "uCamPos"), *[float(x) for x in pos])
            gl.glUniform1ui(self._u(p, "uCount"), self.count)
            gl.glUniform1ui(self._u(p, "uMode"), mode)
            gl.glUniform1i(self._u(p, "uShDegree"), min(self.degree, max(0, int(settings.sh_degree))))
            gl.glUniform1ui(self._u(p, "uShStride"), self.stride)
            gl.glUniform1f(self._u(p, "uSplatScale"), float(settings.splat_scale))
            gl.glUniform1f(self._u(p, "uAlphaMin"), float(settings.alpha_min))
            gl.glUniform1f(self._u(p, "uNear"), float(camera.near))
            gl.glUniform2f(self._u(p, "uDepthRange"), *[float(x) for x in self.depth_range])
            aa = self.antialiased_scene if settings.antialias == "auto" else settings.antialias == "on"
            gl.glUniform1ui(self._u(p, "uAntialias"), 1 if aa else 0)
            gl.glUniform1ui(self._u(p, "uHighlight"), 1 if settings.highlight_selection else 0)
            gl.glDispatchCompute((self.count + 255) // 256, 1, 1)
            gl.glMemoryBarrier(gl.SHADER_STORAGE_BARRIER_BIT)

            gl.glEnable(gl.BLEND)
            gl.glBlendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ZERO, gl.ONE)
            gl.glUseProgram(self.p_splat)
            gl.glBindBufferBase(gl.SHADER_STORAGE_BUFFER, 4, self._bufs["verts"])
            gl.glBindBufferBase(gl.SHADER_STORAGE_BUFFER, 5, self._bufs["order"])
            gl.glUniform2f(self._u(self.p_splat, "uViewport"), rw, rh)
            gl.glUniform1ui(self._u(self.p_splat, "uMode"), mode)
            gl.glBindVertexArray(self.vao_empty)
            gl.glDrawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, min(self.n_draw, self.count))
            gl.glDisable(gl.BLEND)

        if self._line_count:
            gl.glEnable(gl.BLEND)
            gl.glBlendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ZERO, gl.ONE)
            gl.glUseProgram(self.p_line)
            vp = np.ascontiguousarray(camera.gl_view_proj(rw, rh), np.float32)
            gl.glUniformMatrix4fv(self._u(self.p_line, "uViewProj"), 1, 1, vp.ctypes.data_as(ctypes.c_void_p))
            gl.glBindVertexArray(self.vao_lines)
            gl.glDrawArrays(gl.LINES, 0, self._line_count)
            gl.glDisable(gl.BLEND)

        gl.glBindVertexArray(0)
        gl.glUseProgram(0)
        if blit:
            gl.glBindFramebuffer(gl.READ_FRAMEBUFFER, self._fbo)
            gl.glBindFramebuffer(gl.DRAW_FRAMEBUFFER, target_fbo)
            gl.glBlitFramebuffer(0, 0, rw, rh, 0, 0, width, height, gl.COLOR_BUFFER_BIT,
                                 gl.LINEAR if (rw, rh) != (width, height) else gl.NEAREST)
        gl.glBindFramebuffer(gl.FRAMEBUFFER, target_fbo)
        return rw, rh

    def read_rgba(self) -> np.ndarray:
        """The last frame's offscreen image, (H, W, 4) uint8, top row first."""
        w, h = self._fbo_size
        out = np.empty((h, w, 4), np.uint8)
        prev = gl.get_int(gl.FRAMEBUFFER_BINDING)
        gl.glBindFramebuffer(gl.FRAMEBUFFER, self._fbo)
        gl.glPixelStorei(gl.PACK_ALIGNMENT, 1)
        gl.glReadPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, out.ctypes.data_as(ctypes.c_void_p))
        gl.glBindFramebuffer(gl.FRAMEBUFFER, prev)
        return out[::-1].copy()


def render_offscreen(renderer: SplatRenderer, camera, width: int, height: int, settings: ViewSettings) -> np.ndarray:
    """Screenshot at an arbitrary resolution (render scale 1)."""
    s = ViewSettings(**{**settings.__dict__, "render_scale": 1.0})
    renderer.render(camera, width, height, s, target_fbo=0, blit=False)
    return renderer.read_rgba()
