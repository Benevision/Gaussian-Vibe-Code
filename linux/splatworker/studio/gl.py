"""Minimal OpenGL 4.3 core bindings with ctypes (no PyOpenGL dependency).

GTK 4 creates its GL contexts with EGL (Wayland and X11), so entry points are resolved with
``eglGetProcAddress`` (EGL 1.5 / ``EGL_KHR_get_all_proc_addresses`` return core functions too), falling back to
``glXGetProcAddressARB`` and finally to symbols exported by ``libOpenGL.so.0`` / ``libGL.so.1``. Functions are
resolved lazily, on first use, while a context is current.

:func:`headless_context` makes a surfaceless EGL context for tests and offscreen rendering (Mesa llvmpipe in CI,
the NVIDIA driver on the PC).
"""
from __future__ import annotations

import ctypes
import ctypes.util
from typing import Dict, Optional

c_uint, c_int, c_float, c_void_p, c_char_p = ctypes.c_uint, ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_char_p
c_size = ctypes.c_ssize_t
GLenum = GLuint = GLbitfield = c_uint
GLint = GLsizei = c_int

# --- constants -----------------------------------------------------------------------------------------------
FALSE, TRUE = 0, 1
COLOR_BUFFER_BIT = 0x4000
DEPTH_BUFFER_BIT = 0x0100
TRIANGLES, TRIANGLE_STRIP, LINES = 0x0004, 0x0005, 0x0001
BLEND, DEPTH_TEST, CULL_FACE, SCISSOR_TEST = 0x0BE2, 0x0B71, 0x0B44, 0x0C11
ONE, ZERO, ONE_MINUS_SRC_ALPHA, SRC_ALPHA = 1, 0, 0x0303, 0x0302
VERTEX_SHADER, FRAGMENT_SHADER, COMPUTE_SHADER = 0x8B31, 0x8B30, 0x91B9
COMPILE_STATUS, LINK_STATUS, INFO_LOG_LENGTH = 0x8B81, 0x8B82, 0x8B84
SHADER_STORAGE_BUFFER, ARRAY_BUFFER = 0x90D2, 0x8892
STATIC_DRAW, DYNAMIC_DRAW, STREAM_DRAW = 0x88E4, 0x88E8, 0x88E0
SHADER_STORAGE_BARRIER_BIT = 0x00002000
VERTEX_ATTRIB_ARRAY_BARRIER_BIT = 0x00000001
TEXTURE_2D, TEXTURE0 = 0x0DE1, 0x84C0
TEXTURE_MIN_FILTER, TEXTURE_MAG_FILTER, TEXTURE_WRAP_S, TEXTURE_WRAP_T = 0x2801, 0x2800, 0x2802, 0x2803
LINEAR, NEAREST, REPEAT, CLAMP_TO_EDGE = 0x2601, 0x2600, 0x2901, 0x812F
RGB, RGBA, RGBA8, RGB8, UNSIGNED_BYTE, FLOAT = 0x1907, 0x1908, 0x8058, 0x8051, 0x1401, 0x1406
UNPACK_ALIGNMENT, PACK_ALIGNMENT = 0x0CF5, 0x0D05
FRAMEBUFFER, READ_FRAMEBUFFER, DRAW_FRAMEBUFFER = 0x8D40, 0x8CA8, 0x8CA9
FRAMEBUFFER_BINDING, READ_FRAMEBUFFER_BINDING = 0x8CA6, 0x8CAA
COLOR_ATTACHMENT0, FRAMEBUFFER_COMPLETE = 0x8CE0, 0x8CD5
VERSION, RENDERER, VENDOR, SHADING_LANGUAGE_VERSION = 0x1F02, 0x1F01, 0x1F00, 0x8B8C
MAJOR_VERSION, MINOR_VERSION, VIEWPORT = 0x821B, 0x821C, 0x0BA2
NO_ERROR = 0
MAX_SHADER_STORAGE_BLOCK_SIZE = 0x90DE

# name: (restype, argtypes)
_SIGS = {
    "glGetError": (GLenum, []),
    "glGetString": (c_char_p, [GLenum]),
    "glGetIntegerv": (None, [GLenum, ctypes.POINTER(GLint)]),
    "glGetInteger64v": (None, [GLenum, ctypes.POINTER(ctypes.c_int64)]),
    "glEnable": (None, [GLenum]), "glDisable": (None, [GLenum]),
    "glBlendFunc": (None, [GLenum, GLenum]),
    "glBlendFuncSeparate": (None, [GLenum, GLenum, GLenum, GLenum]),
    "glClearColor": (None, [c_float, c_float, c_float, c_float]),
    "glClear": (None, [GLbitfield]),
    "glViewport": (None, [GLint, GLint, GLsizei, GLsizei]),
    "glFinish": (None, []), "glFlush": (None, []),
    "glPixelStorei": (None, [GLenum, GLint]),
    "glReadPixels": (None, [GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, c_void_p]),
    "glCreateShader": (GLuint, [GLenum]),
    "glShaderSource": (None, [GLuint, GLsizei, ctypes.POINTER(c_char_p), ctypes.POINTER(GLint)]),
    "glCompileShader": (None, [GLuint]),
    "glGetShaderiv": (None, [GLuint, GLenum, ctypes.POINTER(GLint)]),
    "glGetShaderInfoLog": (None, [GLuint, GLsizei, ctypes.POINTER(GLsizei), c_char_p]),
    "glDeleteShader": (None, [GLuint]),
    "glCreateProgram": (GLuint, []),
    "glAttachShader": (None, [GLuint, GLuint]),
    "glLinkProgram": (None, [GLuint]),
    "glGetProgramiv": (None, [GLuint, GLenum, ctypes.POINTER(GLint)]),
    "glGetProgramInfoLog": (None, [GLuint, GLsizei, ctypes.POINTER(GLsizei), c_char_p]),
    "glUseProgram": (None, [GLuint]),
    "glDeleteProgram": (None, [GLuint]),
    "glGetUniformLocation": (GLint, [GLuint, c_char_p]),
    "glUniform1i": (None, [GLint, GLint]),
    "glUniform1ui": (None, [GLint, GLuint]),
    "glUniform1f": (None, [GLint, c_float]),
    "glUniform2f": (None, [GLint, c_float, c_float]),
    "glUniform3f": (None, [GLint, c_float, c_float, c_float]),
    "glUniform4f": (None, [GLint, c_float, c_float, c_float, c_float]),
    "glUniformMatrix3fv": (None, [GLint, GLsizei, ctypes.c_ubyte, c_void_p]),
    "glUniformMatrix4fv": (None, [GLint, GLsizei, ctypes.c_ubyte, c_void_p]),
    "glGenBuffers": (None, [GLsizei, ctypes.POINTER(GLuint)]),
    "glDeleteBuffers": (None, [GLsizei, ctypes.POINTER(GLuint)]),
    "glBindBuffer": (None, [GLenum, GLuint]),
    "glBufferData": (None, [GLenum, c_size, c_void_p, GLenum]),
    "glBufferSubData": (None, [GLenum, c_size, c_size, c_void_p]),
    "glBindBufferBase": (None, [GLenum, GLuint, GLuint]),
    "glGenVertexArrays": (None, [GLsizei, ctypes.POINTER(GLuint)]),
    "glDeleteVertexArrays": (None, [GLsizei, ctypes.POINTER(GLuint)]),
    "glBindVertexArray": (None, [GLuint]),
    "glEnableVertexAttribArray": (None, [GLuint]),
    "glVertexAttribPointer": (None, [GLuint, GLint, GLenum, ctypes.c_ubyte, GLsizei, c_void_p]),
    "glDispatchCompute": (None, [GLuint, GLuint, GLuint]),
    "glMemoryBarrier": (None, [GLbitfield]),
    "glDrawArrays": (None, [GLenum, GLint, GLsizei]),
    "glDrawArraysInstanced": (None, [GLenum, GLint, GLsizei, GLsizei]),
    "glGenTextures": (None, [GLsizei, ctypes.POINTER(GLuint)]),
    "glDeleteTextures": (None, [GLsizei, ctypes.POINTER(GLuint)]),
    "glBindTexture": (None, [GLenum, GLuint]),
    "glActiveTexture": (None, [GLenum]),
    "glTexImage2D": (None, [GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, c_void_p]),
    "glTexParameteri": (None, [GLenum, GLenum, GLint]),
    "glGenFramebuffers": (None, [GLsizei, ctypes.POINTER(GLuint)]),
    "glDeleteFramebuffers": (None, [GLsizei, ctypes.POINTER(GLuint)]),
    "glBindFramebuffer": (None, [GLenum, GLuint]),
    "glFramebufferTexture2D": (None, [GLenum, GLenum, GLenum, GLuint, GLint]),
    "glCheckFramebufferStatus": (GLenum, [GLenum]),
    "glBlitFramebuffer": (None, [GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum]),
    "glLineWidth": (None, [c_float]),
}


class GLError(RuntimeError):
    pass


def _load(names) -> Optional[ctypes.CDLL]:
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    return None


class _Loader:
    def __init__(self):
        self._egl = _load(["libEGL.so.1", "libEGL.so", ctypes.util.find_library("EGL") or "libEGL.so.1"])
        self._glx = None
        self._direct = None
        self._fns: Dict[str, object] = {}
        if self._egl is not None:
            self._egl.eglGetProcAddress.restype = c_void_p
            self._egl.eglGetProcAddress.argtypes = [c_char_p]

    def resolve(self, name: str):
        fn = self._fns.get(name)
        if fn is not None:
            return fn
        restype, argtypes = _SIGS[name]
        proto = ctypes.CFUNCTYPE(restype, *argtypes)
        addr = None
        if self._egl is not None:
            addr = self._egl.eglGetProcAddress(name.encode())
        if not addr:
            if self._direct is None:
                self._direct = _load(["libOpenGL.so.0", "libGL.so.1", "libGL.so"]) or False
            if self._direct:
                try:
                    addr = ctypes.cast(getattr(self._direct, name), c_void_p).value
                except AttributeError:
                    addr = None
        if not addr:
            if self._glx is None:
                self._glx = _load(["libGL.so.1"]) or False
                if self._glx:
                    self._glx.glXGetProcAddressARB.restype = c_void_p
                    self._glx.glXGetProcAddressARB.argtypes = [c_char_p]
            if self._glx:
                addr = self._glx.glXGetProcAddressARB(name.encode())
        if not addr:
            raise GLError(f"OpenGL function {name} is not available")
        fn = proto(addr)
        self._fns[name] = fn
        return fn


_loader: Optional[_Loader] = None


def __getattr__(name: str):
    """``gl.glClear(...)`` resolves the entry point on first use."""
    if name.startswith("gl") and name in _SIGS:
        global _loader
        if _loader is None:
            _loader = _Loader()
        return _loader.resolve(name)
    raise AttributeError(name)


def fn(name: str):
    return __getattr__(name)


# --- helpers -------------------------------------------------------------------------------------------------

def get_string(e: int) -> str:
    s = fn("glGetString")(e)
    return s.decode("utf-8", "replace") if s else ""


def get_int(e: int) -> int:
    v = GLint(0)
    fn("glGetIntegerv")(e, ctypes.byref(v))
    return v.value


def version() -> tuple:
    return get_int(MAJOR_VERSION), get_int(MINOR_VERSION)


def check(where: str = "") -> None:
    err = fn("glGetError")()
    if err != NO_ERROR:
        raise GLError(f"GL error 0x{err:04x} {where}")


def compile_shader(kind: int, source: str) -> int:
    sh = fn("glCreateShader")(kind)
    src = c_char_p(source.encode())
    fn("glShaderSource")(sh, 1, ctypes.byref(src), None)
    fn("glCompileShader")(sh)
    ok = GLint(0)
    fn("glGetShaderiv")(sh, COMPILE_STATUS, ctypes.byref(ok))
    if not ok.value:
        n = GLint(0)
        fn("glGetShaderiv")(sh, INFO_LOG_LENGTH, ctypes.byref(n))
        buf = ctypes.create_string_buffer(max(n.value, 1))
        fn("glGetShaderInfoLog")(sh, len(buf), None, buf)
        fn("glDeleteShader")(sh)
        raise GLError("shader compile failed:\n" + buf.value.decode("utf-8", "replace"))
    return sh


def link_program(*shaders: int) -> int:
    prog = fn("glCreateProgram")()
    for s in shaders:
        fn("glAttachShader")(prog, s)
    fn("glLinkProgram")(prog)
    for s in shaders:
        fn("glDeleteShader")(s)
    ok = GLint(0)
    fn("glGetProgramiv")(prog, LINK_STATUS, ctypes.byref(ok))
    if not ok.value:
        n = GLint(0)
        fn("glGetProgramiv")(prog, INFO_LOG_LENGTH, ctypes.byref(n))
        buf = ctypes.create_string_buffer(max(n.value, 1))
        fn("glGetProgramInfoLog")(prog, len(buf), None, buf)
        raise GLError("program link failed:\n" + buf.value.decode("utf-8", "replace"))
    return prog


def gen(name: str, n: int = 1) -> list:
    arr = (GLuint * n)()
    fn(name)(n, arr)
    return list(arr)


def delete(name: str, ids) -> None:
    ids = [i for i in ids if i]
    if ids:
        arr = (GLuint * len(ids))(*ids)
        fn(name)(len(ids), arr)


# --- headless EGL context (tests, offscreen) ----------------------------------------------------------------

class HeadlessContext:
    """A current, surfaceless OpenGL 4.3 core EGL context (``EGL_KHR_surfaceless_context``)."""

    def __init__(self):
        egl = _load(["libEGL.so.1", "libEGL.so"])
        if egl is None:
            raise GLError("libEGL is not available")
        self.egl = egl
        egl.eglGetDisplay.restype = c_void_p
        egl.eglGetDisplay.argtypes = [c_void_p]
        egl.eglInitialize.argtypes = [c_void_p, ctypes.POINTER(GLint), ctypes.POINTER(GLint)]
        egl.eglChooseConfig.argtypes = [c_void_p, ctypes.POINTER(GLint), ctypes.POINTER(c_void_p), GLint,
                                        ctypes.POINTER(GLint)]
        egl.eglBindAPI.argtypes = [c_uint]
        egl.eglCreateContext.restype = c_void_p
        egl.eglCreateContext.argtypes = [c_void_p, c_void_p, c_void_p, ctypes.POINTER(GLint)]
        egl.eglMakeCurrent.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p]
        egl.eglDestroyContext.argtypes = [c_void_p, c_void_p]
        egl.eglTerminate.argtypes = [c_void_p]
        self.dpy = self._display(egl)
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_NONE = 0x3040, 0x0008, 0x3033, 0x0001, 0x3038
        attrs = (GLint * 5)(EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_NONE)
        cfg = c_void_p()
        n = GLint(0)
        if not egl.eglChooseConfig(self.dpy, attrs, ctypes.byref(cfg), 1, ctypes.byref(n)) or n.value < 1:
            raise GLError("no EGL config for desktop OpenGL")
        egl.eglBindAPI(0x30A2)  # EGL_OPENGL_API
        ctx_attrs = (GLint * 7)(0x3098, 4, 0x30FB, 3, 0x30FD, 0x1, EGL_NONE)   # 4.3 core profile
        self.ctx = egl.eglCreateContext(self.dpy, cfg, None, ctx_attrs)
        if not self.ctx:
            raise GLError("eglCreateContext (OpenGL 4.3 core) failed")
        if not egl.eglMakeCurrent(self.dpy, None, None, self.ctx):
            raise GLError("eglMakeCurrent (surfaceless) failed")

    @staticmethod
    def _display(egl):
        """The default display, else Mesa's surfaceless platform, else the first EGL device (NVIDIA headless)."""
        dpy = egl.eglGetDisplay(None)
        if dpy and egl.eglInitialize(dpy, None, None):
            return dpy
        egl.eglGetProcAddress.restype = c_void_p
        egl.eglGetProcAddress.argtypes = [c_char_p]
        get_platform = egl.eglGetProcAddress(b"eglGetPlatformDisplayEXT")
        if get_platform:
            gpd = ctypes.CFUNCTYPE(c_void_p, c_uint, c_void_p, ctypes.POINTER(GLint))(get_platform)
            dpy = gpd(0x31DD, None, None)                     # EGL_PLATFORM_SURFACELESS_MESA
            if dpy and egl.eglInitialize(dpy, None, None):
                return dpy
            qd = egl.eglGetProcAddress(b"eglQueryDevicesEXT")
            if qd:
                devs = (c_void_p * 8)()
                n = GLint(0)
                ctypes.CFUNCTYPE(c_uint, GLint, ctypes.POINTER(c_void_p), ctypes.POINTER(GLint))(qd)(8, devs, ctypes.byref(n))
                for i in range(n.value):
                    dpy = gpd(0x313F, devs[i], None)          # EGL_PLATFORM_DEVICE_EXT
                    if dpy and egl.eglInitialize(dpy, None, None):
                        return dpy
        raise GLError("eglInitialize failed (no default, surfaceless or device display)")

    def close(self) -> None:
        self.egl.eglMakeCurrent(self.dpy, None, None, None)
        self.egl.eglDestroyContext(self.dpy, self.ctx)
        self.egl.eglTerminate(self.dpy)


def headless_context() -> HeadlessContext:
    return HeadlessContext()
