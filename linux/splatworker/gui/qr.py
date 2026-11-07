"""Optional QR code of the connection details (needs the pure-Python ``qrcode`` package)."""

from __future__ import annotations

from typing import List, Optional, Tuple


def qr_matrix(data: str) -> Optional[List[List[bool]]]:
    try:
        import qrcode  # type: ignore
    except ImportError:
        return None
    qr = qrcode.QRCode(border=0, error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(data)
    qr.make(fit=True)
    return [[bool(c) for c in row] for row in qr.get_matrix()]


def render_rgb(matrix: List[List[bool]], scale: int = 8, border: int = 4) -> Tuple[bytes, int, int]:
    """Matrix -> (RGB bytes, width, height) with a white quiet zone; for Gdk.MemoryTexture."""
    n = len(matrix)
    size = (n + 2 * border) * scale
    white, black = b"\xff\xff\xff", b"\x00\x00\x00"
    rows = []
    blank = white * size
    for _ in range(border * scale):
        rows.append(blank)
    for r in matrix:
        line = white * (border * scale) + b"".join((black if c else white) * scale for c in r) + \
            white * (border * scale)
        rows.extend([line] * scale)
    for _ in range(border * scale):
        rows.append(blank)
    return b"".join(rows), size, size
