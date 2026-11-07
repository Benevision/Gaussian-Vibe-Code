import io
import stat
import zipfile

import pytest

from splatworker.safezip import UnsafeArchive, safe_extract


def _zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name)
            if mode is not None:
                info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return buf.getvalue()


def _extract(tmp_path, blob, **kw):
    p = tmp_path / "in.zip"
    p.write_bytes(blob)
    opts = dict(max_bytes=10 ** 7, max_files=100)
    opts.update(kw)
    return safe_extract(p, tmp_path / "out", **opts)


def test_ok_and_skips_macosx(tmp_path):
    blob = _zip([("a/b.txt", b"hi", None), ("__MACOSX/a/._b.txt", b"junk", None), ("a/.DS_Store", b"x", None)])
    assert _extract(tmp_path, blob) == 2
    assert (tmp_path / "out/a/b.txt").read_bytes() == b"hi"
    assert not (tmp_path / "out/__MACOSX").exists()
    assert not (tmp_path / "out/a/.DS_Store").exists()


@pytest.mark.parametrize("name", ["../evil.txt", "a/../../evil.txt", "/etc/evil", "C:/evil", "a\\..\\evil"])
def test_rejects_traversal(tmp_path, name):
    with pytest.raises(UnsafeArchive):
        _extract(tmp_path, _zip([(name, b"x", None)]))
    assert not (tmp_path / "evil.txt").exists()


def test_rejects_symlink(tmp_path):
    blob = _zip([("link", b"/etc/passwd", stat.S_IFLNK | 0o777)])
    with pytest.raises(UnsafeArchive, match="symlink"):
        _extract(tmp_path, blob)


def test_rejects_too_large_and_too_many(tmp_path):
    with pytest.raises(UnsafeArchive):
        _extract(tmp_path, _zip([("big", b"0" * 5000, None)]), max_bytes=1000)
    with pytest.raises(UnsafeArchive):
        _extract(tmp_path, _zip([(f"f{i}", b"x", None) for i in range(5)]), max_files=3)


def test_rejects_zip_bomb_ratio(tmp_path):
    with pytest.raises(UnsafeArchive, match="ratio"):
        _extract(tmp_path, _zip([("bomb", b"\0" * (8 * 1024 * 1024), None)]), max_bytes=10 ** 9, max_ratio=10)


def test_rejects_garbage(tmp_path):
    with pytest.raises(UnsafeArchive):
        _extract(tmp_path, b"this is not a zip")
