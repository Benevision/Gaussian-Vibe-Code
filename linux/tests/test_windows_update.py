"""SplatWork for Windows updates: release parsing, checksum verification, version from bundle.json."""

import hashlib
import io
import json

from splatworker.gui import windows_update as W

RELEASE = {"tag_name": "v1.3.0", "html_url": "https://github.com/Benevision/SplatWork/releases/tag/v1.3.0",
           "assets": [{"name": "SplatWork-1.3.0.dmg", "browser_download_url": "https://x/dmg"},
                      {"name": "SplatWork-1.3.0-Windows-x64.msi", "browser_download_url": "https://x/msi"},
                      {"name": "SplatWork-1.3.0-Windows-x64.zip", "browser_download_url": "https://x/zip"},
                      {"name": "SHA256SUMS.txt", "browser_download_url": "https://x/sums"}]}


def test_parse_release():
    info = W.parse_release(RELEASE, "1.2.9")
    assert info["version"] == "1.3.0" and info["msi_url"] == "https://x/msi" and info["sums_url"] == "https://x/sums"
    assert W.parse_release(RELEASE, "1.3.0") is None                       # not newer
    assert W.parse_release({**RELEASE, "prerelease": True}, "1.0.0") is None
    assert W.parse_release({**RELEASE, "assets": RELEASE["assets"][:1]}, "1.0.0") is None   # no MSI (Mac-only)


class Resp(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_verifies_checksum(tmp_path):
    msi = b"MSI" * 1000
    good = f"{hashlib.sha256(msi).hexdigest()}  SplatWork-1.3.0-Windows-x64.msi\n"
    info = W.parse_release(RELEASE, "1.0.0")
    files = {"https://x/sums": good.encode(), "https://x/msi": msi}
    got = []
    out = W.download(info, tmp_path, got.append, opener=lambda req, timeout=None: Resp(files[req.full_url]))
    assert out.read_bytes() == msi and got[-1] == 1.0
    files["https://x/sums"] = good.replace(good[:8], "00000000").encode()
    try:
        W.download(info, tmp_path / "b", opener=lambda req, timeout=None: Resp(files[req.full_url]))
        raise AssertionError("bad checksum accepted")
    except RuntimeError as e:
        assert "checksum" in str(e)
    assert not list((tmp_path / "b").glob("*.msi"))


def test_check_and_installed_version(tmp_path, monkeypatch):
    (tmp_path / "bundle.json").write_text(json.dumps({"version": "1.2.0"}))
    env = {"SPLATWORK_BUNDLE": str(tmp_path)}
    assert W.installed_version(env) == "1.2.0" and W.installed_version({}) is None
    monkeypatch.delenv("SPLATWORKER_UPDATE_URL", raising=False)
    info = W.check("1.2.0", opener=lambda req, timeout=None: Resp(json.dumps(RELEASE).encode()))
    assert info["version"] == "1.3.0"
    assert W.check("1.2.0", opener=lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))) is None
    assert W.install_argv(tmp_path / "a.msi")[:2] == ["msiexec", "/i"]
