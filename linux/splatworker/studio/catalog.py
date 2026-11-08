"""The settings reference (``catalog.json``, generated from the Mac's SettingsCatalog.swift by
``linux/scripts/gen_catalog.py``)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

PATH = Path(__file__).with_name("catalog.json")


@lru_cache(maxsize=1)
def load() -> Dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def groups() -> List[str]:
    return load()["groups"]


def settings() -> List[Dict]:
    return load()["settings"]


def doc(setting_id: str) -> Optional[Dict]:
    return next((s for s in settings() if s["id"] == setting_id), None)


def search(text: str) -> List[Dict]:
    q = text.strip().lower()
    if not q:
        return settings()
    return [s for s in settings()
            if q in " ".join((s["id"], s["title"], s["summary"], s["detail"], s["group"])).lower()]


def summary(setting_id: str) -> str:
    d = doc(setting_id)
    return d["summary"] if d else ""


def tooltip(setting_id: str) -> str:
    d = doc(setting_id)
    if not d:
        return ""
    parts = [d["summary"], "", d["detail"]]
    if d.get("defaultValue"):
        parts.append(f"\nDefault: {d['defaultValue']}" + (f"   Range: {d['range']}" if d.get("range") else ""))
    if d.get("effects"):
        parts.append(f"Effect: {d['effects']}")
    return "\n".join(parts)
