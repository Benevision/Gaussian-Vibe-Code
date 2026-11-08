#!/usr/bin/env python3
"""Regenerate splatworker/studio/catalog.json from the Mac app's SettingsCatalog.swift (single source of truth
for every setting's documentation). A test checks the two stay in sync.

    python3 linux/scripts/gen_catalog.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWIFT = ROOT / "mac/Sources/SplatStudio/Core/SettingsCatalog.swift"
OUT = ROOT / "linux/splatworker/studio/catalog.json"

_STR = r'"((?:[^"\\]|\\.)*)"'


def _unescape(s: str) -> str:
    return s.encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8") if "\\" in s else s


def parse(text: str) -> dict:
    groups = {m.group(1): _unescape(m.group(2)) for m in re.finditer(r"case (\w+) = " + _STR, text)}
    order = list(groups)
    items = []
    for block in re.split(r"\.init\(", text)[1:]:
        m = re.match(r'\s*id:\s*' + _STR, block)
        if not m:
            continue
        body = block.split("\n        .init(")[0]
        d = {"id": m.group(1)}
        g = re.search(r"group:\s*\.(\w+)", body)
        d["group"] = groups.get(g.group(1), g.group(1)) if g else ""
        for key in ("title", "summary", "detail", "defaultValue", "range", "effects"):
            mm = re.search(r"\b" + key + r":\s*" + _STR, body)
            d[key] = _unescape(mm.group(1)) if mm else ""
        a = re.search(r"advanced:\s*(true|false)", body)
        d["advanced"] = (a.group(1) == "true") if a else True
        items.append(d)
    return {"groups": [groups[k] for k in order], "settings": items}


def main() -> int:
    data = parse(SWIFT.read_text(encoding="utf-8"))
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(data['settings'])} settings → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
