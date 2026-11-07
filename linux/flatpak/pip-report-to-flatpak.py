#!/usr/bin/env python3
"""Convert a ``pip install --dry-run --report`` JSON into a flatpak-builder module.

Produces the same shape as flatpak-pip-generator (one module, every wheel a ``file``
source with its sha256, installed offline with ``pip3 install --no-index``). Used by
generate-deps.sh; see there for how the report is made.

    pip-report-to-flatpak.py report.json requirements-flatpak.txt > python3-deps.json
"""

import json
import re
import sys


def main(report_path: str, requirements_path: str) -> int:
    report = json.load(open(report_path, encoding="utf-8"))
    sources = []
    for item in sorted(report["install"], key=lambda i: i["metadata"]["name"].lower()):
        info = item["download_info"]
        sha = (info.get("archive_info") or {}).get("hashes", {}).get("sha256")
        if not sha:
            sha_field = (info.get("archive_info") or {}).get("hash", "")
            sha = sha_field.split("=", 1)[1] if sha_field.startswith("sha256=") else None
        if not sha or not info["url"].startswith("https://"):
            print(f"error: no https URL + sha256 for {item['metadata']['name']}", file=sys.stderr)
            return 1
        sources.append({"type": "file", "url": info["url"], "sha256": sha})
    top = []
    for line in open(requirements_path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line:
            top.append(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0])
    module = {
        "name": "python3-deps",
        "buildsystem": "simple",
        "build-commands": [
            "pip3 install --verbose --exists-action=i --no-index --find-links=\"file://${PWD}\" "
            "--prefix=${FLATPAK_DEST} --no-build-isolation " + " ".join(top)
        ],
        "sources": sources,
    }
    json.dump(module, sys.stdout, indent=4)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
