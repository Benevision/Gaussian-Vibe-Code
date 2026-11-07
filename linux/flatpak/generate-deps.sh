#!/usr/bin/env bash
# Regenerate python3-deps.json (the Flatpak's bundled lightweight Python packages) from
# requirements-flatpak.txt.
#
#   ./generate-deps.sh                  # default: pip's resolver, wheels for the runtime's Python
#   ./generate-deps.sh --pip-generator  # use flatpak-pip-generator inside the GNOME SDK instead
#
# Both pin every file by URL + sha256. Keep RUNTIME_PY in sync with the manifest's
# runtime-version (org.gnome.Platform 51 = freedesktop-sdk 26.08 = Python 3.14).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQS="$HERE/requirements-flatpak.txt"
OUT="$HERE/python3-deps.json"
RUNTIME_PY="${RUNTIME_PY:-3.14}"
SDK="${SDK:-org.gnome.Sdk//51}"
ABI="cp${RUNTIME_PY/./}"

if [[ "${1:-}" == "--pip-generator" ]]; then
  # https://github.com/flatpak/flatpak-builder-tools/tree/master/pip
  GEN="$HERE/.flatpak-pip-generator"
  if [[ ! -f "$GEN" ]]; then
    curl -fsSL -o "$GEN" \
      https://raw.githubusercontent.com/flatpak/flatpak-builder-tools/master/pip/flatpak-pip-generator.py
  fi
  # --runtime makes the generator resolve inside the SDK, so wheels match its Python.
  # Note: packages it can only get as sdists (e.g. pydantic-core needs Rust) will not
  # build in the Flatpak; the default mode below only ever picks binary wheels.
  python3 "$GEN" --runtime="$SDK" --requirements-file="$REQS" --output="${OUT%.json}"
  echo "wrote $OUT"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
python3 -m venv "$TMP/venv"
"$TMP/venv/bin/pip" install -q --upgrade "pip>=23"
"$TMP/venv/bin/pip" install -q --dry-run --ignore-installed --report "$TMP/report.json" \
  --only-binary=:all: --implementation cp --python-version "$RUNTIME_PY" --abi "$ABI" \
  --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
  --target "$TMP/target" -r "$REQS"
python3 "$HERE/pip-report-to-flatpak.py" "$TMP/report.json" "$REQS" > "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
echo "wrote $OUT ($(grep -c '"url"' "$OUT") files)"
