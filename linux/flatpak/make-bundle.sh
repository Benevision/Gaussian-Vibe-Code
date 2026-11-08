#!/usr/bin/env bash
# Build a single-file Flatpak bundle for sharing:
#
#   ./make-bundle.sh                    -> dist/SplatWorkNode-<version>-x86_64.flatpak
#
# Install it on another machine with:
#   flatpak install --user SplatWorkNode-<version>-x86_64.flatpak
# The GNOME runtime is fetched from Flathub automatically (--runtime-repo below).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ID=com.example.SplatWorker
VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$HERE/../splatworker/__init__.py")"
ARCH="$(flatpak --default-arch)"
OUT_DIR="$HERE/dist"
OUT="$OUT_DIR/SplatWorkNode-$VERSION-$ARCH.flatpak"

if command -v flatpak-builder >/dev/null 2>&1; then
  BUILDER=(flatpak-builder)
elif flatpak info org.flatpak.Builder >/dev/null 2>&1; then
  BUILDER=(flatpak run --filesystem="$HERE/.." org.flatpak.Builder)
else
  echo "flatpak-builder not found (see build.sh)" >&2
  exit 1
fi

flatpak remote-add --user --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo
cd "$HERE"
"${BUILDER[@]}" --user --install-deps-from=flathub --force-clean \
  --state-dir="$HERE/.flatpak-builder" --repo="$HERE/repo" --default-branch=stable \
  "$HERE/build-dir" "$HERE/$APP_ID.yml"

mkdir -p "$OUT_DIR"
# SPLAT_REPO_URL (set by the release workflow): the GitHub Pages Flatpak repo becomes the bundle's update
# origin, so installs from the .flatpak file get updates (and the app's self-update) like remote installs.
REPO_URL_ARGS=()
[ -n "${SPLAT_REPO_URL:-}" ] && REPO_URL_ARGS=(--repo-url="$SPLAT_REPO_URL")
flatpak build-bundle --runtime-repo=https://dl.flathub.org/repo/flathub.flatpakrepo ${REPO_URL_ARGS[@]+"${REPO_URL_ARGS[@]}"} \
  "$HERE/repo" "$OUT" "$APP_ID" stable
echo "bundle: $OUT ($(du -h "$OUT" | cut -f1))"
echo "install: flatpak install --user $(basename "$OUT")"
