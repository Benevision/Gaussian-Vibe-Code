#!/bin/bash
# Packs build/Splat Studio.app into build/SplatStudio-<version>-macOS-arm64.dmg (drag-to-Applications layout).
# Run scripts/build_app.sh first (with SPLAT_VERSION set for a release).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT_DIR="${SPLAT_OUT:-build}"
APP="$OUT_DIR/Splat Studio.app"
[ -d "$APP" ] || { echo "no $APP — run scripts/build_app.sh first" >&2; exit 1; }
VERSION="$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$APP/Contents/Info.plist")"
OUT="$OUT_DIR/SplatStudio-$VERSION-macOS-arm64.dmg"
STAGE="$(mktemp -d)/Splat Studio"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "$OUT"
hdiutil create -volname "SplatWork $VERSION" -srcfolder "$STAGE" -fs HFS+ -format UDZO -imagekey zlib-level=9 "$OUT" >/dev/null
rm -rf "$(dirname "$STAGE")"
echo "$OUT ($(du -h "$OUT" | cut -f1))"
