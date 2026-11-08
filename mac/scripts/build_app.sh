#!/bin/bash
# Builds Splat Studio.app (release, arm64) into mac/build/.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-release}"
# SPLAT_VERSION: marketing version (e.g. 1.4.0, set by the release workflow from the tag); build number = commit count.
LAST_TAG="$(git describe --tags --abbrev=0 --match 'v*' 2>/dev/null || echo v1.0.0)"
VERSION="${SPLAT_VERSION:-${LAST_TAG#v}-dev}"   # local builds: last release + "-dev"
BUILD="$(git rev-list --count HEAD 2>/dev/null || echo 1)"
# SPLAT_BUILD_PATH: build somewhere else, e.g. while a command-line run uses the default .build binary
# (relinking a binary that is running can crash it).
BP=(); [ -n "${SPLAT_BUILD_PATH:-}" ] && BP=(--build-path "$SPLAT_BUILD_PATH")
swift build -c "$CONFIG" --arch arm64 ${BP[@]+"${BP[@]}"}
BIN="$(swift build -c "$CONFIG" --arch arm64 ${BP[@]+"${BP[@]}"} --show-bin-path)/SplatStudio"
# SPLAT_OUT: output folder (default build/) — use another one while the app in build/ is running.
OUT_DIR="${SPLAT_OUT:-build}"
APP="$OUT_DIR/Splat Studio.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/SplatStudio"
# Bundle the Python processing toolkit (optional Models pack installs from here).
if [ -d ../tools/splatprep ]; then
  rsync -a --exclude '__pycache__' --exclude '.pytest_cache' --exclude 'tests' ../tools/ "$APP/Contents/Resources/tools/"
fi
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Splat Studio</string>
  <key>CFBundleDisplayName</key><string>Splat Studio</string>
  <key>CFBundleIdentifier</key><string>com.example.splatstudio</string>
  <key>CFBundleExecutable</key><string>SplatStudio</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$BUILD</string>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
  <key>NSLocalNetworkUsageDescription</key><string>Splat Studio receives captures from your iPhone and talks to Splat Worker GPU servers on your local network.</string>
  <key>NSBonjourServices</key><array><string>_splatcapture._tcp</string><string>_splatworker._tcp</string></array>
  <key>CFBundleDocumentTypes</key><array>
    <dict><key>CFBundleTypeName</key><string>Gaussian Splat</string><key>CFBundleTypeRole</key><string>Editor</string>
      <key>CFBundleTypeExtensions</key><array><string>ply</string><string>spz</string><string>splat</string></array></dict>
    <dict><key>CFBundleTypeName</key><string>Splat Capture</string><key>CFBundleTypeRole</key><string>Viewer</string>
      <key>CFBundleTypeExtensions</key><array><string>splatcapture</string></array><key>LSTypeIsPackage</key><true/></dict>
  </array>
</dict></plist>
PLIST
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || echo "warning: ad-hoc codesign failed"
echo "Built $APP ($VERSION, build $BUILD)"
