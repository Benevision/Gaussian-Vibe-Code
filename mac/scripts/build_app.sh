#!/bin/bash
# Builds Splat Studio.app (release, arm64) into mac/build/.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-release}"
swift build -c "$CONFIG" --arch arm64
BIN="$(swift build -c "$CONFIG" --arch arm64 --show-bin-path)/SplatStudio"
APP="build/Splat Studio.app"
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
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
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
echo "Built $APP"
