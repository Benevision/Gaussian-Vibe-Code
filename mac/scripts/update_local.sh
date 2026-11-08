#!/bin/bash
# Builds the current source and installs it as mac/build/SplatWork.app — the copy you run locally — but only
# while SplatWork is closed (replacing a running app's files can crash it). If it's open, the new build waits
# in mac/build/.next/ and is swapped in by the next run of this script after you quit the app.
#   mac/scripts/update_local.sh            build + install (or stage)
#   mac/scripts/update_local.sh --open     … and open it afterwards
set -euo pipefail
cd "$(dirname "$0")/.."
NEXT="build/.next"
SPLAT_OUT="$NEXT" SPLAT_BUILD_PATH="${SPLAT_BUILD_PATH:-.build-app}" scripts/build_app.sh
if pgrep -f "build/SplatWork.app/Contents/MacOS/SplatStudio" >/dev/null || pgrep -f "build/Splat Studio.app/Contents/MacOS/SplatStudio" >/dev/null; then
  echo "SplatWork is open: the new build is staged in mac/$NEXT — quit the app and run this script again to install it."
  exit 0
fi
rm -rf "build/SplatWork.app" "build/Splat Studio.app"     # the app was called Splat Studio before 2026-11-08
mv "$NEXT/SplatWork.app" "build/SplatWork.app"
echo "Installed build/SplatWork.app ($(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "build/SplatWork.app/Contents/Info.plist"), build $(/usr/libexec/PlistBuddy -c 'Print CFBundleVersion' "build/SplatWork.app/Contents/Info.plist"))"
[ "${1:-}" = "--open" ] && open "build/SplatWork.app"
exit 0
