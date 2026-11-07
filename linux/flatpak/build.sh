#!/usr/bin/env bash
# Build Splat Worker with flatpak-builder and install it for the current user.
#
#   ./build.sh                 build + install (user installation)
#   ./build.sh --run           ... and start it
#   extra args are passed to flatpak-builder (e.g. --ccache, --keep-build-dirs)
#
# Needs: flatpak and flatpak-builder (or the org.flatpak.Builder Flatpak).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ID=com.example.SplatWorker
MANIFEST="$HERE/$APP_ID.yml"
RUN=0
ARGS=()

# The manifest bundles ../../tools (splatprep), so copy the whole project, not just linux/.
if [[ ! -d "$HERE/../../tools/splatprep" ]]; then
  echo "Missing $(cd "$HERE/../.." && pwd)/tools — copy the whole project folder (linux/ AND tools/) and try again." >&2
  exit 1
fi
# flatpak-builder executes build steps from its state dir; that fails on drives mounted noexec
# (common for USB / external disks under /run/media).
if command -v findmnt >/dev/null 2>&1 && findmnt -no OPTIONS --target "$HERE" | tr ',' '\n' | grep -qx noexec; then
  echo "This folder is on a drive mounted with 'noexec', where builds can't run." >&2
  echo "Copy the project to your home folder first, e.g.:  cp -r \"$(cd "$HERE/../.." && pwd)\" ~/splat-studio" >&2
  exit 1
fi
for a in "$@"; do
  if [[ "$a" == "--run" ]]; then RUN=1; else ARGS+=("$a"); fi
done

if command -v flatpak-builder >/dev/null 2>&1; then
  BUILDER=(flatpak-builder)
elif flatpak info org.flatpak.Builder >/dev/null 2>&1; then
  BUILDER=(flatpak run --filesystem="$HERE/.." org.flatpak.Builder)
else
  echo "flatpak-builder not found. Install it (e.g. 'sudo apt install flatpak-builder') or run:" >&2
  echo "  flatpak install --user flathub org.flatpak.Builder" >&2
  exit 1
fi

flatpak remote-add --user --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo

cd "$HERE"
"${BUILDER[@]}" --user --install --install-deps-from=flathub --force-clean \
  --state-dir="$HERE/.flatpak-builder" --repo="$HERE/repo" \
  ${ARGS[@]+"${ARGS[@]}"} "$HERE/build-dir" "$MANIFEST"

cat <<MSG

Installed. Start it from the app grid ("Splat Worker") or with:
  flatpak run $APP_ID                # desktop app
  flatpak run $APP_ID --headless     # server only (uses the app's settings)
MSG

if [[ "$RUN" == 1 ]]; then
  exec flatpak run "$APP_ID"
fi
