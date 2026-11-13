#!/usr/bin/env bash
# Builds SplatWork.exe (GUI) and splatwork-cli.exe (console) in the MSYS2 UCRT64 shell.
#   windows/launcher/build.sh 1.2.1 windows/build/out
set -euo pipefail
version="${1:-0.0.0}"; out="${2:-windows/build/out}"
here="$(cd "$(dirname "$0")" && pwd)"
IFS=. read -r major minor patch <<<"${version%%-*}"
mkdir -p "$out"
py="$(python -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
cflags="$(python3-config --includes 2>/dev/null || python -c 'import sysconfig; print("-I" + sysconfig.get_paths()["include"])')"
windres -DSPLATWORK_VERSION_MAJOR="$major" -DSPLATWORK_VERSION_MINOR="$minor" -DSPLATWORK_VERSION_PATCH="${patch:-0}" \
  -I "$here" "$here/splatwork.rc" -O coff -o "$out/splatwork.res"
gcc -O2 -municode -mwindows $cflags "$here/splatwork.c" "$out/splatwork.res" -l"$py" -lshell32 -lole32 -o "$out/SplatWork.exe"
gcc -O2 -municode -DSPLATWORK_CONSOLE $cflags "$here/splatwork.c" "$out/splatwork.res" -l"$py" -lshell32 -lole32 -o "$out/splatwork-cli.exe"
rm -f "$out/splatwork.res"
ls -la "$out"
