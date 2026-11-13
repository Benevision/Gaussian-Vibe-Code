# SplatWork branding

The hex mesh mark: six linked nodes (your devices) around a splat. All files are generated from the same geometry.

| File | Use |
|---|---|
| `logo-mark.svg` | The mark on a transparent background: README, website (`website/logo.svg`), documents |
| `app-icon-macos.svg` | Mac app icon (824 px rounded square on the 1024 px grid) → `mac/Resources/AppIcon.icns` |
| `app-icon-ios.svg` | iPhone app icon, full-bleed (iOS rounds the corners) → `ios/.../AppIcon.appiconset/AppIcon.png` |
| `app-icon-linux.svg` | SplatWork for Linux icon → `linux/flatpak/icons/com.example.SplatWorker.svg` |
| `logo-symbolic.svg` | One-colour icon for small sizes → `linux/flatpak/icons/com.example.SplatWorker-symbolic.svg` |

The Mac app's start page draws the same mark in code (`mac/Sources/SplatStudio/UI/HexMeshLogo.swift`).

Colours: nodes and links `#3b82f6` / `#60a5fa`, splat `#a855f7` and `#3b82f6`, background `#18213f` → `#0a0f1f`.

Regenerating the Mac icon after editing the SVG:

```bash
mkdir -p AppIcon.iconset
for s in 16 32 128 256 512; do
  rsvg-convert -w $s app-icon-macos.svg -o AppIcon.iconset/icon_${s}x${s}.png
  rsvg-convert -w $((s*2)) app-icon-macos.svg -o AppIcon.iconset/icon_${s}x${s}@2x.png
done
iconutil -c icns AppIcon.iconset -o ../mac/Resources/AppIcon.icns
```
