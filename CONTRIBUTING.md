# Contributing to SplatWork

Thanks for your interest! Bug reports, ideas, documentation fixes and code are all welcome.

## Reporting bugs and asking for features

[Open an issue](https://github.com/Benevision/SplatWork/issues/new/choose). For a bug, say which app and version
(SplatWork → About, or SplatWork Node's Preferences), what you did, what you expected and what happened. A project's
`prep.log`, the training dashboard and a screenshot help a lot. Security problems go through
[SECURITY.md](SECURITY.md), not public issues.

## Repository layout

| Folder | Contents | Language |
|---|---|---|
| `mac/` | SplatWork for Mac: app, Metal renderer and trainer, pipeline, worker client (`Sources/SplatStudio`) | Swift 6, SwiftUI, Metal |
| `ios/` | SplatWork Capture: capture app plus the on-device trainer (`SplatTrainKit`) | Swift, ARKit |
| `linux/` | SplatWork Node: worker server (`splatworker/`), GTK studio (`splatworker/gui/`), Flatpak (`flatpak/`) | Python, FastAPI, GTK 4 |
| `tools/` | splatprep processing toolkit and evaluation scripts | Python, PyTorch |
| `docs/` | User guide, settings reference, architecture, protocols, developer notes | Markdown |
| `steamframe/` | Experimental headset capture prototype | Python |

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how the parts fit together, and
[docs/PROTOCOL.md](docs/PROTOCOL.md) defines the formats and the HTTP API they share.

## Building and testing

**Mac app** (Apple silicon, Xcode 16+):

```bash
cd mac
swift build -c release            # build the app and its command line
./scripts/build_app.sh            # make build/SplatWork.app
./scripts/update_local.sh         # build and install it, waiting until the app is closed
```

**iPhone app** (Xcode 26+, XcodeGen): `cd ios && xcodegen generate`, then build the *SplatCapture* scheme. Run
`xcodegen generate` again whenever you add or remove files; `project.yml` is the source of truth.

**SplatWork Node and splatprep** (Python 3.10+):

```bash
cd linux && python -m pytest -q    # worker, studio core and GUI logic (no GPU needed)
cd tools && python -m pytest -q    # splatprep
cd linux/flatpak && bash build.sh  # build and install the Flatpak locally
```

The GTK viewer can be tested without a display; see [docs/dev/LINUX_STUDIO.md](docs/dev/LINUX_STUDIO.md).

**Settings.** Every user-facing setting is described once, in `mac/Sources/SplatStudio/Core/SettingsCatalog.swift`.
After changing it, regenerate the reference and the Linux catalog:

```bash
mac/.build/release/SplatStudio --dump-settings > docs/SETTINGS.md
python linux/scripts/gen_catalog.py
```

## Sending changes

1. Fork the repository and create a branch from `main`.
2. Keep each pull request to one change, with tests where it makes sense.
3. Run the tests above for the parts you touched, and say in the pull request what you checked by hand.
4. Write user-facing text plainly, and use the product names SplatWork, SplatWork Capture and SplatWork Node.

Every push to `main` builds the Mac developer build and the Linux Flatpak repository, so `main` should always be
releasable.

## Licensing

SplatWork is licensed under the [Apache License 2.0](LICENSE). By contributing, you agree that your contributions
are licensed under the same terms. Don't copy code from projects with incompatible licences (for example GPL code);
reimplement ideas from papers and other projects in your own code and credit them in comments.
