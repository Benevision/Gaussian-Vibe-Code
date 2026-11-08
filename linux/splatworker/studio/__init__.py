"""SplatWork for Linux: the GTK-free core of the desktop studio.

Everything in this package is plain Python + numpy (no GTK, no torch), so it runs in the Flatpak's own
Python (GNOME runtime) next to the GUI and can be unit tested anywhere:

* :mod:`.splats`    ``GaussianCloud`` (structure of arrays, INRIA 3DGS conventions) + learned skybox sidecar
* :mod:`.formats`   PLY, PlayCanvas compressed PLY, antimatter15 ``.splat``, Niantic ``.spz`` (read + write)
* :mod:`.camera`    orbit / fly navigation, OpenCV-convention view + projection matrices, scene orientation
* :mod:`.sorting`   depth-sort keys + a background sorter for the viewer
* :mod:`.edit`      crop box, thresholds, brush selection, delete / undo
* :mod:`.autoclean` Auto clean plan from a contribution census (same rules as the Mac)
* :mod:`.presets`   training presets and the New Project options → splatprep / TrainingConfig requests
* :mod:`.projects`  project folders (``project.json``), recent files
* :mod:`.pipeline`  New Project → prepare → train on this PC, as jobs of the local worker server
* :mod:`.catalog`   the settings reference (generated from the Mac app's SettingsCatalog.swift)

GPU work (splatprep, the gsplat trainer, the Auto clean census) never runs in the GUI process: it is
submitted as ordinary jobs to the worker server this app already runs, so local and remote (Mac) jobs share
one queue per GPU.
"""
