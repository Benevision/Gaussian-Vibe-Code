# Splat Studio — settings reference

Generated from `SettingsCatalog.swift` (`SplatStudio --dump-settings`). 103 settings. **Easy** settings are always visible; **Advanced** ones appear when the toolbar is set to Advanced. The same text is in the app under **Help → Settings Reference** and behind every ⓘ button.

## Contents

- [Viewer](#viewer) — 20 settings
- [Editing](#editing) — 5 settings
- [Sources & frame extraction](#sources--frame-extraction) — 13 settings
- [Camera solving (SfM)](#camera-solving-sfm) — 10 settings
- [Masking](#masking) — 2 settings
- [Training — basics](#training--basics) — 9 settings
- [Training — densification](#training--densification) — 12 settings
- [Training — learning rates](#training--learning-rates) — 5 settings
- [Training — losses & regularisation](#training--losses--regularisation) — 15 settings
- [Compute & networking](#compute--networking) — 7 settings
- [Export](#export) — 5 settings

## Viewer

| Setting | Default | Range | Mode |
|---|---|---|---|
| [Display mode](#display-mode) | Colour | – | Easy |
| [View-dependent colour (SH degree)](#view-dependent-colour-sh-degree) | 3 | 0–3 | Easy |
| [Splat scale](#splat-scale) | 1.0 | 0.1–3.0 | Advanced |
| [Background colour](#background-colour) | Near-black | – | Easy |
| [Fast 16-bit depth sort](#fast-16-bit-depth-sort) | On | – | Advanced |
| [Render resolution](#render-resolution) | 1.0 | 0.25–1.0 | Advanced |
| [Minimum opacity](#minimum-opacity) | 1/255 (0.004) | 0–0.5 | Advanced |
| [Skip work when idle](#skip-work-when-idle) | On | – | Advanced |
| [Show training cameras](#show-training-cameras) | On | – | Easy |
| [Navigation mode](#navigation-mode) | Orbit | – | Easy |
| [Rotation speed](#rotation-speed) | 1.0 | 0.2–3 | Easy |
| [Pan speed](#pan-speed) | 1.0 | 0.2–3 | Advanced |
| [Zoom speed](#zoom-speed) | 1.0 | 0.2–3 | Easy |
| [Fly speed](#fly-speed) | 1.0 | 0.1–5 | Advanced |
| [Two-finger drag](#two-finger-drag) | Orbit | – | Easy |
| [Invert vertical rotation](#invert-vertical-rotation) | Off | – | Advanced |
| [Smooth camera](#smooth-camera) | On | – | Advanced |
| [Scene up direction](#scene-up-direction) | −Y (or as saved with the file) | – | Easy |
| [Field of view](#field-of-view) | 50° | 20–110° | Advanced |
| [Anti-aliasing filter](#anti-aliasing-filter) | Automatic | – | Advanced |

### Display mode

> What each splat shows: colour, depth, centres, ellipse outlines or opacity.

| | |
|---|---|
| **Default** | Colour |
| **Effect** | No performance impact. |
| **Mode** | Easy |
| **ID** | `viewer.displayMode` |

**How it works.** Colour is the normal view. Depth colours every splat by its distance from the camera (warm = near), which is the fastest way to spot floaters — blobs hanging in mid-air show up as colour discontinuities.

- Centres draws every gaussian as a small dot so you can judge density and coverage.
- Ellipses draws each gaussian's 2σ outline so you can see size and orientation; long needle-like splats indicate under-constrained regions.
- Opacity colours by opacity (blue = transparent, red = opaque) and helps tune the 'Delete below opacity' tool.

### View-dependent colour (SH degree)

> How many spherical-harmonic bands are used for reflections and sheen.

| | |
|---|---|
| **Default** | 3 |
| **Range** | 0–3 |
| **Effect** | Higher = more realistic reflections, slightly more GPU work per splat. |
| **Mode** | Easy |
| **ID** | `viewer.maxSHDegree` |

**How it works.** Gaussian splats store colour as spherical harmonics so a surface can look different from different angles (specular highlights, reflections).

- Degree 0 is a flat colour per splat, 1–3 add progressively finer view-dependence.

**When to change it**

- Lowering it here only affects display: it is a good way to check how much of the look comes from view-dependent effects, and degree 0–1 renders noticeably faster on very large scenes.

### Splat scale

> Multiplies every splat's size (display only).

| | |
|---|---|
| **Default** | 1.0 |
| **Range** | 0.1–3.0 |
| **Mode** | Advanced |
| **ID** | `viewer.splatScale` |

**How it works.** Values below 1 shrink splats and reveal the structure underneath; above 1 fills holes at the cost of blur.

- Purely a visualisation aid — exports are unaffected.

### Background colour

> Colour behind the scene.

| | |
|---|---|
| **Default** | Near-black |
| **Mode** | Easy |
| **ID** | `viewer.background` |

**How it works.** Use a contrasting background to spot semi-transparent haze around objects (try white for dark scenes).

- Display only.

### Fast 16-bit depth sort

> Sorts splats with 16-bit logarithmic depth keys (2 passes instead of 4).

| | |
|---|---|
| **Default** | On |
| **Effect** | Off = exact ordering, ~2× more sort time. |
| **Mode** | Advanced |
| **ID** | `viewer.sortPrecision` |

**How it works.** Splats are blended back-to-front, so they must be depth-sorted every frame.

- 16-bit log-depth keys are ~2× faster to sort and, thanks to the logarithmic mapping, accurate to ~0.02 % of the distance — visually identical in almost all scenes.

**When to change it**

- Switch it off if you see shimmering between overlapping surfaces at very different distances (e.g. a close object in front of a huge landscape).

### Render resolution

> Renders at a fraction of the window's pixel resolution, then upscales.

| | |
|---|---|
| **Default** | 1.0 |
| **Range** | 0.25–1.0 |
| **Effect** | Lower = faster, softer image. |
| **Mode** | Advanced |
| **ID** | `viewer.renderScale` |

**How it works.** On Retina displays the viewer normally renders at full backing resolution.

- 0.5 renders a quarter of the pixels, which keeps navigation fluid with 5–10 M splats.
- Screenshots and videos always render at their own requested resolution.

### Minimum opacity

> Splats more transparent than this are skipped.

| | |
|---|---|
| **Default** | 1/255 (0.004) |
| **Range** | 0–0.5 |
| **Mode** | Advanced |
| **ID** | `viewer.alphaCutoff` |

**How it works.** Raising it hides faint haze and floaters and speeds up rendering; it is a non-destructive preview of 'Delete below opacity'.

### Skip work when idle

> Re-uses the last frame's projection and sort while the camera is still.

| | |
|---|---|
| **Default** | On |
| **Mode** | Advanced |
| **ID** | `viewer.lazySort` |

**How it works.** Saves power and heat when you are not moving the camera.

**When to change it**

- Disable only when profiling.

### Show training cameras

> Draws the solved camera positions as frustums.

| | |
|---|---|
| **Default** | On |
| **Mode** | Easy |
| **ID** | `viewer.showCameras` |

**How it works.** Lets you check the camera solve: frustums should follow the path you walked.

- Double-click a training camera in the Cameras list to look through it.

### Navigation mode

> Orbit around a point, or fly freely with W A S D.

| | |
|---|---|
| **Default** | Orbit |
| **Mode** | Easy |
| **ID** | `viewer.cameraMode` |

**How it works.** Orbit: drag to turn the scene (like grabbing it), ⇧-drag or right-drag to pan, scroll or pinch to zoom, double-click a point to orbit around it.

- Fly: W/S forward/back, A/D strafe, Q/E down/up, drag to look, ⇧ faster, ⌥ slower.
- R resets the view, F frames everything, 1–5 switch display modes, ? shows all shortcuts.

### Rotation speed

> How fast dragging turns the scene.

| | |
|---|---|
| **Default** | 1.0 |
| **Range** | 0.2–3 |
| **Mode** | Easy |
| **ID** | `nav.rotateSpeed` |

**How it works.** 1.0 turns the scene about a quarter of a degree per point dragged — a drag across a laptop-sized viewer turns it roughly 270°.

- Dragging works like grabbing the scene: drag right and it turns right, drag down and its top tilts towards you.

### Pan speed

> How fast ⇧-drag, right-drag (or two-finger drag in Pan mode) slides the view.

| | |
|---|---|
| **Default** | 1.0 |
| **Range** | 0.2–3 |
| **Mode** | Advanced |
| **ID** | `nav.panSpeed` |

**How it works.** At 1.0 the point under the cursor stays under the cursor while you pan.

### Zoom speed

> How fast the scroll wheel and pinch zoom.

| | |
|---|---|
| **Default** | 1.0 |
| **Range** | 0.2–3 |
| **Mode** | Easy |
| **ID** | `nav.zoomSpeed` |

**How it works.** At 1.0 one mouse-wheel notch zooms about 8 %.

- Trackpad: pinch, or ⌘ + two-finger drag.

### Fly speed

> Walking speed in Fly mode (W A S D Q E).

| | |
|---|---|
| **Default** | 1.0 |
| **Range** | 0.1–5 |
| **Mode** | Advanced |
| **ID** | `nav.flySpeed` |

**How it works.** At 1.0 you travel a quarter of the scene's size per second.

**When to change it**

- Hold ⇧ for 4× faster, ⌥ for 4× slower.

### Two-finger drag

> What a two-finger trackpad drag does: orbit or pan.

| | |
|---|---|
| **Default** | Orbit |
| **Mode** | Easy |
| **ID** | `nav.trackpad` |

**How it works.** Orbit follows your fingers like grabbing the scene; hold ⇧ to temporarily switch to the other mode.

- Pinch always zooms, two-finger rotate turns the scene.
- The direction follows your fingers regardless of the system 'natural scrolling' setting.

### Invert vertical rotation

> Reverse the up/down direction of orbiting.

| | |
|---|---|
| **Default** | Off |
| **Mode** | Advanced |
| **ID** | `nav.invertVertical` |

**How it works.** For people used to flight-stick style controls.

### Smooth camera

> Eases camera movement instead of jumping exactly with each input event.

| | |
|---|---|
| **Default** | On |
| **Mode** | Advanced |
| **ID** | `nav.smoothing` |

**How it works.** Makes navigation feel calmer, especially with trackpads and mouse wheels that send coarse steps.

**When to change it**

- Turn off for the most direct response.

### Scene up direction

> Which axis of the scene points up. Fix upside-down or sideways splats here.

| | |
|---|---|
| **Default** | −Y (or as saved with the file) |
| **Mode** | Easy |
| **ID** | `viewer.orientation` |

**How it works.** Splat files don't store which way is up, and tools disagree (COLMAP/3DGS use −Y, OpenGL/ARKit +Y, Blender and drones +Z).

- Splat Studio levels every dataset it builds to −Y, but files from elsewhere may need help: pick an axis, Flip upside down, or rotate until it looks level and choose Level to view.
- Your choice is remembered for that file, and Export can bake it in so other viewers show it upright too.

### Field of view

> Vertical field of view of the viewer camera.

| | |
|---|---|
| **Default** | 50° |
| **Range** | 20–110° |
| **Mode** | Advanced |
| **ID** | `viewer.fov` |

**How it works.** Wider angles show more of the scene but exaggerate perspective.

- Most phone footage is around 55–70°.

### Anti-aliasing filter

> Applies the Mip-Splatting style screen-space filter when rendering.

| | |
|---|---|
| **Default** | Automatic |
| **Mode** | Advanced |
| **ID** | `viewer.antialias` |

**How it works.** Scenes trained with anti-aliasing must be rendered with it (otherwise they look slightly too thin/bright when zoomed out); scenes trained without it look best without.

- 'Automatic' follows the flag stored in the file.

## Editing

| Setting | Default | Range | Mode |
|---|---|---|---|
| [Crop box](#crop-box) | Off | – | Easy |
| [Delete below opacity](#delete-below-opacity) | 0.05 | 0–1 | Easy |
| [Delete larger than](#delete-larger-than) | 0.1 | 0.001–1 | Advanced |
| [Remove isolated splats](#remove-isolated-splats) | N = 4, radius 1 % | – | Advanced |
| [Selection brush](#selection-brush) | Radius 20 px | – | Easy |

### Crop box

> Hides everything outside a box; Apply deletes it permanently.

| | |
|---|---|
| **Default** | Off |
| **Mode** | Easy |
| **ID** | `edit.cropBox` |

**How it works.** Use the sliders (or type values) to fit the box around your subject. While enabled the outside is only hidden; 'Apply crop' marks it deleted, and exporting writes only the remaining splats.

- Undo with ⌘Z.

### Delete below opacity

> Removes nearly transparent splats (usually haze and floaters).

| | |
|---|---|
| **Default** | 0.05 |
| **Range** | 0–1 |
| **Mode** | Easy |
| **ID** | `edit.opacityThreshold` |

**How it works.** Most floaters are low-opacity gaussians that only contribute in a few training views.

**When to change it**

- Deleting below 0.05–0.1 typically removes 10–30 % of splats with no visible loss.
- Preview it first with Display → Opacity.

### Delete larger than

> Removes splats whose largest axis exceeds this fraction of the scene size.

| | |
|---|---|
| **Default** | 0.1 |
| **Range** | 0.001–1 |
| **Mode** | Advanced |
| **ID** | `edit.sizeThreshold` |

**How it works.** Huge splats are usually sky or background blobs that only make sense from training views.

- 0.1 = 10 % of the scene diameter.

### Remove isolated splats

> Deletes splats with fewer than N neighbours within a radius.

| | |
|---|---|
| **Default** | N = 4, radius 1 % |
| **Mode** | Advanced |
| **ID** | `edit.outlierRadius` |

**How it works.** Statistical outlier removal: good for specks floating far from any surface.

- Radius is relative to the scene size.

### Selection brush

> Click or drag in the viewer to select splats under the cursor.

| | |
|---|---|
| **Default** | Radius 20 px |
| **Mode** | Easy |
| **ID** | `edit.brush` |

**How it works.** Hold ⌥ while dragging to deselect. Selected splats are tinted orange; press Delete to remove them or ⌘I to invert the selection.

- The brush selects everything along the view ray within the brush radius — rotate the scene to select only the front-most splats.

## Sources & frame extraction

| Setting | Default | Range | Mode |
|---|---|---|---|
| [Maximum image size](#maximum-image-size) | 3200 px | 800–8192 px | Advanced |
| [Frames per second to extract](#frames-per-second-to-extract) | 3 fps | 0.5–30 | Easy |
| [Target frame count](#target-frame-count) | 0 (use fps) | 0–3000 | Easy |
| [Pick sharpest frame](#pick-sharpest-frame) | On | – | Easy |
| [Reject blurry photos](#reject-blurry-photos) | 0.35 | 0–0.9 | Advanced |
| [360°: views per frame](#360-views-per-frame) | 8 | 4–20 | Easy |
| [360°: view field of view](#360-view-field-of-view) | 90° | 60–120° | Advanced |
| [360°: drop the floor view](#360-drop-the-floor-view) | On | – | Easy |
| [Stereo layout](#stereo-layout) | Automatic | – | Easy |
| [Stereo baseline](#stereo-baseline) | From metadata | – | Advanced |
| [Compute stereo depth](#compute-stereo-depth) | Off | – | Advanced |
| [HDR tone mapping](#hdr-tone-mapping) | Automatic | – | Advanced |
| [Hold out every Nth image](#hold-out-every-nth-image) | 25 (Easy) / 8 (Advanced) | 0–50 | Advanced |

### Maximum image size

> Longest side of the images stored in the dataset.

| | |
|---|---|
| **Default** | 3200 px |
| **Range** | 800–8192 px |
| **Mode** | Advanced |
| **ID** | `source.maxImageSize` |

**How it works.** Source photos are downscaled once when the dataset is built. Bigger keeps more detail for later high-quality training but takes more disk and longer preparation.

- Training has its own (smaller or equal) limit.

### Frames per second to extract

> How densely to sample frames from a video.

| | |
|---|---|
| **Default** | 3 fps |
| **Range** | 0.5–30 |
| **Mode** | Easy |
| **ID** | `source.videoFPS` |

**How it works.** Gaussian splatting needs lots of overlap between views but nearly duplicate frames waste time.

**When to change it**

- Walking slowly with a phone, 2–4 fps is ideal; for fast drone or car footage use 5–10.
- Alternatively set a target frame count and the app spaces frames evenly.

### Target frame count

> Extract roughly this many frames regardless of video length (0 = use fps).

| | |
|---|---|
| **Default** | 0 (use fps) |
| **Range** | 0–3000 |
| **Mode** | Easy |
| **ID** | `source.targetFrames` |

**How it works.** 100–300 frames is the sweet spot for a room or object; 300–800 for larger outdoor scenes.

**When to change it**

- More frames = better coverage but slower camera solving (exhaustive matching grows quadratically).

### Pick sharpest frame

> Within each sampling window, keep the least blurry frame.

| | |
|---|---|
| **Default** | On |
| **Mode** | Easy |
| **ID** | `source.sharpestInWindow` |

**How it works.** Measures sharpness (variance of the Laplacian) of every candidate frame in the window and keeps the best one.

- This alone removes most motion blur, which is the #1 cause of fuzzy splats.
- Costs a little extra decode time.

### Reject blurry photos

> Drops photos that are much blurrier than the rest of the set.

| | |
|---|---|
| **Default** | 0.35 |
| **Range** | 0–0.9 |
| **Mode** | Advanced |
| **ID** | `source.blurRejection` |

**How it works.** A photo is rejected when its sharpness score is below this fraction of the median.

- 0 disables rejection.

### 360°: views per frame

> How many perspective cameras are cut out of each equirectangular frame.

| | |
|---|---|
| **Default** | 8 |
| **Range** | 4–20 |
| **Mode** | Easy |
| **ID** | `source.360views` |

**How it works.** 360° frames can't be used directly by the trainer (pinhole cameras only), so each frame is reprojected into several virtual cameras looking in different directions.

- 8 views (a ring of 6 + up/down) covers the sphere; 4–6 is enough when walking through a space.
- Each virtual view becomes a training image, so this multiplies dataset size.

### 360°: view field of view

> Field of view of each virtual camera.

| | |
|---|---|
| **Default** | 90° |
| **Range** | 60–120° |
| **Mode** | Advanced |
| **ID** | `source.360fov` |

**How it works.** 90° with 8 views gives good overlap between neighbouring views (needed for camera solving).

- Wider views add more overlap but stretch pixels at the edges.

### 360°: drop the floor view

> Skips the straight-down view, which usually shows the tripod, selfie stick or you.

| | |
|---|---|
| **Default** | On |
| **Mode** | Easy |
| **ID** | `source.360dropNadir` |

**How it works.** Recommended for handheld/stick captures.

- Also see Masking for removing the operator from the remaining views.

### Stereo layout

> How left and right eyes are packed: side-by-side, top-bottom, two files, MV-HEVC, or automatic.

| | |
|---|---|
| **Default** | Automatic |
| **Mode** | Easy |
| **ID** | `source.stereoLayout` |

**How it works.** Automatic inspects the video (Apple spatial video MV-HEVC layers, Arcturus Vision recordings, 2:1 / 1:2 aspect ratios).

- Each eye becomes its own camera; the known baseline between the eyes is used to give the reconstruction real-world (metric) scale.

### Stereo baseline

> Distance between the two lenses in millimetres (0 = unknown / from metadata).

| | |
|---|---|
| **Default** | From metadata |
| **Mode** | Advanced |
| **ID** | `source.stereoBaseline` |

**How it works.** Used to scale the scene to metres and, when stereo depth is enabled, to convert disparity to depth.

- Arcturus Vision ≈ 64 mm, iPhone spatial video ≈ 19 mm (wide + ultra-wide).

### Compute stereo depth

> Estimates a depth map for the left eye from the stereo pair.

| | |
|---|---|
| **Default** | Off |
| **Mode** | Advanced |
| **ID** | `source.stereoDepth` |

**How it works.** Depth maps can supervise training (see Depth supervision) which reduces floaters on texture-poor surfaces.

- Quality depends on calibration; with uncalibrated footage leave this off.

### HDR tone mapping

> How 10-bit HLG/PQ footage is converted to standard images.

| | |
|---|---|
| **Default** | Automatic |
| **Mode** | Advanced |
| **ID** | `source.hdr` |

**How it works.** Training works on standard-range images.

- 'Automatic' applies a filmic tone curve that keeps highlight detail; 'Clip' simply clamps (brighter, loses highlights).

### Hold out every Nth image

> Reserves some images for measuring quality instead of training on them.

| | |
|---|---|
| **Default** | 25 (Easy) / 8 (Advanced) |
| **Range** | 0–50 |
| **Mode** | Advanced |
| **ID** | `source.holdout` |

**How it works.** Held-out images give an honest PSNR (quality) number in the training dashboard, the same way research papers evaluate.

- Training PSNR is a poor guide on its own: it dips while the learning rate is high, and jumps when moving-object masking starts ignoring the worst pixels (about 30 % into training), while the held-out number shows whether the splat really generalises.
- Easy mode holds out every 25th photo (4 %: a trustworthy trend for almost no cost); Advanced uses every 8th (the research convention).
- 0 trains on everything.

## Camera solving (SfM)

| Setting | Default | Range | Mode |
|---|---|---|---|
| [Camera solver](#camera-solver) | Automatic (capture poses if present, else Object Capture) | – | Easy |
| [Feature model](#feature-model) | SuperPoint + LightGlue (if installed) else SIFT | – | Easy |
| [Pair selection](#pair-selection) | Automatic | – | Advanced |
| [Sequential overlap](#sequential-overlap) | 10 | 3–50 | Advanced |
| [Retrieval neighbours](#retrieval-neighbours) | 30 | 5–150 | Advanced |
| [Mapper](#mapper) | Automatic | – | Advanced |
| [Lens model](#lens-model) | Automatic | – | Advanced |
| [All images from one camera](#all-images-from-one-camera) | On | – | Advanced |
| [Keypoints per image](#keypoints-per-image) | 4096 | 512–16384 | Advanced |
| [Refine capture poses](#refine-capture-poses) | Off | – | Advanced |

### Camera solver

> How camera positions are recovered from your images.

| | |
|---|---|
| **Default** | Automatic (capture poses if present, else Object Capture) |
| **Mode** | Easy |
| **ID** | `sfm.solver` |

**How it works.** Apple Object Capture: built in, uses the Neural Engine and GPU, very robust for objects and rooms, no install.

- COLMAP/GLOMAP + learned features (via the Models pack): the research standard, more control, best for large or difficult scenes, can run on your Linux GPU worker.
- 'From capture' uses the poses recorded by the iPhone (ARKit) — instant and metric, but slightly less precise; enable 'Refine poses' to polish them with SfM.

### Feature model

> Which algorithm finds and matches points between images.

| | |
|---|---|
| **Default** | SuperPoint + LightGlue (if installed) else SIFT |
| **Mode** | Easy |
| **ID** | `sfm.features` |

**How it works.** See the Models panel for benchmark results on this Mac.

- Short version: SIFT is the classic baseline (no download, struggles with low texture and big viewpoint changes); SuperPoint+LightGlue is the best all-rounder (fast, robust; non-commercial licence); ALIKED+LightGlue and DISK+LightGlue are permissively licensed alternatives; LoFTR is dense and handles texture-poor walls; MASt3R and VGGT are 3D foundation models that work with very few images (10–30) but need lots of memory.

### Pair selection

> Which image pairs are matched: sequential, exhaustive, retrieval or automatic.

| | |
|---|---|
| **Default** | Automatic |
| **Mode** | Advanced |
| **ID** | `sfm.matching` |

**How it works.** Exhaustive matches every pair (most robust, cost grows with N²; fine up to ~200 images).

- Sequential matches each frame with its neighbours (ideal for video) plus loop-closure checks.
- Retrieval uses a global image descriptor to pick the K most similar images (best for large unordered photo sets).
- Automatic chooses for you.

### Sequential overlap

> Number of following frames each video frame is matched with.

| | |
|---|---|
| **Default** | 10 |
| **Range** | 3–50 |
| **Mode** | Advanced |
| **ID** | `sfm.overlap` |

**How it works.** Higher is more robust to blurry or textureless stretches but slower.

### Retrieval neighbours

> Number of most-similar images matched per image in retrieval mode.

| | |
|---|---|
| **Default** | 30 |
| **Range** | 5–150 |
| **Mode** | Advanced |
| **ID** | `sfm.topK` |

**How it works.** 30–50 is typical.

**When to change it**

- Raise it if parts of the scene fail to register.

### Mapper

> Automatic, GLOMAP (global, fast on big sets) or COLMAP (incremental, robust).

| | |
|---|---|
| **Default** | Automatic |
| **Mode** | Advanced |
| **ID** | `sfm.mapper` |

**How it works.** GLOMAP solves all cameras at once and is 10–100× faster on large sets.

- COLMAP's incremental mapper adds cameras one at a time: slower on big sets but much more forgiving with weak connectivity, repeated textures and unknown focal length — in our benchmark a 360° orbit without EXIF focal scored AUC@5° 0.98 incremental vs 0.38 global.
- Automatic uses COLMAP up to 300 images (it only takes seconds to minutes there) and GLOMAP above, estimating the focal length from the view graph first when photos have no EXIF focal (e.g. video frames).

### Lens model

> Pinhole, OpenCV (radial + tangential distortion) or fisheye.

| | |
|---|---|
| **Default** | Automatic |
| **Mode** | Advanced |
| **ID** | `sfm.cameraModel` |

**How it works.** Automatic estimates distortion and then undistorts images for training.

**When to change it**

- Choose fisheye for GoPro/action cameras and 'Pinhole' only for already-undistorted images.

### All images from one camera

> Share one set of lens intrinsics across all images.

| | |
|---|---|
| **Default** | On |
| **Mode** | Advanced |
| **ID** | `sfm.singleCamera` |

**How it works.** Correct for a single video or a photo set from one camera at one zoom.

**When to change it**

- Turn off when mixing devices or zoom levels.

### Keypoints per image

> Maximum number of features extracted per image.

| | |
|---|---|
| **Default** | 4096 |
| **Range** | 512–16384 |
| **Mode** | Advanced |
| **ID** | `sfm.maxKeypoints` |

**How it works.** More keypoints help with large images and low texture at the cost of matching time.

### Refine capture poses

> Uses iPhone/headset poses as a starting point and polishes them with SfM.

| | |
|---|---|
| **Default** | Off |
| **Mode** | Advanced |
| **ID** | `sfm.refineCapturePoses` |

**How it works.** ARKit poses drift by a few centimetres over a long capture; refining removes the drift while keeping metric scale.

- Adds a few minutes.

## Masking

| Setting | Default | Range | Mode |
|---|---|---|---|
| [Mask model](#mask-model) | None | – | Easy |
| [Invert mask](#invert-mask) | Off | – | Advanced |

### Mask model

> Automatically ignore moving things (people, pets, vehicles), the sky, or everything except the subject.

| | |
|---|---|
| **Default** | None |
| **Mode** | Easy |
| **ID** | `mask.model` |

**How it works.** Masked pixels are excluded from the training loss, so moving people, pets, cars, the operator's shadow or a featureless sky don't create floaters.

- Options: 'Remove people & pets — quick' (Apple Vision, on-device, no download; finds people well, cats/dogs only when clearly visible).
- 'Remove moving things — best' (Models pack: open-vocabulary detection of people, animals, vehicles, bags… plus SAM 2 tracking through the sequence, so objects stay masked even in frames where detection misses them).
- 'Keep main subject' for object captures.
- Sky removal for outdoor scenes.
- You can review and correct any masks afterwards with Edit masks… (click an object once and SAM 2 tracks it through every frame).
- Masks combine with 'Ignore moving objects' during training.

### Invert mask

> Swap what is kept and what is ignored.

| | |
|---|---|
| **Default** | Off |
| **Mode** | Advanced |
| **ID** | `mask.invert` |

**How it works.** For example SAM 2 on a person + invert = remove that person.

## Training — basics

| Setting | Default | Range | Mode |
|---|---|---|---|
| [Quality preset](#quality-preset) | Balanced | – | Easy |
| [Iterations](#iterations) | 30 000 | 1 000–100 000 | Advanced |
| [Training resolution](#training-resolution) | 1600 px | 400–4096 px | Advanced |
| [SH degree](#sh-degree) | 3 | 0–3 | Advanced |
| [SH warm-up interval](#sh-warm-up-interval) | 1000 | 100–5000 | Advanced |
| [Training background](#training-background) | Random | – | Advanced |
| [Anti-aliased rendering](#anti-aliased-rendering) | On | – | Advanced |
| [Random seed](#random-seed) | 42 | – | Advanced |
| [Compute device](#compute-device) | This Mac | – | Easy |

### Quality preset

> Quick preview → Maximum. Sets iterations, resolution and splat budget.

| | |
|---|---|
| **Default** | Balanced |
| **Mode** | Easy |
| **ID** | `train.preset` |

**How it works.** Start with Quick preview to check the camera solve and coverage, then train Balanced or High quality.

- Times are for an M4 Pro with ~200 images.

### Iterations

> Number of optimisation steps (one image per step).

| | |
|---|---|
| **Default** | 30 000 |
| **Range** | 1 000–100 000 |
| **Effect** | More = sharper and fewer artefacts, longer training. |
| **Mode** | Advanced |
| **ID** | `train.iterations` |

**How it works.** Quality rises quickly for the first ~7 k steps and slowly after 15 k. 30 k is the research standard.

- Doubling iterations roughly doubles training time.

### Training resolution

> Longest image side used during training.

| | |
|---|---|
| **Default** | 1600 px |
| **Range** | 400–4096 px |
| **Effect** | Higher = finer detail, more memory, quadratically slower. |
| **Mode** | Advanced |
| **ID** | `train.maxImageSize` |

**How it works.** The single biggest speed lever: cost is proportional to pixel count. 1600 px is the standard 3DGS setting.

**When to change it**

- Use 1000 for previews and 2400+ only when you need to zoom into fine detail.

### SH degree

> Maximum view-dependent colour detail learned.

| | |
|---|---|
| **Default** | 3 |
| **Range** | 0–3 |
| **Mode** | Advanced |
| **ID** | `train.shDegree` |

**How it works.** Degree 3 (16 coefficients) captures reflections and gloss; degree 0 makes files ~4× smaller and trains faster but everything looks matte.

- Most of the file size of a splat is SH coefficients.

### SH warm-up interval

> Adds one SH band every N steps.

| | |
|---|---|
| **Default** | 1000 |
| **Range** | 100–5000 |
| **Mode** | Advanced |
| **ID** | `train.shDegreeInterval` |

**How it works.** Learning flat colour first and adding view-dependence gradually prevents the model from 'explaining' geometry errors with colour tricks.

### Training background

> Colour composited behind the splats during training: black, white or random.

| | |
|---|---|
| **Default** | Random |
| **Mode** | Advanced |
| **ID** | `train.background` |

**How it works.** Random changes every step, which forces the model to make surfaces fully opaque instead of using the background colour; it reduces haze in the final scene.

**When to change it**

- Use black/white for masked object captures where the background really is that colour.

### Anti-aliased rendering

> Mip-Splatting style 2D filter with opacity compensation.

| | |
|---|---|
| **Default** | On |
| **Mode** | Advanced |
| **ID** | `train.antialiased` |

**How it works.** Standard 3DGS dilates every splat by a fixed 0.3 pixels, which makes scenes look too thick when zoomed out and aliased when zoomed in.

- The anti-aliased variant compensates opacity for the dilation, so the scene stays correct at any distance or resolution.

**When to change it**

- Recommended on; the flag is saved in the file so viewers render it correctly.

### Random seed

> Makes runs reproducible.

| | |
|---|---|
| **Default** | 42 |
| **Mode** | Advanced |
| **ID** | `train.seed` |

**How it works.** Same data + same settings + same seed = the same result.

- Change it to get a different random initialisation.

### Compute device

> Train on this Mac, on a networked Linux GPU worker, or both.

| | |
|---|---|
| **Default** | This Mac |
| **Mode** | Easy |
| **ID** | `train.device` |

**How it works.** This Mac: Metal trainer, no setup. Remote worker: runs Splat Worker (CUDA, gsplat) on your PC — typically 3–8× faster on a recent NVIDIA GPU, with live previews streamed back.

- Both: the camera solve and training stages are placed on whichever machine is faster, and large scenes are split into spatial blocks that train in parallel on both machines and are merged automatically.

## Training — densification

| Setting | Default | Range | Mode |
|---|---|---|---|
| [Densification strategy](#densification-strategy) | MCMC | – | Advanced |
| [Splat budget](#splat-budget) | 1 500 000 | 10 000–10 000 000 | Advanced |
| [Densify from step](#densify-from-step) | 500 | – | Advanced |
| [Densify until step](#densify-until-step) | 15 000 | – | Advanced |
| [Densify every](#densify-every) | 100 | – | Advanced |
| [Gradient threshold (adaptive)](#gradient-threshold-adaptive) | 0.0002 | 0.00005–0.001 | Advanced |
| [Absolute gradients (AbsGS)](#absolute-gradients-absgs) | Off | – | Advanced |
| [Opacity reset interval (adaptive)](#opacity-reset-interval-adaptive) | 3000 | – | Advanced |
| [Dead opacity](#dead-opacity) | 0.005 | – | Advanced |
| [Initial opacity](#initial-opacity) | 0.1 | – | Advanced |
| [Random init count](#random-init-count) | 100 000 | – | Advanced |
| [Long-axis splitting](#long-axis-splitting) | On (Mac) · opt-in on the Linux worker until validated there | – | Advanced |

### Densification strategy

> How new splats are created: MCMC (budgeted) or adaptive (classic).

| | |
|---|---|
| **Default** | MCMC |
| **Mode** | Advanced |
| **ID** | `train.strategy` |

**How it works.** MCMC (3DGS as Markov Chain Monte Carlo, 2024) treats splats as samples: dead splats are relocated to where they're needed and the total grows to a fixed budget.

- Predictable memory, fewer floaters, typically +0.5–1 dB PSNR.
- Adaptive density control (original 3DGS) clones/splits splats with large screen-space gradients and prunes transparent ones; it decides the final count itself, which can be 3–6 M for big scenes.

### Splat budget

> Maximum number of gaussians.

| | |
|---|---|
| **Default** | 1 500 000 |
| **Range** | 10 000–10 000 000 |
| **Effect** | More = finer detail, more memory, larger files, slower. |
| **Mode** | Advanced |
| **ID** | `train.maxGaussians` |

**How it works.** With MCMC the scene grows to exactly this number. Memory is roughly 1 KB per gaussian during training (parameters + optimiser state + gradients) and 248 bytes in a PLY.

- 1–1.5 M suits rooms and objects; 3–6 M large outdoor scenes.

### Densify from step

> First step at which splats are added.

| | |
|---|---|
| **Default** | 500 |
| **Mode** | Advanced |
| **ID** | `train.densifyStart` |

**How it works.** A short warm-up lets the initial point cloud settle before growth.

### Densify until step

> Last step at which splats are added or relocated.

| | |
|---|---|
| **Default** | 15 000 |
| **Mode** | Advanced |
| **ID** | `train.densifyStop` |

**How it works.** After this the set of splats is fixed and only refined.

- Typically half of the iterations for adaptive, ~80 % for MCMC.

### Densify every

> Steps between densification passes.

| | |
|---|---|
| **Default** | 100 |
| **Mode** | Advanced |
| **ID** | `train.densifyInterval` |

**How it works.** The standard 100 works for nearly everything.

### Gradient threshold (adaptive)

> How much a splat must 'want to move' before it is cloned or split.

| | |
|---|---|
| **Default** | 0.0002 |
| **Range** | 0.00005–0.001 |
| **Mode** | Advanced |
| **ID** | `train.gradThreshold` |

**How it works.** Lower creates more splats (more detail, more memory).

- The default 0.0002 is from the original paper; with AbsGS absolute gradients it is multiplied by 4.

### Absolute gradients (AbsGS)

> Uses the sum of absolute per-pixel gradients to decide densification.

| | |
|---|---|
| **Default** | Off |
| **Mode** | Advanced |
| **ID** | `train.absGrad` |

**How it works.** Fixes a known weakness of classic densification where opposing gradients cancel out in large blurry regions, leaving them under-reconstructed.

**When to change it**

- Recommended with the adaptive strategy.

### Opacity reset interval (adaptive)

> Periodically makes every splat nearly transparent.

| | |
|---|---|
| **Default** | 3000 |
| **Mode** | Advanced |
| **ID** | `train.opacityReset` |

**How it works.** Forces the optimiser to re-justify every splat, which removes floaters.

- Not used by MCMC.

### Dead opacity

> Splats below this opacity are pruned (adaptive) or relocated (MCMC).

| | |
|---|---|
| **Default** | 0.005 |
| **Mode** | Advanced |
| **ID** | `train.pruneOpacity` |

**How it works.** Lower keeps more faint splats; higher is more aggressive cleanup.

### Initial opacity

> Opacity of the splats created from the initial point cloud.

| | |
|---|---|
| **Default** | 0.1 |
| **Mode** | Advanced |
| **ID** | `train.initialOpacity` |

**How it works.** MCMC works best starting at 0.5; classic 3DGS uses 0.1.

### Random init count

> Splats created when there is no point cloud.

| | |
|---|---|
| **Default** | 100 000 |
| **Mode** | Advanced |
| **ID** | `train.randomInitCount` |

**How it works.** Random initialisation works surprisingly well with MCMC; for adaptive training a real point cloud is strongly recommended.

### Long-axis splitting

> New splats are placed along the parent's longest axis instead of stacked on top of it.

| | |
|---|---|
| **Default** | On (Mac) · opt-in on the Linux worker until validated there |
| **Mode** | Advanced |
| **ID** | `train.longAxisSplit` |

**How it works.** MCMC densification grows and recycles splats by cloning existing ones.

- Standard MCMC stacks the clones on exactly the same spot and shrinks them all; with dense captures and large splat budgets those stacks piled up faster than training could separate them, and long PC runs collapsed.
- Long-axis splitting (ImprovedGS / IGS+) spreads each group evenly along the parent's longest axis and shortens only that axis, so new splats start where detail actually is.
- On a 649-photo bedroom capture it improved held-out PSNR by 2.2 dB.

## Training — learning rates

| Setting | Default | Range | Mode |
|---|---|---|---|
| [Position learning rate](#position-learning-rate) | 1.6e-4 → 1.6e-6 | – | Advanced |
| [Colour learning rate](#colour-learning-rate) | 2.5e-3 / 1.25e-4 | – | Advanced |
| [Opacity learning rate](#opacity-learning-rate) | 0.05 | – | Advanced |
| [Scale learning rate](#scale-learning-rate) | 5e-3 | – | Advanced |
| [Rotation learning rate](#rotation-learning-rate) | 1e-3 | – | Advanced |

### Position learning rate

> How fast splats move (start → end, exponential decay, × scene size).

| | |
|---|---|
| **Default** | 1.6e-4 → 1.6e-6 |
| **Mode** | Advanced |
| **ID** | `train.lrPosition` |

**How it works.** Too high makes the scene jittery and unstable; too low leaves splats stuck.

- It is multiplied by the scene extent so it adapts to scene scale automatically.

### Colour learning rate

> Learning rate for base colour (and 1/20 of it for view-dependent bands).

| | |
|---|---|
| **Default** | 2.5e-3 / 1.25e-4 |
| **Mode** | Advanced |
| **ID** | `train.lrColor` |

**How it works.** Higher adapts colours faster; the separate, smaller rate for higher SH bands keeps view-dependence from overfitting.

### Opacity learning rate

> How fast splats fade in or out.

| | |
|---|---|
| **Default** | 0.05 |
| **Mode** | Advanced |
| **ID** | `train.lrOpacity` |

**How it works.** The default works for all strategies.

### Scale learning rate

> How fast splats grow or shrink (in log space).

| | |
|---|---|
| **Default** | 5e-3 |
| **Mode** | Advanced |
| **ID** | `train.lrScale` |

**How it works.** Lower it if you see splats 'exploding' into huge blobs early in training.

### Rotation learning rate

> How fast splats rotate.

| | |
|---|---|
| **Default** | 1e-3 |
| **Mode** | Advanced |
| **ID** | `train.lrRotation` |

**How it works.** Rarely needs changing.

## Training — losses & regularisation

| Setting | Default | Range | Mode |
|---|---|---|---|
| [Structural similarity weight](#structural-similarity-weight) | 0.2 | 0–1 | Advanced |
| [Depth supervision](#depth-supervision) | 0.1 (when depth present) | 0–1 | Advanced |
| [Ignore moving objects](#ignore-moving-objects) | Off | – | Easy |
| [Moving-object sensitivity](#moving-object-sensitivity) | 3× | 2–8 | Advanced |
| [Edge-aware depth](#edge-aware-depth) | On | – | Advanced |
| [Needle-splat penalty](#needle-splat-penalty) | 0.1 | 0–1 | Advanced |
| [Needle ratio limit](#needle-ratio-limit) | 10 | 2–50 | Advanced |
| [Coverage needed to ignore moving objects](#coverage-needed-to-ignore-moving-objects) | 15° | 5–45 | Advanced |
| [Per-photo colour correction](#per-photo-colour-correction) | On | – | Easy |
| [Colour-correction learning rate](#colour-correction-learning-rate) | 0.02 | 0.001–0.1 | Advanced |
| [Colour-correction strength limit](#colour-correction-strength-limit) | 0.001 | 0–0.1 | Advanced |
| [Opacity regularisation (MCMC)](#opacity-regularisation-mcmc) | 0.01 | – | Advanced |
| [Scale regularisation (MCMC)](#scale-regularisation-mcmc) | 0.01 | – | Advanced |
| [Exploration noise (MCMC)](#exploration-noise-mcmc) | 5e5 | – | Advanced |
| [Evaluate every](#evaluate-every) | 2000 | – | Advanced |

### Structural similarity weight

> Balance between pixel error (L1) and structural similarity (SSIM).

| | |
|---|---|
| **Default** | 0.2 |
| **Range** | 0–1 |
| **Mode** | Advanced |
| **ID** | `train.ssimWeight` |

**How it works.** The loss is (1 − λ)·L1 + λ·(1 − SSIM). SSIM rewards correct local structure and contrast, which keeps textures crisp.

- 0.2 is the standard value.

### Depth supervision

> How strongly LiDAR / stereo depth maps guide geometry.

| | |
|---|---|
| **Default** | 0.1 (when depth present) |
| **Range** | 0–1 |
| **Mode** | Advanced |
| **ID** | `train.depthWeight` |

**How it works.** With iPhone LiDAR captures this dramatically reduces floaters on white walls, floors and windows, where colour alone is ambiguous.

- Ignored when the dataset has no depth.

**When to change it**

- Too high and colour fidelity suffers near depth-map errors (edges, glass).

### Ignore moving objects

> Robust training: pixels that keep disagreeing with every other view (people walking, pets, cars) are left out of the loss.

| | |
|---|---|
| **Default** | Off |
| **Mode** | Easy |
| **ID** | `train.transientMasking` |

**How it works.** Anything that moves while you capture is inconsistent between views; ordinary training tries to explain it with semi-transparent 'ghost' splats (floaters).

- With this on, after textures have had time to converge (30 % of training) each image's solid, strongly-wrong regions in front of already-reconstructed scenery are excluded from training.
- It needs no masks or models and complements them: masks remove known objects up front, this catches whatever slips through.
- Off by default: on real captures it often mistook hard-to-learn detail (textured ceilings, foliage, clutter) for movement, costing 1–2 dB on held-out photos; turn it on when people, pets or cars actually moved through the scene.
- The dashboard shows the share of pixels currently ignored — typically 0–5 %; much more suggests a very busy scene.

### Moving-object sensitivity

> How much worse than typical a region must be to count as moving (× the image's median error).

| | |
|---|---|
| **Default** | 3× |
| **Range** | 2–8 |
| **Mode** | Advanced |
| **ID** | `train.transientSensitivity` |

**How it works.** Lower catches subtler movers (shadows, swaying plants) but risks ignoring hard-to-learn detail; higher only catches obvious objects.

- There is also a fixed minimum of 10 % colour error.

### Edge-aware depth

> Trusts LiDAR / stereo depth less at image edges, where depth sensors are least reliable.

| | |
|---|---|
| **Default** | On |
| **Mode** | Advanced |
| **ID** | `train.depthEdgeAware` |

**How it works.** Depth maps from phones are low resolution and blur across object boundaries, so their pull on splats is weighted by how flat the photo is at each pixel (DN-Splatter's exp(−|∇I|) weighting).

- On our bedroom test, edge-aware depth plus the needle penalty improved held-out PSNR by about 0.3 dB.

**When to change it**

- Only matters when the dataset has depth.

### Needle-splat penalty

> Discourages splats much longer than they are wide — the streaks you see in sparsely covered areas.

| | |
|---|---|
| **Default** | 0.1 |
| **Range** | 0–1 |
| **Mode** | Advanced |
| **ID** | `train.anisotropyReg` |

**How it works.** When a surface is seen from only a few directions, splats can stretch along the line of sight and still match those photos, which shows up as radial streaks from other viewpoints.

- This adds a penalty when a splat's longest axis exceeds the ratio below times its middle axis, and only shortens the long axis.
- Flat discs (two long axes, one very thin) are the normal shape of splats on surfaces and are never penalised.
- 0 turns it off.

### Needle ratio limit

> How elongated a splat may get before the needle penalty applies.

| | |
|---|---|
| **Default** | 10 |
| **Range** | 2–50 |
| **Mode** | Advanced |
| **ID** | `train.anisotropyMaxRatio` |

**How it works.** Long thin splats are legitimate on hair, wires and edges, so the penalty only starts above this length-to-width ratio.

**When to change it**

- Lower = rounder splats everywhere (can blur thin detail).

### Coverage needed to ignore moving objects

> When moving-object masking is on, it switches itself off if photos are further apart than this.

| | |
|---|---|
| **Default** | 15° |
| **Range** | 5–45 |
| **Mode** | Advanced |
| **ID** | `train.transientMaxViewGap` |

**How it works.** To tell a moving object from scenery, every surface must be seen from several nearby views.

- The trainer measures the median gap between each photo and its most similar neighbour (view angle plus camera offset); above this many degrees it keeps moving-object masking off and says so in the log.
- On a sparse bedroom capture (~21° apart) turning it off improved held-out PSNR by about 1 dB, because real scenery was being ignored as 'moving'.
- Dense captures are typically 3–10°.

### Per-photo colour correction

> Learns each photo's exposure, white balance and tint, so differences between photos don't turn into haze or colour noise.

| | |
|---|---|
| **Default** | On |
| **Mode** | Easy |
| **ID** | `train.appearanceCorrection` |

**How it works.** Phones and cameras change exposure and white balance between shots (auto-exposure, walking from sun into shade).

- Without correction the scene has to average those differences, which shows up as washed-out haze, blotchy colour and floaters.
- With this on, every training photo gets a small learned colour transform (3×3 colour matrix + offset) that is applied only while comparing to that photo; the saved splat stays at a neutral, median exposure. iPhone captures seed it with each frame's recorded shutter time and ISO.
- On our garden test it added about 0.4 dB on held-out views even with locked exposure; expect more on video and handheld photos.

**When to change it**

- Turn off only for perfectly controlled studio captures.

### Colour-correction learning rate

> How fast each photo's correction adapts.

| | |
|---|---|
| **Default** | 0.02 |
| **Range** | 0.001–0.1 |
| **Mode** | Advanced |
| **ID** | `train.appearanceLR` |

**How it works.** Each photo's correction only updates when that photo is drawn (once per pass over the dataset), so this is much higher than the per-splat rates.

**When to change it**

- Lower if colours drift between runs; raise for very short trainings.

### Colour-correction strength limit

> Pulls each correction back towards its starting value.

| | |
|---|---|
| **Default** | 0.001 |
| **Range** | 0–0.1 |
| **Mode** | Advanced |
| **ID** | `train.appearanceReg` |

**How it works.** Higher keeps corrections small (the scene itself absorbs more of the variation); lower lets corrections explain larger differences.

**When to change it**

- Too low can let a correction 'paint' content that belongs in the scene.

### Opacity regularisation (MCMC)

> Gently pushes splats towards transparency so unnecessary ones die and get relocated.

| | |
|---|---|
| **Default** | 0.01 |
| **Mode** | Advanced |
| **ID** | `train.opacityReg` |

**How it works.** Part of the MCMC recipe; higher = sparser, cleaner scenes but can thin out fine structure.

### Scale regularisation (MCMC)

> Gently pushes splats to be small.

| | |
|---|---|
| **Default** | 0.01 |
| **Mode** | Advanced |
| **ID** | `train.scaleReg` |

**How it works.** Discourages huge blurry splats; higher favours fine detail but needs a larger budget.

### Exploration noise (MCMC)

> Random jitter applied to low-opacity splats so they explore the scene.

| | |
|---|---|
| **Default** | 5e5 |
| **Mode** | Advanced |
| **ID** | `train.noiseLR` |

**How it works.** The Langevin-dynamics noise term of 3DGS-MCMC. Only nearly transparent splats move noticeably.

**When to change it**

- Set to 0 to disable.

### Evaluate every

> How often quality (PSNR) is measured on held-out images.

| | |
|---|---|
| **Default** | 2000 |
| **Mode** | Advanced |
| **ID** | `train.evalInterval` |

**How it works.** Evaluation takes a few hundred milliseconds; 0 disables it.

## Compute & networking

| Setting | Default | Range | Mode |
|---|---|---|---|
| [Linux GPU workers](#linux-gpu-workers) | None | – | Easy |
| [Work placement](#work-placement) | This Mac | – | Easy |
| [Split into blocks above](#split-into-blocks-above) | 400 images | – | Advanced |
| [Block overlap](#block-overlap) | 15 % | 5–40 % | Advanced |
| [Receive from iPhone](#receive-from-iphone) | On | – | Easy |
| [Require pairing code](#require-pairing-code) | On | – | Advanced |
| [Models pack (Python)](#models-pack-python) | Not installed | – | Easy |

### Linux GPU workers

> Splat Worker servers found on your network (Bonjour) or added manually.

| | |
|---|---|
| **Default** | None |
| **Mode** | Easy |
| **ID** | `net.workers` |

**How it works.** Install Splat Worker on a Linux PC with an NVIDIA GPU (see linux/README.md).

- It appears here automatically.
- Add it manually with host:port if your network blocks Bonjour.
- A token protects it from other devices on the LAN.

### Work placement

> Only this Mac, only the worker, or both.

| | |
|---|---|
| **Default** | This Mac |
| **Mode** | Easy |
| **ID** | `net.placement` |

**How it works.** Both: preparation that only exists on the Mac (Object Capture, Apple Vision masks, video decoding) stays local, learned feature matching and training go to the fastest device, and scenes bigger than the block threshold are split into blocks trained on both machines at once.

### Split into blocks above

> Scenes with more images than this are trained as parallel blocks in Both mode.

| | |
|---|---|
| **Default** | 400 images |
| **Mode** | Advanced |
| **ID** | `net.blockThreshold` |

**How it works.** Blocks overlap by a margin and are merged seamlessly.

**When to change it**

- Smaller scenes train as one job on the fastest device, which is more efficient than splitting.

### Block overlap

> Extra border around each block, as a fraction of the block size.

| | |
|---|---|
| **Default** | 15 % |
| **Range** | 5–40 % |
| **Mode** | Advanced |
| **ID** | `net.blockMargin` |

**How it works.** More overlap avoids seams at block borders but duplicates work.

### Receive from iPhone

> Lets Splat Capture on your iPhone send captures straight to this Mac over Wi-Fi.

| | |
|---|---|
| **Default** | On |
| **Mode** | Easy |
| **ID** | `net.receive` |

**How it works.** The Mac advertises itself on the local network.

**When to change it**

- Enable 'Require pairing code' so only devices you pair can send captures; received captures land in the iPhone Inbox in the sidebar.

### Require pairing code

> Only accept captures that include the 6-digit code shown on this Mac.

| | |
|---|---|
| **Default** | On |
| **Mode** | Advanced |
| **ID** | `net.pairing` |

**How it works.** Recommended on shared networks.

### Models pack (Python)

> Optional learned feature matchers and segmentation models.

| | |
|---|---|
| **Default** | Not installed |
| **Mode** | Easy |
| **ID** | `net.python` |

**How it works.** Installs a private Python environment with PyTorch (Metal accelerated) and the selected models into ~/Library/Application Support/SplatStudio.

- Nothing is installed system-wide; remove it any time from this panel.

## Export

| Setting | Default | Range | Mode |
|---|---|---|---|
| [File format](#file-format) | PLY | – | Easy |
| [Export SH degree](#export-sh-degree) | Same as scene | – | Advanced |
| [Export only kept splats](#export-only-kept-splats) | On | – | Advanced |
| [Bake orientation](#bake-orientation) | On | – | Easy |
| [Turntable / path video](#turntable--path-video) | 1920×1080, 30 fps, 8 s | – | Advanced |

### File format

> PLY (lossless standard), SPZ (≈10× smaller) or .splat (web).

| | |
|---|---|
| **Default** | PLY |
| **Mode** | Easy |
| **ID** | `export.format` |

**How it works.** PLY is the reference 3DGS format every tool reads (≈248 bytes/splat at SH 3).

- SPZ (Niantic, open) quantises and compresses to ≈25 bytes/splat with very little visible loss and is supported by most modern viewers. .splat is the compact format used by many web viewers but has no view-dependent colour.

### Export SH degree

> Drop view-dependent colour bands to shrink the file.

| | |
|---|---|
| **Default** | Same as scene |
| **Mode** | Advanced |
| **ID** | `export.shDegree` |

**How it works.** SH degree 0 makes PLY files ~4× smaller.

**When to change it**

- Good for web delivery of matte scenes.

### Export only kept splats

> Leaves out deleted splats and anything outside an active crop box.

| | |
|---|---|
| **Default** | On |
| **Mode** | Advanced |
| **ID** | `export.onlyVisible` |

**How it works.** Turn off to export the original scene regardless of edits.

### Bake orientation

> Rotate the exported scene so it opens upright in other viewers.

| | |
|---|---|
| **Default** | On |
| **Mode** | Easy |
| **ID** | `export.bakeOrientation` |

**How it works.** Applies the up direction you chose in the viewer to the splats themselves (positions, rotations and view-dependent colour, rotated exactly), producing a standard −Y-up file.

### Turntable / path video

> Renders a 360° orbit or a fly-through of saved viewpoints to MP4/HEVC.

| | |
|---|---|
| **Default** | 1920×1080, 30 fps, 8 s |
| **Mode** | Advanced |
| **ID** | `export.video` |

**How it works.** Rendered offline at the chosen resolution and frame rate, independent of the viewer window, using hardware video encoding.


