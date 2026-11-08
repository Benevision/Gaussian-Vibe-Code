import Foundation

/// Single source of truth for user-facing documentation of every setting.
/// The UI shows `summary` inline, the ⓘ popover shows everything, the Settings Reference
/// window lists them all, and `SplatStudio --dump-settings` regenerates docs/SETTINGS.md.
struct SettingDoc: Identifiable, Hashable {
    enum Group: String, CaseIterable {
        case viewer = "Viewer"
        case editing = "Editing"
        case sources = "Sources & frame extraction"
        case cameras = "Camera solving (SfM)"
        case masking = "Masking"
        case trainingBasics = "Training — basics"
        case densification = "Training — densification"
        case learningRates = "Training — learning rates"
        case regularisation = "Training — losses & regularisation"
        case compute = "Compute & networking"
        case export = "Export"
    }
    let id: String
    let group: Group
    let title: String
    /// One line, shown under the control.
    let summary: String
    /// Full explanation: what it does, how it works, when to change it.
    let detail: String
    let defaultValue: String
    var range: String = ""
    /// Effects of raising the value / enabling the option.
    var effects: String = ""
    /// Hidden in Easy mode.
    var advanced: Bool = true
}

enum SettingsCatalog {
    static func doc(_ id: String) -> SettingDoc? { byID[id] }
    static let byID: [String: SettingDoc] = Dictionary(uniqueKeysWithValues: all.map { ($0.id, $0) })

    static let all: [SettingDoc] = viewer + editing + sources + cameras + masking + training + compute + export

    // MARK: Viewer
    static let viewer: [SettingDoc] = [
        .init(id: "viewer.displayMode", group: .viewer, title: "Display mode",
              summary: "What each splat shows: colour, depth, centres, ellipse outlines or opacity.",
              detail: "Colour is the normal view. Depth colours every splat by its distance from the camera (warm = near), which is the fastest way to spot floaters — blobs hanging in mid-air show up as colour discontinuities. Centres draws every gaussian as a small dot so you can judge density and coverage. Ellipses draws each gaussian's 2σ outline so you can see size and orientation; long needle-like splats indicate under-constrained regions. Opacity colours by opacity (blue = transparent, red = opaque) and helps tune the 'Delete below opacity' tool.",
              defaultValue: "Colour", effects: "No performance impact.", advanced: false),
        .init(id: "viewer.maxSHDegree", group: .viewer, title: "View-dependent colour (SH degree)",
              summary: "How many spherical-harmonic bands are used for reflections and sheen.",
              detail: "Gaussian splats store colour as spherical harmonics so a surface can look different from different angles (specular highlights, reflections). Degree 0 is a flat colour per splat, 1–3 add progressively finer view-dependence. Lowering it here only affects display: it is a good way to check how much of the look comes from view-dependent effects, and degree 0–1 renders noticeably faster on very large scenes.",
              defaultValue: "3", range: "0–3", effects: "Higher = more realistic reflections, slightly more GPU work per splat.", advanced: false),
        .init(id: "viewer.splatScale", group: .viewer, title: "Splat scale",
              summary: "Multiplies every splat's size (display only).",
              detail: "Values below 1 shrink splats and reveal the structure underneath; above 1 fills holes at the cost of blur. Purely a visualisation aid — exports are unaffected.",
              defaultValue: "1.0", range: "0.1–3.0"),
        .init(id: "viewer.background", group: .viewer, title: "Background colour",
              summary: "Colour behind the scene.",
              detail: "Use a contrasting background to spot semi-transparent haze around objects (try white for dark scenes). Display only.",
              defaultValue: "Near-black", advanced: false),
        .init(id: "viewer.sortPrecision", group: .viewer, title: "Fast 16-bit depth sort",
              summary: "Sorts splats with 16-bit logarithmic depth keys (2 passes instead of 4).",
              detail: "Splats are blended back-to-front, so they must be depth-sorted every frame. 16-bit log-depth keys are ~2× faster to sort and, thanks to the logarithmic mapping, accurate to ~0.02 % of the distance — visually identical in almost all scenes. Switch it off if you see shimmering between overlapping surfaces at very different distances (e.g. a close object in front of a huge landscape).",
              defaultValue: "On", effects: "Off = exact ordering, ~2× more sort time."),
        .init(id: "viewer.renderScale", group: .viewer, title: "Render resolution",
              summary: "Renders at a fraction of the window's pixel resolution, then upscales.",
              detail: "On Retina displays the viewer normally renders at full backing resolution. 0.5 renders a quarter of the pixels, which keeps navigation fluid with 5–10 M splats. Screenshots and videos always render at their own requested resolution.",
              defaultValue: "1.0", range: "0.25–1.0", effects: "Lower = faster, softer image."),
        .init(id: "viewer.alphaCutoff", group: .viewer, title: "Minimum opacity",
              summary: "Splats more transparent than this are skipped.",
              detail: "Raising it hides faint haze and floaters and speeds up rendering; it is a non-destructive preview of 'Delete below opacity'.",
              defaultValue: "1/255 (0.004)", range: "0–0.5"),
        .init(id: "viewer.lazySort", group: .viewer, title: "Skip work when idle",
              summary: "Re-uses the last frame's projection and sort while the camera is still.",
              detail: "Saves power and heat when you are not moving the camera. Disable only when profiling.",
              defaultValue: "On"),
        .init(id: "viewer.showCameras", group: .viewer, title: "Show training cameras",
              summary: "Draws the solved camera positions as frustums.",
              detail: "Lets you check the camera solve: frustums should follow the path you walked. Double-click a training camera in the Cameras list to look through it.",
              defaultValue: "On", advanced: false),
        .init(id: "viewer.cameraMode", group: .viewer, title: "Navigation mode",
              summary: "Orbit around a point, or fly freely with W A S D.",
              detail: "Orbit: drag to turn the scene (like grabbing it), ⇧-drag or right-drag to pan, scroll or pinch to zoom, double-click a point to orbit around it. Fly: W/S forward/back, A/D strafe, Q/E down/up, drag to look, ⇧ faster, ⌥ slower. R resets the view, F frames everything, 1–5 switch display modes, ? shows all shortcuts.",
              defaultValue: "Orbit", advanced: false),
        .init(id: "nav.rotateSpeed", group: .viewer, title: "Rotation speed",
              summary: "How fast dragging turns the scene.",
              detail: "1.0 turns the scene about a quarter of a degree per point dragged — a drag across a laptop-sized viewer turns it roughly 270°. Dragging works like grabbing the scene: drag right and it turns right, drag down and its top tilts towards you.",
              defaultValue: "1.0", range: "0.2–3", advanced: false),
        .init(id: "nav.panSpeed", group: .viewer, title: "Pan speed",
              summary: "How fast ⇧-drag, right-drag (or two-finger drag in Pan mode) slides the view.",
              detail: "At 1.0 a point at the distance you're looking at stays under the cursor while you pan — the orbit pivot in Orbit mode, the surface ahead in Fly mode — so panning feels the same in both modes and at any zoom.",
              defaultValue: "1.0", range: "0.2–3"),
        .init(id: "nav.zoomSpeed", group: .viewer, title: "Zoom speed",
              summary: "How fast the scroll wheel and pinch zoom.",
              detail: "At 1.0 one mouse-wheel notch zooms about 8 %. Trackpad: pinch, or ⌘ + two-finger drag.",
              defaultValue: "1.0", range: "0.2–3", advanced: false),
        .init(id: "nav.flySpeed", group: .viewer, title: "Fly speed",
              summary: "Walking speed in Fly mode (W A S D Q E).",
              detail: "Speed follows how far away the surface in front of you is (measured a few times a second): at 1.0 you cover a quarter of that distance per second, so you slow down as you approach something and speed up across open space — from the framed overview that is a quarter of the scene per second. Hold ⇧ for 4× faster, ⌥ for 4× slower. Switching back to Orbit pivots on the surface in the middle of the view.",
              defaultValue: "1.0", range: "0.1–5"),
        .init(id: "nav.trackpad", group: .viewer, title: "Two-finger drag",
              summary: "What a two-finger trackpad drag does: orbit or pan.",
              detail: "Orbit follows your fingers like grabbing the scene; hold ⇧ to temporarily switch to the other mode. Pinch always zooms, two-finger rotate turns the scene. The direction follows your fingers regardless of the system 'natural scrolling' setting.",
              defaultValue: "Orbit", advanced: false),
        .init(id: "nav.invertVertical", group: .viewer, title: "Invert vertical rotation",
              summary: "Reverse the up/down direction of orbiting.",
              detail: "For people used to flight-stick style controls.",
              defaultValue: "Off"),
        .init(id: "nav.smoothing", group: .viewer, title: "Smooth camera",
              summary: "Eases camera movement instead of jumping exactly with each input event.",
              detail: "Makes navigation feel calmer, especially with trackpads and mouse wheels that send coarse steps. Turn off for the most direct response.",
              defaultValue: "On"),
        .init(id: "viewer.orientation", group: .viewer, title: "Scene up direction",
              summary: "Which axis of the scene points up. Fix upside-down or sideways splats here.",
              detail: "Splat files don't store which way is up, and tools disagree (COLMAP/3DGS use −Y, OpenGL/ARKit +Y, Blender and drones +Z). Splat Studio levels every dataset it builds to −Y, but files from elsewhere may need help: pick an axis, Flip upside down, or rotate until it looks level and choose Level to view. Your choice is remembered for that file, and Export can bake it in so other viewers show it upright too.",
              defaultValue: "−Y (or as saved with the file)", advanced: false),
        .init(id: "viewer.fov", group: .viewer, title: "Field of view",
              summary: "Vertical field of view of the viewer camera.",
              detail: "Wider angles show more of the scene but exaggerate perspective. Most phone footage is around 55–70°.",
              defaultValue: "50°", range: "20–110°"),
        .init(id: "viewer.antialias", group: .viewer, title: "Anti-aliasing filter",
              summary: "Applies the Mip-Splatting style screen-space filter when rendering.",
              detail: "Scenes trained with anti-aliasing must be rendered with it (otherwise they look slightly too thin/bright when zoomed out); scenes trained without it look best without. 'Automatic' follows the flag stored in the file.",
              defaultValue: "Automatic"),
    ]

    // MARK: Editing
    static let editing: [SettingDoc] = [
        .init(id: "edit.cropBox", group: .editing, title: "Crop box",
              summary: "Hides everything outside a box; Apply deletes it permanently.",
              detail: "Use the sliders (or type values) to fit the box around your subject. While enabled the outside is only hidden; 'Apply crop' marks it deleted, and exporting writes only the remaining splats. Undo with ⌘Z.",
              defaultValue: "Off", advanced: false),
        .init(id: "edit.autoClean", group: .editing, title: "Auto clean",
              summary: "Selects splats that make no visible difference from any of the project's photos, for review before deleting.",
              detail: "Renders every training and held-out photo and records, for each splat, the most it adds to any pixel. Splats that never add more than 2 % are selected: on a 649-photo bedroom that was 16 % of the splats, and on a PC-trained drone survey 69 %, with no change in held-out quality, so files get smaller and stray specks go. Include floaters (below) also selects splats seen by only a few photos. It does not make an indoor scene look tidy from outside — what you see from outside a room is mostly the back of its own walls, which every inside view needs (F jumps back inside). Nothing is deleted until you press Delete selected; Undo restores it. Needs the project open and a finished splat.",
              defaultValue: "2 % threshold", advanced: false),
        .init(id: "edit.autoCleanFewViews", group: .editing, title: "Include floaters",
              summary: "Auto clean also selects splats clearly visible in only a few photos — blobs hanging in front of the cameras.",
              detail: "A real surface shows up in most photos that look its way; a floater just in front of a few cameras only in those few, where it helps match them and blocks the view from anywhere closer. With this on, splats clearly visible in fewer than a quarter as many photos as a typical splat are selected too. On a 1,258-photo drone survey that removed every floater more than 25 m above the ground (the ones you fly into when zooming down) and touched 0.2 % of ground and tree splats, for −0.1 dB held-out. Indoors it can also pick close-up surfaces that only a few photos saw (bedroom: −0.07 dB on average), so check the orange selection before deleting.",
              defaultValue: "On", advanced: false),
        .init(id: "edit.opacityThreshold", group: .editing, title: "Delete below opacity",
              summary: "Removes nearly transparent splats (usually haze and floaters).",
              detail: "Most floaters are low-opacity gaussians that only contribute in a few training views. Deleting below 0.05–0.1 typically removes 10–30 % of splats with no visible loss. Preview it first with Display → Opacity.",
              defaultValue: "0.05", range: "0–1", advanced: false),
        .init(id: "edit.sizeThreshold", group: .editing, title: "Delete larger than",
              summary: "Removes splats whose largest axis exceeds this fraction of the scene size.",
              detail: "Huge splats are usually sky or background blobs that only make sense from training views. 0.1 = 10 % of the scene diameter.",
              defaultValue: "0.1", range: "0.001–1"),
        .init(id: "edit.outlierRadius", group: .editing, title: "Remove isolated splats",
              summary: "Deletes splats with fewer than N neighbours within a radius.",
              detail: "Statistical outlier removal: good for specks floating far from any surface. Radius is relative to the scene size.",
              defaultValue: "N = 4, radius 1 %"),
        .init(id: "edit.brush", group: .editing, title: "Selection brush",
              summary: "Click or drag in the viewer to select splats under the cursor.",
              detail: "Hold ⌥ while dragging to deselect. Selected splats are tinted orange; press Delete to remove them or ⌘I to invert the selection. The brush selects everything along the view ray within the brush radius — rotate the scene to select only the front-most splats.",
              defaultValue: "Radius 20 px", advanced: false),
    ]

    // MARK: Sources
    static let sources: [SettingDoc] = [
        .init(id: "source.maxImageSize", group: .sources, title: "Maximum image size",
              summary: "Longest side of the images stored in the dataset.",
              detail: "Source photos are downscaled once when the dataset is built. Bigger keeps more detail for later high-quality training but takes more disk and longer preparation. Training has its own (smaller or equal) limit.",
              defaultValue: "3200 px", range: "800–8192 px", advanced: true),
        .init(id: "source.videoFPS", group: .sources, title: "Frames per second to extract",
              summary: "How densely to sample frames from a video.",
              detail: "Gaussian splatting needs lots of overlap between views but nearly duplicate frames waste time. Walking slowly with a phone, 2–4 fps is ideal; for fast drone or car footage use 5–10. Alternatively set a target frame count and the app spaces frames evenly.",
              defaultValue: "3 fps", range: "0.5–30", advanced: false),
        .init(id: "source.targetFrames", group: .sources, title: "Target frame count",
              summary: "Extract roughly this many frames regardless of video length (0 = use fps).",
              detail: "100–300 frames is the sweet spot for a room or object; 300–800 for larger outdoor scenes. More frames = better coverage but slower camera solving (exhaustive matching grows quadratically).",
              defaultValue: "0 (use fps)", range: "0–3000", advanced: false),
        .init(id: "source.sharpestInWindow", group: .sources, title: "Pick sharpest frame",
              summary: "Within each sampling window, keep the least blurry frame.",
              detail: "Measures sharpness (variance of the Laplacian) of every candidate frame in the window and keeps the best one. This alone removes most motion blur, which is the #1 cause of fuzzy splats. Costs a little extra decode time.",
              defaultValue: "On", advanced: false),
        .init(id: "source.blurRejection", group: .sources, title: "Reject blurry photos",
              summary: "Drops photos that are much blurrier than the rest of the set.",
              detail: "A photo is rejected when its sharpness score is below this fraction of the median. 0 disables rejection.",
              defaultValue: "0.35", range: "0–0.9"),
        .init(id: "source.360views", group: .sources, title: "360°: views per frame",
              summary: "How many perspective cameras are cut out of each equirectangular frame.",
              detail: "360° frames can't be used directly by the trainer (pinhole cameras only), so each frame is reprojected into several virtual cameras looking in different directions. 8 views (a ring of 6 + up/down) covers the sphere; 4–6 is enough when walking through a space. Each virtual view becomes a training image, so this multiplies dataset size.",
              defaultValue: "8", range: "4–20", advanced: false),
        .init(id: "source.360fov", group: .sources, title: "360°: view field of view",
              summary: "Field of view of each virtual camera.",
              detail: "90° with 8 views gives good overlap between neighbouring views (needed for camera solving). Wider views add more overlap but stretch pixels at the edges.",
              defaultValue: "90°", range: "60–120°"),
        .init(id: "source.360dropNadir", group: .sources, title: "360°: drop the floor view",
              summary: "Skips the straight-down view, which usually shows the tripod, selfie stick or you.",
              detail: "Recommended for handheld/stick captures. Also see Masking for removing the operator from the remaining views.",
              defaultValue: "On", advanced: false),
        .init(id: "source.stereoLayout", group: .sources, title: "Stereo layout",
              summary: "How left and right eyes are packed: side-by-side, top-bottom, two files, MV-HEVC, or automatic.",
              detail: "Automatic inspects the video (Apple spatial video MV-HEVC layers, Arcturus Vision recordings, 2:1 / 1:2 aspect ratios). Each eye becomes its own camera; the known baseline between the eyes is used to give the reconstruction real-world (metric) scale.",
              defaultValue: "Automatic", advanced: false),
        .init(id: "source.stereoBaseline", group: .sources, title: "Stereo baseline",
              summary: "Distance between the two lenses in millimetres (0 = unknown / from metadata).",
              detail: "Used to scale the scene to metres and, when stereo depth is enabled, to convert disparity to depth. Arcturus Vision ≈ 64 mm, iPhone spatial video ≈ 19 mm (wide + ultra-wide).",
              defaultValue: "From metadata"),
        .init(id: "source.stereoDepth", group: .sources, title: "Compute stereo depth",
              summary: "Estimates a depth map for the left eye from the stereo pair.",
              detail: "Depth maps can supervise training (see Depth supervision) which reduces floaters on texture-poor surfaces. Quality depends on calibration; with uncalibrated footage leave this off.",
              defaultValue: "Off"),
        .init(id: "source.hdr", group: .sources, title: "HDR tone mapping",
              summary: "How 10-bit HLG/PQ footage is converted to standard images.",
              detail: "Training works on standard-range images. 'Automatic' applies a filmic tone curve that keeps highlight detail; 'Clip' simply clamps (brighter, loses highlights).",
              defaultValue: "Automatic"),
        .init(id: "source.holdout", group: .sources, title: "Hold out every Nth image",
              summary: "Reserves some images for measuring quality instead of training on them.",
              detail: "Held-out images give an honest PSNR (quality) number in the training dashboard, the same way research papers evaluate. Training PSNR is a poor guide on its own: it dips while the learning rate is high, and jumps when moving-object masking starts ignoring the worst pixels (about 30 % into training), while the held-out number shows whether the splat really generalises. Easy mode holds out every 25th photo (4 %: a trustworthy trend for almost no cost); Advanced uses every 8th (the research convention). 0 trains on everything.",
              defaultValue: "25 (Easy) / 8 (Advanced)", range: "0–50"),
    ]

    // MARK: Cameras
    static let cameras: [SettingDoc] = [
        .init(id: "sfm.solver", group: .cameras, title: "Camera solver",
              summary: "How camera positions are recovered from your images.",
              detail: "Apple Object Capture: built in, uses the Neural Engine and GPU, very robust for objects and rooms, no install. COLMAP/GLOMAP + learned features (via the Models pack): the research standard, more control, best for large or difficult scenes, can run on your Linux GPU worker. 'From capture' uses the poses recorded by the iPhone (ARKit) — instant and metric, but slightly less precise; enable 'Refine poses' to polish them with SfM.",
              defaultValue: "Automatic (capture poses if present; 360° video: learned SfM, on the worker when in use; else Object Capture)", advanced: false),
        .init(id: "sfm.features", group: .cameras, title: "Feature model",
              summary: "Which algorithm finds and matches points between images.",
              detail: "See the Models panel for benchmark results on this Mac. Short version: SIFT is the classic baseline (no download, struggles with low texture and big viewpoint changes); SuperPoint+LightGlue is the best all-rounder (fast, robust; non-commercial licence); ALIKED+LightGlue and DISK+LightGlue are permissively licensed alternatives; LoFTR is dense and handles texture-poor walls; MASt3R and VGGT are 3D foundation models that work with very few images (10–30) but need lots of memory.",
              defaultValue: "SuperPoint + LightGlue (if installed) else SIFT", advanced: false),
        .init(id: "sfm.matching", group: .cameras, title: "Pair selection",
              summary: "Which image pairs are matched: sequential, exhaustive, retrieval or automatic.",
              detail: "Exhaustive matches every pair (most robust, cost grows with N²; fine up to ~200 images). Sequential matches each frame with its neighbours (ideal for video) plus loop-closure checks. Retrieval uses a global image descriptor to pick the K most similar images (best for large unordered photo sets). Automatic chooses for you.",
              defaultValue: "Automatic"),
        .init(id: "sfm.overlap", group: .cameras, title: "Sequential overlap",
              summary: "Number of following frames each video frame is matched with.",
              detail: "Higher is more robust to blurry or textureless stretches but slower.",
              defaultValue: "10", range: "3–50"),
        .init(id: "sfm.topK", group: .cameras, title: "Retrieval neighbours",
              summary: "Number of most-similar images matched per image in retrieval mode.",
              detail: "30–50 is typical. Raise it if parts of the scene fail to register.",
              defaultValue: "30", range: "5–150"),
        .init(id: "sfm.mapper", group: .cameras, title: "Mapper",
              summary: "Automatic, GLOMAP (global, fast on big sets) or COLMAP (incremental, robust).",
              detail: "GLOMAP solves all cameras at once and is 10–100× faster on large sets. COLMAP's incremental mapper adds cameras one at a time: slower on big sets, but more forgiving of weak connectivity, repeated textures and unknown focal length. Neither wins everywhere — in our benchmark incremental was far better on a 360° orbit (pose accuracy 0.52–0.98 vs 0.12–0.38) while global was far better on a forward walk with learned matchers (0.90–0.96 vs 0.14–0.50). Automatic therefore runs both on the same matches for sets up to 1,000 photos (mapping takes seconds to minutes; matching is the slow part) and keeps the one that places more photos with longer feature tracks — a measure that picked the more accurate result in every benchmark pair. Above 1,000 photos it uses GLOMAP, estimating focal length from the view graph first when photos have no EXIF focal. When photos carry a focal length (EXIF or iPhone capture), it is held fixed while mapping and freed only in a final adjustment that is kept if it moves less than 8% — on flat scenes such as straight-down drone surveys a free focal length collapses (2795 px → 573 px, 33 of 120 photos placed; held fixed: 118 of 120).",
              defaultValue: "Automatic"),
        .init(id: "sfm.cameraModel", group: .cameras, title: "Lens model",
              summary: "Pinhole, OpenCV (radial + tangential distortion) or fisheye.",
              detail: "Automatic estimates distortion and then undistorts images for training. Choose fisheye for GoPro/action cameras and 'Pinhole' only for already-undistorted images.",
              defaultValue: "Automatic"),
        .init(id: "sfm.singleCamera", group: .cameras, title: "All images from one camera",
              summary: "Share one set of lens intrinsics across all images.",
              detail: "Correct for a single video or a photo set from one camera at one zoom. Turn off when mixing devices or zoom levels.",
              defaultValue: "On"),
        .init(id: "sfm.maxKeypoints", group: .cameras, title: "Keypoints per image",
              summary: "Maximum number of features extracted per image.",
              detail: "More keypoints help with large images and low texture at the cost of matching time.",
              defaultValue: "4096", range: "512–16384"),
        .init(id: "sfm.refineCapturePoses", group: .cameras, title: "Refine capture poses",
              summary: "Uses iPhone/headset poses as a starting point and polishes them with SfM.",
              detail: "ARKit tracks tilt well (gravity) but its compass heading drifts — about 1° over a room scan, which at 1600 px is a 20-pixel disagreement between views of the same wall, so training blurs fine detail. Refinement keeps the capture's poses, per-frame lens values and scale, matches frames that look at the same place (revisits are what correct drift), builds 3D points from the capture poses and adjusts only the camera poses. On a 649-frame bedroom scan: 3.6 min instead of 28 for a full re-solve, all frames kept, +2.1 dB held-out PSNR (a full re-solve gave +2.5 dB). Runs on the Linux worker when one is in use. If it can't run (no Models pack, worker offline) or its corrections look implausible, the capture poses are used as recorded.",
              defaultValue: "On"),
    ]

    // MARK: Masking
    static let masking: [SettingDoc] = [
        .init(id: "mask.model", group: .masking, title: "Mask model",
              summary: "Automatically ignore moving things (people, pets, vehicles), the sky, or everything except the subject.",
              detail: "Masked pixels are excluded from the training loss, so moving people, pets, cars, the operator's shadow or a featureless sky don't create floaters. Options: 'Remove people & pets — quick' (Apple Vision, on-device, no download; finds people well, cats/dogs only when clearly visible). 'Remove moving things — best' (Models pack: open-vocabulary detection of people, animals, vehicles, bags… plus SAM 2 tracking through the sequence, so objects stay masked even in frames where detection misses them). 'Keep main subject' for object captures. Sky removal for outdoor scenes. You can review and correct any masks afterwards with Edit masks… (click an object once and SAM 2 tracks it through every frame). Masks combine with 'Ignore moving objects' during training.",
              defaultValue: "None", advanced: false),
        .init(id: "mask.sky", group: .masking, title: "Sky → learned skybox",
              summary: "Finds the sky in every photo; it's drawn by a learned backdrop at infinity instead of by splats.",
              detail: "Splats can't represent something kilometres away: they try to paint the sky as a brown haze with shards floating in it (the worst views of a drone flight). With this on, a sky model (SegFormer, or a simple bright-and-smooth-at-the-top heuristic when it isn't installed) marks the sky in each photo as a separate layer, stacked on any other masks. Camera solving ignores it, and training draws those pixels with a learned skybox — a panorama behind the scene — while keeping splats out of them. On by default for videos, 360° footage and photo folders; indoor scenes simply have no sky.",
              defaultValue: "On for videos, 360° and photos"),
        .init(id: "mask.staticRegions", group: .masking, title: "Mask fixed parts of the frame",
              summary: "Masks things that move with the camera: a lens rim, a burned-in timestamp or logo, propellers, a car bonnet or selfie stick.",
              detail: "For each camera (each 360° view direction separately) up to 48 frames are compared. Pixels that stay the same while the camera moves, and are either near-black or have edges in the average image, are masked out of training, so they don't get painted into the scene as a smear that follows the camera. Plain sky also stays the same but has no edges, so it is kept, and the boundary of a fixed band (a horizon that stays put in forward flight) is not treated as an overlay. If more than a quarter of the frame looks fixed (the camera barely moves), nothing is masked for that camera.",
              defaultValue: "Off", advanced: true),
        .init(id: "mask.invert", group: .masking, title: "Invert mask",
              summary: "Swap what is kept and what is ignored.",
              detail: "For example SAM 2 on a person + invert = remove that person.",
              defaultValue: "Off"),
    ]

    // MARK: Training
    static let training: [SettingDoc] = [
        .init(id: "train.preset", group: .trainingBasics, title: "Quality preset",
              summary: "Quick preview → Maximum. Sets iterations, resolution and splat budget.",
              detail: "Start with Quick preview to check the camera solve and coverage, then train Balanced or High quality. Times are for an M4 Pro with ~200 images.",
              defaultValue: "Balanced", advanced: false),
        .init(id: "train.iterations", group: .trainingBasics, title: "Iterations",
              summary: "Number of optimisation steps (one image per step).",
              detail: "Quality rises quickly for the first ~7 k steps and slowly after 15 k. 30 k is the research standard. Doubling iterations roughly doubles training time.",
              defaultValue: "30 000", range: "1 000–100 000", effects: "More = sharper and fewer artefacts, longer training."),
        .init(id: "train.maxImageSize", group: .trainingBasics, title: "Training resolution",
              summary: "Longest image side used during training.",
              detail: "The single biggest speed lever: cost is proportional to pixel count. 1600 px is the standard 3DGS setting. Use 1000 for previews and 2400+ only when you need to zoom into fine detail.",
              defaultValue: "1600 px", range: "400–4096 px", effects: "Higher = finer detail, more memory, quadratically slower."),
        .init(id: "train.shDegree", group: .trainingBasics, title: "SH degree",
              summary: "Maximum view-dependent colour detail learned.",
              detail: "Degree 3 (16 coefficients) captures reflections and gloss; degree 0 makes files ~4× smaller and trains faster but everything looks matte. Most of the file size of a splat is SH coefficients.",
              defaultValue: "3", range: "0–3"),
        .init(id: "train.shDegreeInterval", group: .trainingBasics, title: "SH warm-up interval",
              summary: "Adds one SH band every N steps.",
              detail: "Learning flat colour first and adding view-dependence gradually prevents the model from 'explaining' geometry errors with colour tricks.",
              defaultValue: "1000", range: "100–5000"),
        .init(id: "train.background", group: .trainingBasics, title: "Training background",
              summary: "Colour composited behind the splats during training: black, white or random.",
              detail: "Random changes every step, which forces the model to make surfaces fully opaque instead of using the background colour; it reduces haze in the final scene. Use black/white for masked object captures where the background really is that colour.",
              defaultValue: "Random"),
        .init(id: "train.antialiased", group: .trainingBasics, title: "Anti-aliased rendering",
              summary: "Mip-Splatting style 2D filter with opacity compensation.",
              detail: "Standard 3DGS dilates every splat by a fixed 0.3 pixels, which makes scenes look too thick when zoomed out and aliased when zoomed in. The anti-aliased variant compensates opacity for the dilation, so the scene stays correct at any distance or resolution. Recommended on; the flag is saved in the file so viewers render it correctly.",
              defaultValue: "On"),
        .init(id: "train.seed", group: .trainingBasics, title: "Random seed",
              summary: "Makes runs reproducible.",
              detail: "Same data + same settings + same seed = the same result. Change it to get a different random initialisation.",
              defaultValue: "42"),
        .init(id: "train.device", group: .trainingBasics, title: "Compute devices",
              summary: "Tick the machines that may work on this project: this Mac, Linux GPU workers, iPhones running Compute.",
              detail: "One device ticked: everything runs there (this Mac: Metal trainer, no setup; a Linux worker: Splat Worker with CUDA, typically 2–4× faster than an M4 Pro, with live previews). Several: preparation runs on the fastest, and scenes of 400+ photos are split into spatial blocks shared between the ticked devices in proportion to their measured speed (Compute → Benchmark) and memory — a block only goes to a device it fits — then merged automatically. An iPhone with Splat Capture's Compute screen open takes small blocks.",
              defaultValue: "This Mac", advanced: false),

        .init(id: "train.strategy", group: .densification, title: "Densification strategy",
              summary: "How new splats are created: MCMC (budgeted) or adaptive (classic).",
              detail: "MCMC (3DGS as Markov Chain Monte Carlo, 2024) treats splats as samples: dead splats are relocated to where they're needed and the total grows to a fixed budget. Predictable memory, fewer floaters, typically +0.5–1 dB PSNR. Adaptive density control (original 3DGS) clones/splits splats with large screen-space gradients and prunes transparent ones; it decides the final count itself, which can be 3–6 M for big scenes.",
              defaultValue: "MCMC"),
        .init(id: "train.maxGaussians", group: .densification, title: "Splat budget",
              summary: "Maximum number of gaussians.",
              detail: "With MCMC the scene grows to exactly this number. Memory is roughly 1 KB per gaussian during training (parameters + optimiser state + gradients) and 248 bytes in a PLY. 1–1.5 M suits rooms and objects; 3–6 M large outdoor scenes.",
              defaultValue: "1 500 000", range: "10 000–10 000 000", effects: "More = finer detail, more memory, larger files, slower."),
        .init(id: "train.optimizerPrecision", group: .densification, title: "Optimizer memory",
              summary: "Stores the optimizer's running averages in compressed form so more splats fit in GPU memory.",
              detail: "Training keeps two running averages (Adam moments) for every number of every splat, which normally takes more memory than the splats themselves. Compact stores them as 16-bit values, and the colour detail (higher SH) ones as 8-bit codes, with unbiased random rounding so training behaves almost the same. On this Mac it cuts the optimizer's share from 472 to 152 bytes per splat (about 1 GB less at 3 M splats), which matters most on iPhones and smaller Macs. Same scheme as the Linux worker's setting. Full keeps the exact 32-bit values.",
              defaultValue: "Full (fp32)", effects: "Compact: less memory per splat, slightly noisier optimisation; speed about the same."),
        .init(id: "train.densifyStart", group: .densification, title: "Densify from step",
              summary: "First step at which splats are added.",
              detail: "A short warm-up lets the initial point cloud settle before growth.",
              defaultValue: "500"),
        .init(id: "train.densifyStop", group: .densification, title: "Densify until step",
              summary: "Last step at which splats are added or relocated.",
              detail: "After this the set of splats is fixed and only refined. Roughly half of the iterations: stopping growth early leaves more steps to sharpen the splats that exist (on a 30k bedroom run, 14k scored 0.3 dB higher than 25k).",
              defaultValue: "15 000"),
        .init(id: "train.densifyInterval", group: .densification, title: "Densify every",
              summary: "Steps between densification passes.",
              detail: "The standard 100 works for nearly everything.",
              defaultValue: "100"),
        .init(id: "train.gradThreshold", group: .densification, title: "Gradient threshold (adaptive)",
              summary: "How much a splat must 'want to move' before it is cloned or split.",
              detail: "Lower creates more splats (more detail, more memory). The default 0.0002 is from the original paper; with AbsGS absolute gradients it is multiplied by 4.",
              defaultValue: "0.0002", range: "0.00005–0.001"),
        .init(id: "train.absGrad", group: .densification, title: "Absolute gradients (AbsGS)",
              summary: "Uses the sum of absolute per-pixel gradients to decide densification.",
              detail: "Fixes a known weakness of classic densification where opposing gradients cancel out in large blurry regions, leaving them under-reconstructed. Recommended with the adaptive strategy.",
              defaultValue: "Off"),
        .init(id: "train.opacityReset", group: .densification, title: "Opacity reset interval (adaptive)",
              summary: "Periodically makes every splat nearly transparent.",
              detail: "Forces the optimiser to re-justify every splat, which removes floaters. Not used by MCMC.",
              defaultValue: "3000"),
        .init(id: "train.pruneOpacity", group: .densification, title: "Dead opacity",
              summary: "Splats below this opacity are pruned (adaptive) or relocated (MCMC).",
              detail: "Lower keeps more faint splats; higher is more aggressive cleanup.",
              defaultValue: "0.005"),
        .init(id: "train.initialOpacity", group: .densification, title: "Initial opacity",
              summary: "Opacity of the splats created from the initial point cloud.",
              detail: "MCMC works best starting at 0.5; classic 3DGS uses 0.1.",
              defaultValue: "0.1"),
        .init(id: "train.randomInitCount", group: .densification, title: "Random init count",
              summary: "Splats created when there is no point cloud.",
              detail: "Random initialisation works surprisingly well with MCMC; for adaptive training a real point cloud is strongly recommended.",
              defaultValue: "100 000"),

        .init(id: "train.lrPosition", group: .learningRates, title: "Position learning rate",
              summary: "How fast splats move (start → end, exponential decay, × scene size).",
              detail: "Too high makes the scene jittery and unstable; too low leaves splats stuck. It is multiplied by the scene extent so it adapts to scene scale automatically.",
              defaultValue: "1.6e-4 → 1.6e-6"),
        .init(id: "train.lrColor", group: .learningRates, title: "Colour learning rate",
              summary: "Learning rate for base colour (and 1/20 of it for view-dependent bands).",
              detail: "Higher adapts colours faster; the separate, smaller rate for higher SH bands keeps view-dependence from overfitting.",
              defaultValue: "2.5e-3 / 1.25e-4"),
        .init(id: "train.lrOpacity", group: .learningRates, title: "Opacity learning rate",
              summary: "How fast splats fade in or out.",
              detail: "The default works for all strategies.",
              defaultValue: "0.05"),
        .init(id: "train.lrScale", group: .learningRates, title: "Scale learning rate",
              summary: "How fast splats grow or shrink (in log space).",
              detail: "Lower it if you see splats 'exploding' into huge blobs early in training.",
              defaultValue: "5e-3"),
        .init(id: "train.lrRotation", group: .learningRates, title: "Rotation learning rate",
              summary: "How fast splats rotate.",
              detail: "Rarely needs changing.",
              defaultValue: "1e-3"),

        .init(id: "train.ssimWeight", group: .regularisation, title: "Structural similarity weight",
              summary: "Balance between pixel error (L1) and structural similarity (SSIM).",
              detail: "The loss is (1 − λ)·L1 + λ·(1 − SSIM). SSIM rewards correct local structure and contrast, which keeps textures crisp. 0.2 is the standard value.",
              defaultValue: "0.2", range: "0–1"),
        .init(id: "train.depthWeight", group: .regularisation, title: "Depth supervision",
              summary: "How strongly LiDAR / stereo depth maps guide geometry.",
              detail: "With iPhone LiDAR captures this dramatically reduces floaters on white walls, floors and windows, where colour alone is ambiguous. Ignored when the dataset has no depth. Too high and colour fidelity suffers near depth-map errors (edges, glass).",
              defaultValue: "0.1 (when depth present)", range: "0–1"),
        .init(id: "train.transientMasking", group: .regularisation, title: "Ignore moving objects",
              summary: "Robust training: pixels that keep disagreeing with every other view (people walking, pets, cars) are left out of the loss.",
              detail: "Anything that moves while you capture is inconsistent between views; ordinary training tries to explain it with semi-transparent 'ghost' splats (floaters). With this on, after textures have had time to converge (30 % of training) each image's solid, strongly-wrong regions in front of already-reconstructed scenery are excluded from training. It needs no masks or models and complements them: masks remove known objects up front, this catches whatever slips through. Off by default: on real captures it often mistook hard-to-learn detail (textured ceilings, foliage, clutter) for movement, costing 1–2 dB on held-out photos; turn it on when people, pets or cars actually moved through the scene. The dashboard shows the share of pixels currently ignored — typically 0–5 %; much more suggests a very busy scene.",
              defaultValue: "Off", advanced: false),
        .init(id: "train.transientSensitivity", group: .regularisation, title: "Moving-object sensitivity",
              summary: "How much worse than typical a region must be to count as moving (× the image's median error).",
              detail: "Lower catches subtler movers (shadows, swaying plants) but risks ignoring hard-to-learn detail; higher only catches obvious objects. There is also a fixed minimum of 10 % colour error.",
              defaultValue: "3×", range: "2–8"),
        .init(id: "train.depthEdgeAware", group: .regularisation, title: "Edge-aware depth",
              summary: "Trusts LiDAR / stereo depth less at image edges, where depth sensors are least reliable.",
              detail: "Depth maps from phones are low resolution and blur across object boundaries, so their pull on splats is weighted by how flat the photo is at each pixel (DN-Splatter's exp(−|∇I|) weighting). On our bedroom test, edge-aware depth plus the needle penalty improved held-out PSNR by about 0.3 dB. Only matters when the dataset has depth.",
              defaultValue: "On"),
        .init(id: "train.anisotropyReg", group: .regularisation, title: "Needle-splat penalty",
              summary: "Discourages splats much longer than they are wide — the streaks you see in sparsely covered areas.",
              detail: "When a surface is seen from only a few directions, splats can stretch along the line of sight and still match those photos, which shows up as radial streaks from other viewpoints. This adds a penalty when a splat's longest axis exceeds the ratio below times its middle axis, and only shortens the long axis. Flat discs (two long axes, one very thin) are the normal shape of splats on surfaces and are never penalised. 0 turns it off.",
              defaultValue: "0.1", range: "0–1"),
        .init(id: "train.bigSplatReg", group: .regularisation, title: "Big-splat penalty",
              summary: "Shrinks splats that fill a large part of a photo they're drawn in — fog and blobs just in front of a camera.",
              detail: "A splat very close to one camera can cover much of that photo and still help match it, while it fogs the view from everywhere else. This measures how wide each drawn splat looks from the camera (out to where it fades to nothing) and, above about a 33° radius, shrinks it, longest axis first, more the further over it is. On our 649-photo bedroom test it raised held-out PSNR by 0.07 dB and the worst photo by 0.6 dB. 0 turns it off.",
              defaultValue: "10", range: "0–30"),
        .init(id: "train.anisotropyMaxRatio", group: .regularisation, title: "Needle ratio limit",
              summary: "How elongated a splat may get before the needle penalty applies.",
              detail: "Long thin splats are legitimate on hair, wires and edges, so the penalty only starts above this length-to-width ratio. Lower = rounder splats everywhere (can blur thin detail).",
              defaultValue: "10", range: "2–50"),
        .init(id: "train.transientMaxViewGap", group: .regularisation, title: "Coverage needed to ignore moving objects",
              summary: "When moving-object masking is on, it switches itself off if photos are further apart than this.",
              detail: "To tell a moving object from scenery, every surface must be seen from several nearby views. The trainer measures the median gap between each photo and its most similar neighbour (view angle plus camera offset); above this many degrees it keeps moving-object masking off and says so in the log. On a sparse bedroom capture (~21° apart) turning it off improved held-out PSNR by about 1 dB, because real scenery was being ignored as 'moving'. Dense captures are typically 3–10°.",
              defaultValue: "15°", range: "5–45"),
        .init(id: "train.longAxisSplit", group: .densification, title: "Long-axis splitting",
              summary: "New splats are placed along the parent's longest axis instead of stacked on top of it.",
              detail: "MCMC densification grows and recycles splats by cloning existing ones. Standard MCMC stacks the clones on exactly the same spot and shrinks them all; with dense captures and large splat budgets those stacks piled up faster than training could separate them, and long PC runs collapsed. Long-axis splitting (ImprovedGS / IGS+) spreads each group evenly along the parent's longest axis and shortens only that axis, so new splats start where detail actually is. On a 649-photo bedroom capture it improved held-out PSNR by 2.2 dB.",
              defaultValue: "On (Mac and Linux worker)"),
        .init(id: "train.bilateralGrid", group: .regularisation, title: "Local colour correction",
              summary: "Lets each photo have its own shadows and lighting, for long captures where the sun moved.",
              detail: "Each photo gets a coarse 16×16 grid (× 8 brightness levels) of colour adjustments, so a patch that is in shadow in one photo and in sun in another no longer forces the splats to blur the two into a dull smear. The splats keep the average lighting, so moving shadows come out faded rather than baked in. On a 96-minute, 1,258-photo drone flight: 17.16 → 17.91 dB held-out, with the smeared shadow patches gone; training took about 50 % longer (24 vs 16 min on an RTX 4080). Replaces the per-photo colour correction above while on. Works on the Mac and on Linux workers (the Mac estimates the grid gradient from every 4th pixel in each direction). On the Mac: drone 16.37 → 17.18 dB; a 649-photo indoor scan was unchanged (23.44 → 23.46 dB) at the same training time, so it is on by default.",
              defaultValue: "On (Mac and Linux worker)"),
        .init(id: "train.appearanceCorrection", group: .regularisation, title: "Per-photo colour correction",
              summary: "Learns each photo's exposure, white balance and tint, so differences between photos don't turn into haze or colour noise.",
              detail: "Phones and cameras change exposure and white balance between shots (auto-exposure, walking from sun into shade). Without correction the scene has to average those differences, which shows up as washed-out haze, blotchy colour and floaters. With this on, every training photo gets a small learned colour transform (3×3 colour matrix + offset) that is applied only while comparing to that photo; the saved splat stays at a neutral, median exposure. iPhone captures seed it with each frame's recorded shutter time and ISO. It switches itself off when every photo has the same recorded exposure (Splat Capture locks exposure): there is nothing to correct, and the extra freedom cost about 1 dB on held-out views of a 649-frame bedroom scan and let colours drift too bright. Exports bake in the average correction, so the saved splat matches a typical photo. Leave it on for drone footage, video and handheld photos, where exposure changes.",
              defaultValue: "On (off automatically when exposure is locked)", advanced: false),
        .init(id: "train.appearanceLR", group: .regularisation, title: "Colour-correction learning rate",
              summary: "How fast each photo's correction adapts.",
              detail: "Each photo's correction only updates when that photo is drawn (once per pass over the dataset), so this is much higher than the per-splat rates. Lower if colours drift between runs; raise for very short trainings.",
              defaultValue: "0.02", range: "0.001–0.1"),
        .init(id: "train.appearanceReg", group: .regularisation, title: "Colour-correction strength limit",
              summary: "Pulls each correction back towards its starting value.",
              detail: "Higher keeps corrections small (the scene itself absorbs more of the variation); lower lets corrections explain larger differences. Too low can let a correction 'paint' content that belongs in the scene.",
              defaultValue: "0.001", range: "0–0.1"),
        .init(id: "train.opacityReg", group: .regularisation, title: "Opacity regularisation (MCMC)",
              summary: "Gently pushes splats towards transparency so unnecessary ones die and get relocated.",
              detail: "Part of the MCMC recipe; higher = sparser, cleaner scenes but can thin out fine structure.",
              defaultValue: "0.01"),
        .init(id: "train.scaleReg", group: .regularisation, title: "Scale regularisation (MCMC)",
              summary: "Gently pushes splats to be small.",
              detail: "Discourages huge blurry splats; higher favours fine detail but needs a larger budget.",
              defaultValue: "0.01"),
        .init(id: "train.noiseLR", group: .regularisation, title: "Exploration noise (MCMC)",
              summary: "Random jitter applied to low-opacity splats so they explore the scene.",
              detail: "The Langevin-dynamics noise term of 3DGS-MCMC. Only nearly transparent splats move noticeably. Set to 0 to disable.",
              defaultValue: "5e5"),
        .init(id: "train.evalInterval", group: .regularisation, title: "Evaluate every",
              summary: "How often quality (PSNR) is measured on held-out images.",
              detail: "Evaluation takes a few hundred milliseconds; 0 disables it.",
              defaultValue: "2000"),
    ]

    // MARK: Compute
    static let compute: [SettingDoc] = [
        .init(id: "net.benchmark", group: .compute, title: "Benchmark",
              summary: "Measures how fast each machine trains and how fast files move between them.",
              detail: "Every machine trains the same synthetic scene (about 1 million splats, 1600 × 1200 photos, 2,000 steps; steps 1,000–2,000 are timed) and each worker also gets the scene uploaded and its result downloaded, which measures the network. The results replace the built-in guess of each GPU's speed: the training panel shows how long a job would take on each machine (including the upload), and Both mode trains small scenes on whichever finishes first. Run it again after changing hardware, drivers or the worker's runtime. Training on the Mac while another app uses the GPU will make it look slower than it is.",
              defaultValue: "–", advanced: false),
        .init(id: "net.workers", group: .compute, title: "Linux GPU workers",
              summary: "Splat Worker servers found on your network (Bonjour) or added manually.",
              detail: "Install Splat Worker on a Linux PC with an NVIDIA GPU (see linux/README.md). It appears here automatically. Add it manually with host:port if your network blocks Bonjour. A token protects it from other devices on the LAN.",
              defaultValue: "None", advanced: false),
        .init(id: "net.placement", group: .compute, title: "Work placement",
              summary: "Only this Mac, only the worker, or both.",
              detail: "Both: preparation that only exists on the Mac (Object Capture, Apple Vision masks) stays local, learned feature matching and training go to the fastest device, and scenes bigger than the block threshold are split into blocks trained on both machines at once.",
              defaultValue: "This Mac", advanced: false),
        .init(id: "net.videoOnWorker", group: .compute, title: "Split videos on the worker",
              summary: "Upload the video once and let the worker pick frames and cut 360° views.",
              detail: "The video is streamed to the worker unchanged (no zip, no extra copy on this Mac). The worker decodes it on its CPU cores, keeps the sharpest frame in each window, cuts 360° footage into the same flat views as this Mac would (resampled on its GPU) and sends back the images and masks. This frees the Mac for other work during the slowest part of video preparation — an 8K 360° clip decodes about 10,000 frames. Same frame choice and file layout as splitting here, so results are interchangeable. Needs worker 1.3 or newer with the processing models installed; otherwise, or if anything goes wrong on the worker, the video is split on this Mac instead.",
              defaultValue: "On", advanced: true),
        .init(id: "net.blockThreshold", group: .compute, title: "Split into blocks above",
              summary: "Scenes with more images than this are trained as parallel blocks in Both mode.",
              detail: "Blocks overlap by a margin and are merged seamlessly. Smaller scenes train as one job on the fastest device, which is more efficient than splitting.",
              defaultValue: "400 images"),
        .init(id: "net.blockMargin", group: .compute, title: "Block overlap",
              summary: "Extra border around each block, as a fraction of the block size.",
              detail: "More overlap avoids seams at block borders but duplicates work.",
              defaultValue: "15 %", range: "5–40 %"),
        .init(id: "net.receive", group: .compute, title: "Receive from iPhone",
              summary: "Lets Splat Capture on your iPhone send captures straight to this Mac over Wi-Fi.",
              detail: "The Mac advertises itself on the local network. Enable 'Require pairing code' so only devices you pair can send captures; received captures land in the iPhone Inbox in the sidebar.",
              defaultValue: "On", advanced: false),
        .init(id: "net.pairing", group: .compute, title: "Require pairing code",
              summary: "Only accept captures that include the 6-digit code shown on this Mac.",
              detail: "Recommended on shared networks.",
              defaultValue: "On"),
        .init(id: "net.python", group: .compute, title: "Models pack (Python)",
              summary: "Optional learned feature matchers and segmentation models.",
              detail: "Installs a private Python environment with PyTorch (Metal accelerated) and the selected models into ~/Library/Application Support/SplatStudio. Nothing is installed system-wide; remove it any time from this panel.",
              defaultValue: "Not installed", advanced: false),
    ]

    // MARK: Export
    static let export: [SettingDoc] = [
        .init(id: "export.format", group: .export, title: "File format",
              summary: "PLY (lossless standard), SPZ (≈10× smaller), .splat (web) or compressed PLY (PlayCanvas).",
              detail: "PLY is the reference 3DGS format every tool reads (≈248 bytes/splat at SH 3). SPZ (Niantic, open) quantises and compresses to ≈25 bytes/splat with very little visible loss and is supported by most modern viewers. .splat is the compact format used by many web viewers but has no view-dependent colour. C-PLY writes PlayCanvas compressed PLY (.compressed.ply): 256-splat chunks with quantised position, rotation, scale and colour (16 bytes/splat) plus 1 byte per SH value — read by SuperSplat, the PlayCanvas engine, splat-transform and other web viewers.",
              defaultValue: "PLY", advanced: false),
        .init(id: "export.shDegree", group: .export, title: "Export SH degree",
              summary: "Drop view-dependent colour bands to shrink the file.",
              detail: "SH degree 0 makes PLY files ~4× smaller. Good for web delivery of matte scenes.",
              defaultValue: "Same as scene"),
        .init(id: "export.onlyVisible", group: .export, title: "Export only kept splats",
              summary: "Leaves out deleted splats and anything outside an active crop box.",
              detail: "Turn off to export the original scene regardless of edits.",
              defaultValue: "On"),
        .init(id: "export.bakeOrientation", group: .export, title: "Bake orientation",
              summary: "Rotate the exported scene so it opens upright in other viewers.",
              detail: "Applies the up direction you chose in the viewer to the splats themselves (positions, rotations and view-dependent colour, rotated exactly), producing a standard −Y-up file.",
              defaultValue: "On", advanced: false),
        .init(id: "export.video", group: .export, title: "Turntable / path video",
              summary: "Renders a 360° orbit or a fly-through of saved viewpoints to MP4/HEVC.",
              detail: "Rendered offline at the chosen resolution and frame rate, independent of the viewer window, using hardware video encoding.",
              defaultValue: "1920×1080, 30 fps, 8 s"),
    ]

    /// Markdown reference (docs/SETTINGS.md).
    static func markdown() -> String {
        func cell(_ t: String) -> String { t.isEmpty ? "–" : t.replacingOccurrences(of: "|", with: "\\|") }
        func anchor(_ t: String) -> String {
            t.lowercased().unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) || $0 == " " || $0 == "-" }
                .map(String.init).joined().replacingOccurrences(of: " ", with: "-")
        }
        var s = "# Splat Studio — settings reference\n\n"
        s += "Generated from `SettingsCatalog.swift` (`SplatStudio --dump-settings`). \(all.count) settings. "
        s += "**Easy** settings are always visible; **Advanced** ones appear when the toolbar is set to Advanced. "
        s += "The same text is in the app under **Help → Settings Reference** and behind every ⓘ button.\n\n"
        s += "## Contents\n\n"
        for g in SettingDoc.Group.allCases where all.contains(where: { $0.group == g }) {
            s += "- [\(g.rawValue)](#\(anchor(g.rawValue))) — \(all.filter { $0.group == g }.count) settings\n"
        }
        s += "\n"
        for g in SettingDoc.Group.allCases {
            let items = all.filter { $0.group == g }
            guard !items.isEmpty else { continue }
            s += "## \(g.rawValue)\n\n"
            s += "| Setting | Default | Range | Mode |\n|---|---|---|---|\n"
            for d in items {
                s += "| [\(cell(d.title))](#\(anchor(d.title))) | \(cell(d.defaultValue)) | \(cell(d.range)) | \(d.advanced ? "Advanced" : "Easy") |\n"
            }
            s += "\n"
            for d in items {
                s += "### \(d.title)\n\n"
                s += "> \(d.summary)\n\n"
                s += "| | |\n|---|---|\n"
                s += "| **Default** | \(cell(d.defaultValue)) |\n"
                if !d.range.isEmpty { s += "| **Range** | \(cell(d.range)) |\n" }
                if !d.effects.isEmpty { s += "| **Effect** | \(cell(d.effects)) |\n" }
                s += "| **Mode** | \(d.advanced ? "Advanced" : "Easy") |\n"
                s += "| **ID** | `\(d.id)` |\n\n"
                let parts = d.structuredDetail
                if !parts.overview.isEmpty { s += "**How it works.** \(parts.overview)\n\n" }
                for t in parts.details { s += "- \(t)\n" }
                if !parts.details.isEmpty { s += "\n" }
                if !parts.tips.isEmpty {
                    s += "**When to change it**\n\n"
                    for t in parts.tips { s += "- \(t)\n" }
                    s += "\n"
                }
            }
        }
        return s
    }
}

extension SettingDoc {
    /// The detail text split into an explanation and a list of practical guidance, so long entries can be
    /// shown as a paragraph plus bullet points. Sentences that give advice ("Raise it if…", "Too high…",
    /// "Use … when…") become bullets; the rest stays as the explanation, in order.
    var structuredDetail: (overview: String, details: [String], tips: [String]) {
        let sentences = SettingDoc.sentences(detail)
        guard sentences.count > 1 else { return (detail, [], []) }
        // Lead paragraph: the first sentence, plus the second when the first is very short.
        let lead = sentences[0].count < 70 && sentences.count > 2 ? 2 : 1
        let advice = ["raise", "raising", "lower", "lowering", "higher", "increase", "increasing", "decrease", "reduce", "use ",
                      "turn ", "switch", "too ", "if ", "when ", "try", "set ", "keep", "leave", "prefer", "avoid", "only ",
                      "expect", "pick", "choose", "enable", "disable", "don't", "do not", "recommended", "for best", "tip",
                      "start with", "values below", "values above", "below ", "above ", "bigger", "smaller", "more ", "fewer",
                      "less ", "walking", "for fast", "for slow", "preview", "hold ", "deleting", "alternatively", "worth",
                      "consider", "good for", "useful", "best for", "on slow", "on large", "on big"]
        var details: [String] = [], tips: [String] = []
        for sentence in sentences.dropFirst(lead) {
            let l = sentence.lowercased()
            if advice.contains(where: { l.hasPrefix($0) }) { tips.append(sentence) } else { details.append(sentence) }
        }
        return (sentences.prefix(lead).joined(separator: " "), details, tips)
    }

    static func sentences(_ text: String) -> [String] {
        var out: [String] = [], cur = ""
        let chars = Array(text)
        for (i, c) in chars.enumerated() {
            cur.append(c)
            guard c == "." || c == "!" || c == "?", i + 2 < chars.count, chars[i + 1] == " " else { continue }
            let next = chars[i + 2]
            let tail = cur.suffix(5).lowercased()
            if (next.isUppercase || next.isNumber || "“'(".contains(next)) && !tail.hasSuffix("e.g.") && !tail.hasSuffix("i.e.") && !tail.hasSuffix(" vs.") {
                out.append(cur.trimmingCharacters(in: .whitespaces)); cur = ""
            }
        }
        let rest = cur.trimmingCharacters(in: .whitespaces)
        if !rest.isEmpty { out.append(rest) }
        return out
    }
}