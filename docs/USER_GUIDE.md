# SplatWork user guide

This guide walks through making a splat from start to finish: installing, creating a project, training, viewing,
editing and exporting. Every control in the app also has an ⓘ button with a full explanation; the same text is
collected in the [settings reference](SETTINGS.md) and in **Help → Settings Reference** (⇧⌘/, searchable).

**Contents**

1. [Install](#1-install)
2. [The workspace](#2-the-workspace)
3. [Creating a project](#3-creating-a-project)
4. [Projects, history and resuming](#4-projects-history-and-resuming)
5. [Training](#5-training)
6. [Viewing](#6-viewing)
7. [Editing](#7-editing)
8. [Exporting](#8-exporting)
9. [Moving objects](#9-moving-objects-people-pets-cars)
10. [Compute devices and the benchmark](#10-compute-devices-and-the-benchmark)
11. [Setting up a Linux worker](#11-setting-up-a-linux-worker)
12. [Troubleshooting](#12-troubleshooting)

## 1. Install

**SplatWork for Mac** needs a Mac with Apple silicon and macOS 15 or later. Download the `.dmg` from the
[Releases page](https://github.com/Benevision/SplatWork/releases) and drag SplatWork to Applications. The app isn't
notarized yet, so the first time, right-click it and choose **Open**. It checks for updates at launch and every six
hours, installs them when you quit or relaunch, and can be checked by hand with **SplatWork → Check for Updates…**.

To build it yourself instead: `cd mac && ./scripts/build_app.sh && open build/SplatWork.app`.

**Optional extras:**
- **SplatWork for Linux** on a Linux PC with an NVIDIA GPU trains faster and runs the learned camera solvers and masks.
  See [section 11](#11-setting-up-a-linux-worker).
- **SplatWork Capture** on an iPhone (LiDAR models give the best results). See [ios/README.md](../ios/README.md).
- **The Models pack** adds learned feature matchers and segmentation models to the Mac itself. Install it from
  **Models & Benchmarks**.

**Easy and Advanced.** The switch in the toolbar changes mode at any time. **Easy** shows quality presets and the few
choices that matter; **Advanced** shows every parameter. Switching keeps your values.

## 2. The workspace

The sidebar has these pages:

| Page | What it is for |
|---|---|
| **Viewer** | The 3D view, with the inspector on the right (View, Edit, Train and Export tabs). |
| **iPhone Inbox** | Captures sent from SplatWork Capture: the receive switch, the pairing code, the manual address, and the projects and splats made from each capture. |
| **Compute** | This Mac and every Linux worker (found automatically or added by address), their running and recent jobs, the **Benchmark**, and manual worker setup. A green dot means a worker is online. |
| **Models & Benchmarks** | Install the Models pack and compare models: how many photos they place, pose accuracy, time, memory and licence. |
| **Settings Reference** | Every setting with its default, range and explanation. |
| **Projects** | Every project you've worked on, newest first, with where each one stands. See [section 4](#4-projects-history-and-resuming). |
| **Files** | Splat files you opened that don't belong to a project. |

**Home** (⇧⌘H) returns to the start page and unloads the scene to free memory. A running training keeps going; see
[Show Live View](#show-live-view).

## 3. Creating a project

Choose **New** (⌘N), or drop a folder, a video or a capture onto the window. The guided flow has three steps.

### Step 1: Source

| Source | Notes |
|---|---|
| **Photos folder** | Blurry and duplicate photos can be rejected. |
| **Drone / aerial** | A folder of drone photos or a drone video, with the aerial settings below. Drone photos opened any other way are recognised too. |
| **Video** | Frames per second or a target number of frames, and *Pick sharpest frame* (the sharpest frame in each window). |
| **Home video** | Old VHS, Video8 or Hi8 tapes saved as a video file. See [Home video](#home-video-vhs-camcorder-tapes) below. |
| **360° video / 360° photos** | Each equirectangular frame is cut into flat views (8 by default, 90° each). *Drop the floor view* skips the straight-down view showing the tripod or you. |
| **Stereo footage** | Side-by-side, top/bottom, two files, Apple spatial video (MV-HEVC) or two video tracks. The baseline gives the scene metric scale; stereo depth is optional. |
| **iPhone / headset capture** | Poses (and LiDAR depth) come with the capture. Options: use depth for supervision, build the starting splats from depth, drop frames with limited tracking. |
| **COLMAP project / Nerfstudio transforms.json** | Imports existing camera poses. |

Advanced options include the stored image size (1600 px by default) and *Hold out every Nth image* (every 25th in
Easy, every 8th in Advanced), which keeps some photos aside to measure quality honestly.

### Step 2: Cameras & masks

**Camera solver:**
- **Automatic** (default):
  - captures, COLMAP and Nerfstudio: the poses in the source;
  - **360° video**: learned structure-from-motion (SuperPoint + LightGlue with COLMAP / GLOMAP), on a worker when one
    processes the project;
  - circular fisheye footage: learned structure-from-motion with a fisheye lens model;
  - everything else: Apple Object Capture.
- **Apple Object Capture**: built in, nothing to install. Good for objects and rooms. Advanced: high feature
  sensitivity for low-texture scenes.
- **COLMAP / GLOMAP + learned features**: needs the Models pack or a Linux worker. Advanced mode lets you choose the
  feature model, pair selection, mapper and lens model.
- **Use poses from source**: shown when the source has poses.

**Refine capture poses** (Advanced, on by default) polishes iPhone poses with structure-from-motion while keeping
their scale and gravity. On a 649-frame bedroom scan it took 3.6 minutes instead of 28 for a full re-solve and gave
+2.1 dB on held-out photos. If it can't run, the capture's own poses are used.

**Photo locations (GPS).** Phones and drones record where each photo was taken. With *Use photo locations* on (the
default, shown when photos have locations), the imported copies keep only that location, and learned camera solving
checks the solved cameras against it: if they agree, the scene is put in metres with Z up and stray cameras are
dropped; otherwise you get a warning. Turn it off and the project's photos carry no location at all. Apple Object
Capture doesn't use locations. Cameras placed far off the camera path are dropped for any source.

**Drone footage.** Photos from a drone are recognised by the drone's own markers (the camera maker, DJI's flight and
gimbal tags, DJI file names), never by GPS alone, so phone photos with locations are treated as ordinary photos. For
drone footage, or anything started from the Drone / aerial tile, *Aerial settings* switch camera solving to COLMAP /
GLOMAP with SuperPoint + LightGlue (GPS-checked, which needs the Models pack or a worker), turn on the sky, and pick the
High quality preset. Turn the switch off to go back to the standard settings. Locations from a drone video's flight
log (`.SRT`) aren't used yet.

**Masks** mark pixels that training ignores:
- *Remove people & pets — quick* (Apple Vision, built in).
- *Remove moving things — best* (Models pack or a worker: open-vocabulary detection plus SAM 2 tracking). You can
  edit the list of objects to remove.
- *Keep main subject*, *Remove people*, *Remove sky*, *Isolate one object (SAM 2)*. *Invert mask* swaps kept and
  ignored areas.
- *Find the sky* (on by default for outdoor sources) adds a separate sky layer, so training paints the sky as a
  learned panorama instead of haze.

### Step 3: Training

- **Dataset processing → Process on** picks the device that prepares the dataset: this Mac or a Linux worker that
  processes. On a worker, video splitting, masks and camera solving run there and only what changed travels between
  the machines; steps the worker can't do run on the Mac. The choice is remembered for the next project.
- **Start training as soon as the dataset is ready** chains processing and training.
- Pick a quality preset and the **compute devices** for training (see [section 5](#5-training)).

When the project folder already holds results from an earlier run, **Reuse what the last run in this folder
finished** skips every stage whose settings and source files haven't changed.

While a project is being prepared, the progress screen shows overall progress, each stage with its duration, the
elapsed time, the estimated time left, and when the current step last reported. **Hide** keeps it running in the
background; the progress button in the toolbar brings it back. If processing fails, **Try Again** continues from the
stage that failed.

**Project layout:** `<project>/project.json` (source, settings and status), `<project>/dataset/` (posed images and
`dataset.json`), and `<project>/outputs/*.ply` (results).

### Home video (VHS, camcorder tapes)

Choose the **Home video** tile (or *Change… → Home video*) and pick the digitised tape. SplatWork reads the whole
file first (a minute or two for a long tape) and lists its **clips**. Clips are the shots between scene cuts. Each
clip has a picture and a verdict:

- **Good: the camera moves.** The camera travels, so there is depth to recover. The best of these is marked
  *best* and is ticked.
- **Only turns: little depth.** The camera pans from one spot. You can tick these, but they rarely make a splat on
  their own.
- **Camera hardly moves**, **Mostly blurred**, **Mostly moving things** (people close up, water, a car). These
  rarely give a usable scene.
- Very short stretches and blank (blue or black) screens aren't listed.

One clip is the default because it solves in minutes and reliably. To join clips, tick other clips **of the same
place**. A ticked clip that the solve can't place with the others is left out, and the log says why. For a different
place on the same tape, create another project and tick that clip.

What happens to the frames:
- **Picked frames.** Frames are picked where the camera has moved on and the picture is sharpest, not at a fixed
  rate. Repeated frames, which are common in tape transfers, are skipped.
- **Cleaned frames.** Black borders and the noisy band at the bottom of the picture are cropped. Stretched (non-square)
  pixels are corrected. Interlaced frames are rebuilt and tracking lines are repaired. Each frame is denoised with its
  neighbours, and each clip's colour cast is taken out. *Tape clean-up* turns the optional steps off.
- **Date stamps.** A burnt-in date or time is masked when it's found.
- **Cameras.** Each clip gets its own camera. A clip that zooms gets one camera per zoom setting.
- **Training.** Training uses colour without view-dependent detail (SH degree 1) and ignores moving people and pets,
  unless you changed the training settings yourself.

What to expect: tape holds about 768 × 576 pixels of real detail, so splats from it are soft. Shots that walk
through a place give the best results. Shots that only pan, zoom or follow a moving subject give little or no depth.

### Capture tips

- Move slowly and keep plenty of overlap. Walk around the subject at two or three heights.
- Avoid motion blur: good light, a faster shutter, no zooming.
- Moving people and changing light cause floaters. Use masks, or capture when it's quiet.
- With a LiDAR iPhone, use SplatWork Capture: you get poses and depth (no camera solving needed), the scene is in
  metres, and depth supervision cleans up walls and floors.

### iPhone captures

1. On the Mac, open **iPhone Inbox** and make sure **Receive from iPhone** is on. Note the pairing code.
2. On the iPhone, record in SplatWork Capture, then choose **Library → Send to Mac**. Enter the code if asked. If the
   Mac doesn't appear, enter the manual address shown in the Inbox.
3. In the Inbox, click **Create splat…** next to the capture.

## 4. Projects, history and resuming

The **Projects** list in the sidebar remembers every project you've created, processed, trained or opened, including
results of training. Each row shows where the project stands: *Processing 45 %*, *Processing interrupted at camera
positions*, *Ready to train*, *Training stopped*, *Trained*. Click a project to open its page.

**The project page** shows:
- **A status card** with the next thing to do: *Resume Processing*, *Resume Training*, *Train…*, *Open Latest
  Result*, or *Show Live View* while it runs. When processing can resume, you can choose the device that does it.
- **Processing**: each stage (frames, masks, camera positions, dataset) with what it produced and when.
  **Redo from Here** runs that stage and every later one again; earlier stages are kept. **Change Settings…** opens
  the project's settings; stages whose settings you don't change are reused.
- **Training**: the last run and every result in `outputs/`, each with **Open**.

**What survives an interruption.** Quitting the app, a crash, a restart, pressing Stop or a failure never costs more
than the step that was running:
- **Processing** saves a checkpoint after each stage (in `dataset/.prep/`), fingerprinted by the source files and the
  settings of that stage and all earlier ones. Running the project again skips every stage that still matches.
- **Training on this Mac** saves a checkpoint every 10 minutes and when you press Stop. *Resume from Step N*
  continues from it.
- **Training in blocks** keeps every finished block (in `outputs/.blocks/`). Resuming splits the scene the same way
  and trains only the blocks that weren't finished.
- **Training on a worker** keeps running on the PC while the Mac sleeps, loses the network or quits. Reopening the
  project re-attaches to the job, or downloads the result if it has finished.

The start page also offers projects that stopped partway (*Pick up where you left off*). Right-click a project in the
sidebar to show it in Finder or remove it from the list; the files stay where they are.

## 5. Training

### Presets

| Preset | Steps | Training size | Splat budget | Rough time (M4 Pro, ~200 images) |
|---|---|---|---|---|
| Quick preview | 5,000 | 1000 px | 300 k | 2–4 min |
| Balanced (default) | 15,000 | 1400 px | 800 k | 10–15 min |
| High quality | 30,000 | 1600 px | 1.5 M | 25–40 min |
| Maximum | 40,000 | 2400 px | 3 M | 1 h or more; needs a lot of memory |

Start with *Quick preview* to check the camera solve and coverage. In Advanced mode, changing any value switches the
preset to Custom; *Reset to preset* and *Copy as JSON* are in the Train tab.

Defaults worth knowing (all in the settings reference):
- **Per-photo colour correction** is on. It turns itself off when every photo has the same recorded exposure
  (SplatWork Capture locks exposure). Exports bake in the average correction.
- **Ignore moving objects** is off. Turn it on for scenes with people walking through; it is skipped when the photos
  are too far apart to tell movers from scenery.
- **Long-axis splitting**, the **needle-splat penalty** and the **big-splat penalty** are on. They stop large scenes
  collapsing and reduce streaks and fog in front of the cameras.

### Where it runs

Tick the **compute devices** that may train this project: this Mac, Linux workers running SplatWork for Linux, and other
Macs with *Compute → Act as a worker* turned on.
- **One device**: the whole scene trains there. A worker streams its live preview back.
- **Several devices**: scenes of 400 images or more are split into spatial blocks that train in parallel and are
  merged into one splat. Each device gets blocks it has the memory for, in proportion to its speed. Smaller scenes
  train whole on the device expected to finish first.
- **Too big for one device**: the scene is trained there in blocks that fit, one after another, and merged.

After you run the [benchmark](#10-compute-devices-and-the-benchmark), the line under the device list shows the
estimated time on each machine, including the upload, and marks the fastest.

### The dashboard

The Train tab shows the step, speed, elapsed time and time left, the splat count, and charts of training PSNR,
held-out PSNR and splat count. Block runs list every block with its device, stage, progress, speed and time left.

- **Pause / Resume** holds every device: this Mac's trainer stops between steps, and a worker saves its job's state
  and frees its GPU until you resume.
- **Stop** ends the run. On this Mac it keeps a checkpoint you can resume from (see [section 4](#4-projects-history-and-resuming)).
- **Save checkpoint…** writes the current splats to a file without stopping.
- **A failed block** waits instead of failing the run: **Retry** trains it again on the same device, or pick another
  device from the menu, or **Leave This Block Out** to finish without it (that part of the scene stays empty).

**Held-out PSNR**, measured on photos never trained on, is the honest quality number. Above 25 dB is good and above
30 dB is excellent. Training PSNR alone is misleading.

### Show Live View

The viewer follows the run as it trains. You can open other projects or splats meanwhile; training carries on, and
its previews don't replace what you're looking at. To get back to it, use any of:
- the **Training … %** button in the toolbar,
- **Show Live View** on the banner at the bottom of the viewer, in the Train tab, or on the project's page,
- or open the training project again.

If the run finishes while you're looking at something else, the banner offers **Show Result**.

### Watching a worker's job

In **Compute**, click **Watch** next to a running training job on a worker. The viewer
shows the job's live preview and cameras with the full dashboard; press **F** to look from the job's central camera.
Opening another project stops watching.

### Changing light: local colour correction

**Local colour correction** (Advanced, on by default, on the Mac and on workers) gives each photo its own coarse
grid of colour adjustments instead of one correction for the whole photo, so shadows and sunlight that moved
during the capture don't smear. On a 1,258-photo drone flight it raised held-out PSNR by about 0.8 dB (16.37 →
17.18 dB on the Mac, 17.16 → 17.91 dB on a worker); on an indoor scan it made no difference. Turn it off to save
some training time on captures with steady light.

## 6. Viewing

Open or drop a `.ply`, `.spz` or `.splat` file.

| Input | Action |
|---|---|
| Drag / two-finger drag | Orbit |
| ⇧-drag, right-drag | Pan |
| Scroll, pinch | Zoom |
| Double-click | Set the orbit centre on the splat under the cursor |
| Tab | Switch Orbit / Fly. In Fly, use W A S D Q E (hold ⇧ to go faster) |
| F | Frame. With a project open, look from the most central capture camera; otherwise frame the whole splat |
| R | Reset the view |
| 1–6 | Display mode: Colour, Depth, Centres, Ellipses, Opacity, Coverage |
| ? | Shortcut list |

Speeds, smoothing and what a two-finger trackpad drag does are under **View → Navigation**. Pan and fly speed follow
the distance to the surface you're looking at.

**Display modes:**
- **Depth** shows floaters as sudden colour jumps.
- **Opacity** helps tune *Delete below opacity*.
- **Coverage** (key 6) colours each part of the scene by how many photos saw it and how closely. Red means two or
  fewer views, or much less detail than typical. The View tab shows a legend and a summary that separates "seen
  poorly" from "seen by no photo" (floaters).

**Training cameras.** *Show training cameras* draws the solved cameras. They should follow the path you walked.
Double-click a camera in the Cameras list to look through it. Opening a splat from a project's `outputs/` folder
loads that project's cameras, and with a dataset open the viewer starts at a training camera, because splats look
best from where they were captured.

**Up direction.** Splat files don't record which way is up. Use the **Orientation** menu in the toolbar to flip the
scene, pick an axis, or rotate until level and choose **Level to current view**. The choice is remembered per file,
and Export can bake it in. Datasets made in SplatWork are levelled automatically (iPhone captures from ARKit gravity,
drone photos from GPS).

## 7. Editing

Use the **Edit** tab. Every operation can be undone (⌘Z / ⇧⌘Z), and *Restore all* brings back everything deleted.

- **Auto clean** selects splats that make no visible difference from any of the project's photos (they never add
  more than 2 % to any pixel). Review the orange selection, then press **Delete selected**.
  - **Include floaters** (on) also selects splats clearly visible in fewer than a quarter as many photos as a typical
    splat. Indoors it can catch close-up surfaces few photos saw, so check the selection.
  - In testing it removed 16 % of a bedroom scan's splats with no change in held-out PSNR, and every floater more
    than 25 m above the ground in a drone survey.
  - It needs the project open and a finished splat.
- **Crop box**: fit the box with the sliders or *Fit to scene*. The outside is hidden while you adjust it; *Apply
  crop* deletes it.
- **Brush**: choose Select in the toolbar and drag. Hold ⌥ to deselect. ⌘I inverts and ⌘D clears the selection;
  ⌫ deletes it. The brush selects everything along the view ray, so turn the scene to pick only front-most splats.
- **Delete below opacity**, **Delete larger than** and **Remove isolated splats** clean up haze, oversized blobs and
  stray specks.

## 8. Exporting

Export is ⌘E.

| Format | Notes |
|---|---|
| **PLY** | Lossless; read by every tool (about 248 bytes per splat at SH degree 3). |
| **SPZ** | About 10× smaller (about 25 bytes per splat) with very little visible loss. |
| **.splat** | For web viewers. No view-dependent colour. |

Options:
- *Export SH degree* can drop colour bands (SH 0 makes a PLY about 4× smaller).
- *Export only kept splats* applies your edits and crop.
- *Bake orientation* writes a standard −Y-up file that opens upright in other tools.
- You can also save 4K screenshots and render turntable videos (720p, 1080p or 4K).

A learned sky is saved next to the splat as `<name>.sky.png` and `<name>.sky.json`.

## 9. Moving objects (people, pets, cars)

Anything that moves between frames turns into ghostly floaters. There are three tools, from automatic to precise:
1. **Ignore moving objects** (Train tab, off by default). Solid regions that disagree strongly with every other view
   are left out once the scene has formed. The dashboard shows how much is being ignored.
2. **Masks when creating the project** (see [Step 2](#step-2-cameras--masks)).
3. **Edit masks…** (Train tab). Review every frame, click an object once to outline it with SAM 2, then *Track
   through all frames*, or paint areas out and back by hand. Save and train again.

## 10. Compute devices and the benchmark

The **Compute** page lists this Mac and every Linux worker, with its GPU, capabilities and jobs. A worker without the
processing models (camera solving, moving-object masks) offers to install them remotely.

**Running the benchmark:**
1. Open **Compute** and make sure the workers you want to measure are online.
2. Click **Run benchmark**. Untick *Include this Mac* to measure only the workers.
3. Each machine trains the same synthetic scene (about 1 M splats, 1600 × 1200 photos, 2,000 steps). Workers also
   receive the scene and send back the result, which measures the transfer speed. It takes a few minutes per
   machine.

The results replace the built-in guess of each GPU's speed, set the time estimates next to the device list, and decide
how blocks are shared. Run it again after changing hardware, drivers or a worker's runtime, and don't use the Mac's
GPU for anything else while it runs.

**Other Macs as workers.** On another Mac, turn on *Compute → Act as a worker*. It then appears in the device lists
of your other Macs, protected by an access token you paste once when adding it.

## 11. Setting up a Linux worker

Full details are in [linux/README.md](../linux/README.md).

1. **Install SplatWork for Linux** on the PC from the project's Flatpak repository, so updates arrive through
   `flatpak update`, GNOME Software or KDE Discover:
   ```bash
   flatpak install --user https://benevision.github.io/SplatWork/flatpak/splatworker.flatpakref
   ```
2. **Open SplatWork for Linux.** The setup assistant (**Set Up…**) finds your NVIDIA driver, lists what it will download
   (about 2.7 GB, needing about 6 GB of disk) and installs a private Python, PyTorch and gsplat inside the app's own
   folder. You can cancel and retry; it resumes from the step that failed and never breaks a working setup.
   - Driver 570 or newer (RTX 20 to RTX 50 series): PyTorch 2.7 + CUDA 12.8 + gsplat 1.6.
   - Driver 520–569: PyTorch 2.4 + gsplat 1.5.3.
   - No NVIDIA GPU: a processing-only setup (camera solving and masks on the CPU).
3. **Turn the server on** with the big switch. To run without a window, use
   `flatpak run com.example.SplatWorker --headless`.
4. **On the Mac**, the worker appears under **Compute** by itself. If it doesn't, add it by the address shown on the
   worker's Status page, and paste its token if it asks (tokens are kept in the Keychain).
5. **Firewall:** open TCP 7878 and UDP 5353 (mDNS) to your local network only.
6. **Processing models:** install them on the worker from the Mac's Compute page to run camera solving, learned masks
   and video splitting there.

SplatWork for Linux updates itself when no job is running, keeping its settings, token and GPU runtime.

## 12. Troubleshooting

- **The PC says "NVIDIA driver libraries are missing in the sandbox"** (often after a driver update): Flatpak needs an
  extension that matches your driver version exactly. The message shows the command, for example:
  ```bash
  flatpak install flathub org.freedesktop.Platform.GL.nvidia-620-70-06//1.4
  ```
  then restart SplatWork for Linux. Reboot first if you updated the driver without rebooting.
- **The Mac doesn't find the worker:** check the firewall (TCP 7878, UDP 5353), make sure both are on the same
  network, or add the worker by its address. *Needs token* means the token is missing or wrong.
- **A feature says it needs a newer worker**, or is quietly done on the Mac instead: update SplatWork for Linux with
  `flatpak update`. Settings, token and GPU runtime are kept across updates.
- **Camera solving placed only a few photos:** use more overlapping images, try another camera solver, or try without
  masks (masks can hide most of each photo).
- **The scene is upside down or tilted:** use Orientation → Level to current view, or pick an axis.
- **Training looks hazy or blotchy across photos:** leave per-photo and local colour correction on for video, drone
  and handheld photos.
- **360° video uses a lot of disk:** an 8K 360° clip decodes to about 10,000 frames, each cut into several views (one
  test produced 4.4 GB of frames). Leave room on both machines. A worker keeps uploaded datasets until they are
  deleted, and limits uploads to 20 GB (40 GB once unpacked) by default; change that in the worker's Settings.
- **The benchmark looks slow on the Mac:** close other apps that use the GPU and run it again.
- **Something else:** [open an issue](https://github.com/Benevision/SplatWork/issues) with the project's `prep.log`
  and what you expected to happen.
