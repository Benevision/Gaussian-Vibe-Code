# SplatWork — user guide

Every control in the app has an ⓘ button with a full explanation. The same text is collected in [SETTINGS.md](SETTINGS.md) and in **Help → Settings Reference** (⇧⌘/, searchable). This guide covers the workflow; look up individual settings there.

## 1. Install and run

**Mac app** (macOS 15 or later, Apple silicon):

```bash
cd mac && ./scripts/build_app.sh && open "build/Splat Studio.app"
```

**Optional extras:**
- **SplatWork Capture** on an iPhone (LiDAR models give the best results): build it from `ios/` (see `ios/README.md`).
- **SplatWork Worker** on a Linux PC with an NVIDIA GPU, for faster training and processing: see [section 10](#10-setting-up-the-linux-worker).
- **Models pack**: learned feature matchers and segmentation models, installed from **Models & Benchmarks**.

**Easy and Advanced.** The segmented control in the toolbar switches modes at any time. **Easy** shows quality presets and the few choices that matter. **Advanced** shows every parameter. Switching keeps your values.

## 2. The workspace

The sidebar has these pages:

| Page | What it is for |
|---|---|
| **Viewer** | The 3D view plus the inspector (View, Edit, Train, Export tabs). Recent files are listed below it. |
| **iPhone Inbox** | Captures sent from SplatWork Capture. Shows the receive switch, the pairing code, the manual address, and the projects and splats made from each capture. |
| **Compute** | This Mac, Linux workers (found by Bonjour or added by address), their running and recent jobs, the **Benchmark**, and manual worker setup. The green dot means a worker is online. |
| **Models & Benchmarks** | Install the Models pack and compare models (registration rate, pose accuracy, reprojection error, time, memory, licence). Some models (SuperPoint, MASt3R, the VGGT-1B weights) have non-commercial licences. |
| **Settings Reference** | Every setting with its default, range and explanation. |

⇧⌘H (Home) returns to the start page and unloads the scene to free memory.

## 3. Creating a project

Choose **New** (⌘N), or drop a folder, video or capture on the window. The guided flow has three steps.

### Step 1: Source

| Source | Notes |
|---|---|
| **Photos folder** | Blurry and duplicate photos can be rejected. |
| **Video** | Frames per second or a target frame count, and *Pick sharpest frame* (the sharpest frame in each window). |
| **360° video / 360° photos** | Each equirectangular frame is cut into flat views (8 by default, 90° each). *Drop the floor view* skips the straight-down view showing the tripod or you. |
| **Stereo video** | Side-by-side, top/bottom, two files, Apple spatial (MV-HEVC) or two video tracks. The baseline gives the scene metric scale; optional stereo depth. |
| **iPhone / headset capture** | Poses (and LiDAR depth) come with the capture. Options: use depth for supervision, build initial points from depth, drop frames with limited tracking. |
| **COLMAP project / Nerfstudio transforms.json** | Imports existing poses. |

Advanced options include the stored image size (default 3200 px) and *Hold out every Nth image* (25 in Easy, 8 in Advanced), which keeps some photos aside to measure quality honestly.

When a worker is in use, **Split videos on the worker** (on by default) streams the video to the PC, which picks frames and cuts 360° views there. The frames stay on the worker for the next steps (masks, camera solving, training) and download to the Mac in the background. If the worker can't do it (it needs worker 1.3 or newer with processing models), the video is split on the Mac.

### Step 2: Cameras & masks

**Camera solver:**
- **Automatic** (default):
  - captures, COLMAP and Nerfstudio: the poses in the source;
  - **360° video**: learned SfM (SuperPoint + LightGlue, COLMAP/GLOMAP), on the worker when one is in use. Object Capture placed only 1,041 of 9,824 views of a 360° walk-through, so it is not used for 360°;
  - everything else: Apple Object Capture.
- **Apple Object Capture**: built in, no install. Good for objects and rooms. Advanced: high feature sensitivity for low-texture scenes.
- **COLMAP/GLOMAP + learned features**: needs the Models pack or a Linux worker. Advanced mode lets you choose the feature model, pair selection, mapper and lens model.
- **Use poses from source**: shown when the source has poses.

**Refine capture poses** (Advanced, on by default) polishes iPhone poses with SfM, keeping their scale and gravity. On a 649-frame bedroom scan it took 3.6 min instead of 28 for a full re-solve and gave +2.1 dB held-out PSNR. If it can't run, the capture poses are used as recorded.

**Drone photos with GPS:** learned camera solving checks the solved cameras against EXIF GPS. If they agree, the scene is put in metres with Z up and stray cameras are dropped. Otherwise you get a warning.

**Masks** (pixels excluded from training):
- *Remove people & pets — quick* (Apple Vision, built in).
- *Remove moving things — best* (Models pack: open-vocabulary detection plus SAM 2 tracking). You can edit the list of objects to remove.
- *Keep main subject*, *Remove people*, *Remove sky*, *Isolate one object (SAM 2)*. *Invert mask* swaps kept and ignored areas.

Learned masking can run on the worker.

### Step 3: Training

Pick a preset and where to run it (see [section 4](#4-training)). *Start training as soon as the dataset is ready* chains the two.

**Project layout:** `<project>/dataset/` (posed images, `dataset.json`) and `<project>/outputs/*.ply`. Retrain the same dataset with different settings from the **Train** tab.

While a project is being prepared, the progress screen shows overall progress, each stage with its duration, elapsed time, the estimated time left, and when the current step last reported.

### Capture tips

- Move slowly and keep plenty of overlap. Walk around the subject at two or three heights.
- Avoid motion blur (good light, faster shutter). Don't zoom.
- Moving people and changing light cause floaters. Use masks, or capture when it's quiet.
- iPhone with LiDAR: use SplatWork Capture. You get poses and depth (no SfM needed), the scene is in metres, and depth supervision cleans up walls and floors.

### iPhone captures

1. On the Mac, open **iPhone Inbox** and make sure **Receive from iPhone** is on. Note the pairing code.
2. On the iPhone, record in SplatWork Capture, then **Library → Send to Mac**. Enter the code if asked. If the Mac doesn't appear, enter the manual address shown in the Inbox.
3. In the Inbox, click **Create splat…** next to the capture.

## 4. Training

### Presets

| Preset | Steps | Training size | Splat budget | Rough time (M4 Pro, ~200 images) |
|---|---|---|---|---|
| Quick preview | 5,000 | 1000 px | 300 k | 2–4 min |
| Balanced (default) | 15,000 | 1400 px | 800 k | 10–15 min |
| High quality | 30,000 | 1600 px | 1.5 M | 25–40 min |
| Maximum | 40,000 | 2400 px | 3 M | 1 h or more, needs a lot of RAM |

Start with Quick preview to check the camera solve and coverage. In Advanced mode, changing any value switches the preset to Custom. *Reset to preset* and *Copy as JSON* are in the Train tab.

Defaults that matter (all in the Settings Reference):
- **Per-photo colour correction** is on. It switches itself off when every photo has the same recorded exposure (SplatWork Capture locks exposure). Exports bake in the average correction.
- **Ignore moving objects** is off by default. Turn it on for scenes with people walking through. It is skipped automatically when photos are too far apart to tell movers from scenery.
- **Long-axis splitting**, the **needle-splat penalty** and the **big-splat penalty** are on. They stop large scenes collapsing and reduce streaks and fog in front of cameras.

### Where it runs

The **Compute device** picker has three choices:
- **This Mac**: the Metal trainer, no setup.
- **Linux worker**: SplatWork Worker on your PC, with the live preview streamed back.
- **Both**: each stage goes to the faster machine. Scenes with 400 images or more (Advanced: *Split into blocks above*) are split into spatial blocks that train on both machines and are merged. Smaller scenes train whole on the machine estimated to finish first.

After you run the Benchmark (see [section 9](#9-compute-and-the-benchmark)), the line under the picker shows the estimated time on each machine, including the upload, and marks the fastest.

### The dashboard

The dashboard shows step, speed, elapsed time and ETA, splat count, and train and holdout PSNR charts. The viewer shows the model live as it trains. You can pause, resume, save a checkpoint, or stop.

**Holdout PSNR** (measured on photos never trained on) is the honest quality number. Above 25 dB is good and above 30 dB is excellent. Training PSNR alone is misleading.

### Remote jobs survive the Mac going offline

Once the worker accepts a job, the Mac saves it next to the output (`<output>.remote-job.json`). If the Mac sleeps, loses the network or quits:
- the job keeps running on the PC;
- the Mac retries for up to 12 hours while it is waiting;
- reopening the project re-attaches to the job (progress and live preview), or downloads the result if it has finished.

**Compute** also lists the five latest finished training jobs on each worker, with **Open result**.

### Watching a PC job

1. In **Compute**, click **Watch** next to a running training job. The Viewer opens.
2. You get the full training dashboard (speed, elapsed/ETA, PSNR and splat charts, filled in from the job log), the live preview and that job's cameras.
3. Press **F** to look from the job's central camera.

Opening another project or starting local training stops watching.

### Long drone flights: local colour correction

**Local colour correction (PC)** (Advanced, off by default; used only on a Linux worker) gives each photo its own coarse grid of colour adjustments. Use it when the sun moved during the capture. On a 96-minute, 1,258-photo flight it raised held-out PSNR from 17.16 to 17.91 dB and removed smeared shadows, but training took about 50 % longer. While it is on, it replaces per-photo colour correction.

## 5. Viewing

Open or drop `.ply`, `.spz` or `.splat`.

| Input | Action |
|---|---|
| Drag / two-finger drag | Orbit (drag right and the scene turns right) |
| ⇧-drag, right-drag | Pan |
| Scroll, pinch | Zoom |
| Double-click | Set the orbit centre on the splat under the cursor |
| Tab | Switch Orbit / Fly. In Fly, use W A S D Q E (⇧ = faster) |
| F | Frame. With a project open, look from the most central capture camera; otherwise frame the whole splat |
| R | Reset view |
| 1–6 | Display mode: Colour, Depth, Centres, Ellipses, Opacity, Coverage |
| ? | Shortcut list |

Speeds, smoothing and what a two-finger trackpad drag does are under **View → Navigation**. Pan and fly speed follow the distance to the surface you are looking at.

**Display modes:**
- **Depth** shows floaters as colour jumps.
- **Opacity** helps tune *Delete below opacity*.
- **Coverage** (key 6) colours each part of the scene by how many photos saw it and how closely. Red means 2 or fewer views, or much less detail than typical. The View tab shows a legend and a summary that separates "seen poorly" from "seen by no photo" (floaters).

**Training cameras.** *Show training cameras* draws the solved cameras as frustums. They should follow the path you walked. Double-click a camera in the Cameras list to look through it. Opening a splat from a project's `outputs/` folder loads that project's cameras. When a dataset is open, the viewer starts at a training camera, because splats look best from where they were captured.

**Up direction and level.** Splat files don't record which way is up. Use the toolbar's **Orientation** menu (or View → Orientation) to flip upside down, pick an axis, or rotate until level and choose **Level to current view**. The choice is remembered per file, and Export can bake it in. Datasets built in SplatWork are levelled automatically: iPhone captures use ARKit gravity, drone sets with GPS use the GPS vertical, and PC-trained results record their up direction in the PLY.

## 6. Editing

Use the **Edit** tab. Every operation can be undone (⌘Z / ⇧⌘Z), and *Restore all* brings back everything deleted.

- **Auto clean** selects splats that make no visible difference from any of the project's photos (they never add more than 2 % to any pixel). Review the orange selection, then press **Delete selected**.
  - **Include floaters** (on) also selects splats clearly visible in fewer than a quarter as many photos as a typical splat.
  - Results in testing: on a bedroom scan, 16 % of splats removed with no change in held-out PSNR. On a drone survey, every floater more than 25 m above the ground was removed.
  - Indoors, Include floaters can catch close-up surfaces that few photos saw, so check the selection.
  - Needs the project open and a finished splat.
- **Crop box**: fit the box with the sliders or *Fit to scene*. The outside is hidden while you adjust it, and *Apply crop* deletes it.
- **Brush**: select the Brush (or Select in the toolbar) and drag. Hold ⌥ to deselect. ⌘I inverts and ⌘D clears the selection; ⌫ deletes it. The brush selects everything along the view ray, so rotate the scene to pick only front-most splats.
- **Delete below opacity**, **Delete larger than** and **Remove isolated splats** clean up haze, oversized blobs and stray specks.

## 7. Export

Export is ⌘E.

| Format | Notes |
|---|---|
| **PLY** | Lossless, read by every tool (≈248 bytes/splat at SH 3). |
| **SPZ** | About 10× smaller (≈25 bytes/splat), very little visible loss. |
| **.splat** | For web viewers. No view-dependent colour. |

Export options:
- *Export SH degree* can drop colour bands (SH 0 makes a PLY about 4× smaller).
- *Export only kept splats* applies your edits and crop.
- *Bake orientation* writes a standard −Y-up file that opens upright elsewhere.
- You can also save 4K screenshots and render turntable videos (720p, 1080p or 4K).

## 8. Moving objects (people, pets, cars)

Anything that moves between frames turns into ghostly floaters. There are three tools, from automatic to precise:
1. **Ignore moving objects** (Train tab, off by default). Solid regions that disagree strongly with every other view are left out once the scene has formed. The dashboard shows how much is being ignored.
2. **Masks when creating the project** (see Step 2).
3. **Edit masks…** (Train tab). Review every frame and click an object once to outline it with SAM 2. Then *Track through all frames*, or paint areas out and back by hand. Save and retrain.

## 9. Compute and the Benchmark

The **Compute** page lists this Mac and every Linux worker, with its GPU, capabilities and jobs. A worker that lacks the processing models (camera solving, moving-object masks) offers to install them remotely.

**Benchmark:**
1. Open **Compute** and make sure the workers you want to measure are online.
2. Click **Run benchmark**. Untick *Include this Mac* to measure only the workers.
3. Each machine trains the same synthetic scene (about 1 M splats, 1600 × 1200 photos, 2,000 steps). Each worker also gets the scene uploaded and its result downloaded, which measures transfer speed. It takes a few minutes per machine.

The table shows steps per second, speed relative to the slowest machine, upload and download speed, and when each machine was measured. These results:
- replace the built-in guess of each GPU's speed;
- set the time estimates next to the device picker;
- decide which machine trains small scenes in Both mode, and how blocks are shared.

Run it again after changing hardware, drivers or the worker's runtime. Don't use the Mac's GPU for anything else while it runs.

## 10. Setting up the Linux worker

Full details: [`linux/README.md`](../linux/README.md).

1. **Install the Flatpak** on the PC. It is best to install from the published repository so updates arrive through `flatpak update` or GNOME Software / KDE Discover:
   ```bash
   flatpak install --user https://benevision.github.io/Gaussian-Vibe-Code/flatpak/splatworker.flatpakref
   ```
   You can instead build it from `linux/flatpak` with `bash build.sh`. Copy the project to a folder in your home directory first, because USB drives are often mounted noexec.
2. **Open SplatWork Worker.** The setup assistant (**Set Up…**) finds your NVIDIA driver and lists the downloads (about 2.7 GB, needing about 6 GB of disk). It then installs a private Python, PyTorch and gsplat. You can cancel and retry, and it resumes from the failed step. A failed setup leaves the working runtime untouched.
   - Driver 570+ (RTX 20 to RTX 50): PyTorch 2.7 + CUDA 12.8 + **gsplat 1.6** (a development build).
   - Driver 520–569: PyTorch 2.4 + gsplat 1.5.3.
   - No NVIDIA GPU: a processing-only runtime.
3. **gsplat 1.6 wheel.** Set Up uses a wheel published next to the Flatpak. To build one yourself on the PC (minutes for your own GPU), run:
   ```bash
   cd linux/flatpak && ./build-gsplat-wheel.sh
   ```
   This needs podman. Set Up prefers a local wheel when there is one. Run **Set Up…** again to switch runtimes.
4. **Turn the server on** with the big switch. To run without the window, use `flatpak run com.example.SplatWorker --headless`.
5. **On the Mac**, the worker appears under **Compute** automatically. If it doesn't, add it by the address shown on the worker's Status page. Paste the token if it asks (tokens are kept in the Keychain).
6. **Firewall:** open TCP 7878 and UDP 5353 (mDNS) to your LAN only.

Install the processing models on the worker (from the Mac's Compute page) to run camera solving, learned masks and video splitting there.

## 11. Troubleshooting

- **360° video uses a lot of disk.** An 8K 360° clip decodes to about 10,000 frames, and each frame becomes several views. One test produced 4.4 GB of frames. Leave room on both machines for the video, the frames and the outputs:
  - With *Split videos on the worker*, the frames are made on the PC and kept there between steps. The Mac still downloads a copy in the background, and that download resumes when you reopen the project.
  - The worker keeps uploaded datasets until they are deleted, and caps uploads at 20 GB (40 GB once extracted) by default. Change the cap in the worker's Settings (maximum upload size) or with `--max-upload-gb` / `--max-extracted-gb`.
- **A feature says it needs a newer worker**, or is quietly done on the Mac instead: update SplatWork Worker (`flatpak update`; the worker app also shows a banner when a new version is out). Examples: video splitting needs 1.3, and keeping frames on the worker between steps needs 1.3.8. Settings, token and GPU runtime are kept across updates.
- **Worker setup fails or CUDA is not available after a driver update:** run `flatpak update` so the Flatpak NVIDIA extension matches the host driver exactly, then restart the app. See the Flatpak troubleshooting section in `linux/README.md`.
- **The Mac doesn't find the worker:** check the firewall (TCP 7878, UDP 5353), use the same subnet, or add the worker by IP. A "Needs token" status means the token is missing or wrong.
- **Camera solving placed only a few photos:** use more overlapping images, try another camera solver, or try without masks (masks can hide most of each photo).
- **The scene is upside down or tilted:** use Orientation → Level to current view, or pick an axis.
- **Training looks hazy or blotchy across photos:** leave per-photo colour correction on for video, drone and handheld photos. For long outdoor flights on the PC, try Local colour correction.
- **The Benchmark looks slow on the Mac:** close other apps using the GPU and run it again.
