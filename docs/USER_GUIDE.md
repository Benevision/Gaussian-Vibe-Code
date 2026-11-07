# Splat Studio — user guide

Every control in the app has an ⓘ button with a full explanation; the same text is collected in [SETTINGS.md](SETTINGS.md) and in **Help → Settings Reference** (searchable).

## 1. Easy and Advanced modes
The segmented control in the toolbar switches modes at any time. **Easy** shows quality presets and the handful of choices that matter. **Advanced** reveals every parameter (densification, learning rates, regularisers, SfM details, compute placement…). Nothing is ever locked away: switching modes keeps your values.

## 2. Viewing splats
Open or drop `.ply`, `.spz` or `.splat`. Navigation:

| Input | Action |
|---|---|
| Drag / two-finger drag | Orbit |
| ⇧-drag, right-drag | Pan |
| Scroll, pinch | Zoom |
| Double-click | Set orbit centre on the splat under the cursor |
| Tab | Toggle Orbit / Fly; in Fly use W A S D Q E (⇧ = faster) |
| F | Frame scene · R reset view · 1–5 display modes · ? shortcuts |
| ⇧⌘H / 🏠 | Home: back to the start page, unloading the scene to free memory |

Dragging works like grabbing the scene (drag right → it turns right). Speeds, smoothing and what a two-finger trackpad drag does (orbit or pan) are under **View → Navigation**.

**Upside down or sideways?** Splat files don't record which way is up. Use the toolbar's Orientation menu (or View → Orientation): flip, pick an axis, or rotate until level and choose *Level to view*. The choice is remembered per file, and Export can bake it in. Datasets made in Splat Studio are levelled automatically (iPhone captures use the gravity measured by ARKit).

Splats look best from where they were captured. When a dataset is open the viewer starts at a training camera, and you can double-click any camera in **View → Training cameras** to look through it.

## 3. Creating a splat
**New** (⌘N), or drop a folder / video / capture. The guided flow has three steps:

1. **Source:** photos, video, 360°, stereo, iPhone capture, COLMAP or Nerfstudio. The options depend on the source: frame rate or count, sharpest-frame picking, the 360° view rig, stereo layout and baseline, LiDAR depth use, and holdout.
2. **Cameras & masks:**
   - *Automatic* uses capture poses when present, otherwise Apple Object Capture.
   - *COLMAP/GLOMAP + learned features* needs the Models pack or a Linux worker.
   - Masks remove people, sky, or everything except the subject.
3. **Training:** pick a preset and where to run it (This Mac / Linux worker / Both).

Projects are saved as `<project>/dataset/` (posed images) and `<project>/outputs/*.ply`. You can retrain the same dataset with different settings from the **Train** panel.

### Capture tips
- Move slowly and keep lots of overlap. Walk around the subject at two or three heights.
- Avoid motion blur (good light, faster shutter). Don't zoom.
- Moving people and changing light cause floaters. Use masks or capture when it's quiet.
- iPhone with LiDAR: use Splat Capture. Poses and depth are recorded, so no SfM is needed, the scene is in metres, and depth supervision cleans up walls and floors.

### Moving objects (people, pets, cars)
Anything that moves between frames turns into ghostly floaters. Three tools, from automatic to precise:
1. **Ignore moving objects** (training, on by default): solid regions that disagree strongly with every other view are left out of training once the scene has formed. No setup; the dashboard shows how much is being ignored.
2. **Masks when creating the project** (Cameras & masks step): *Remove people & pets — quick* (Apple Vision, built in) or *Remove moving things — best* (Models pack: open-vocabulary detection — edit the object list — plus SAM 2 tracking through the sequence).
3. **Edit masks…** (Train panel): review every frame, click an object once to outline it (SAM 2, instant after the first click), *Track through all frames*, or paint areas out / back by hand. Save and retrain.

## 4. Training dashboard
The dashboard shows step, speed, ETA, splat count, and train/holdout PSNR charts. The viewer shows the model live as it trains. You can pause, resume, save a checkpoint, or stop. Holdout PSNR (on images never trained on) is the honest quality number: above 25 dB is good and above 30 dB is excellent. While a project is being prepared the progress screen shows overall progress, each stage with its duration, elapsed time, an estimate of the time left and a heartbeat showing when the current step last reported.

## 5. Editing
The **Edit** panel. Every operation is undoable (⌘Z / ⇧⌘Z):
- **Selection brush:** choose the Select tool, drag to select, ⌥ to deselect, ⌫ to delete. You can also invert or clear the selection.
- **Crop box:** a live preview, then *Apply crop*.
- **Clean up:** delete below an opacity, delete over-large splats, remove isolated outliers.

## 6. Export
- **PLY:** lossless, universal.
- **SPZ:** about 12× smaller, roughly −0.3 dB.
- **.splat:** web viewers, no view-dependent colour.

Exports can drop SH degrees to save space. *Export only kept splats* applies your edits. You can also save 4K screenshots and turntable MP4s.

## 7. iPhone captures
Install Splat Capture (see `ios/README.md`), record, then **Library → Send to Mac**. The Mac must have **iPhone Inbox → Receive from iPhone** on, and it shows a pairing code. Captures appear in the Inbox; click *Create splat…*.

## 8. Linux GPU worker
Install the **Splat Worker Flatpak** on the PC (`linux/README.md`) — its first-run assistant downloads the GPU runtime that matches your NVIDIA driver — then flip the big switch, or run `splatworker serve --token …` headless. It appears under **Compute** automatically, or you can add it by IP address. Paste the token if it asks. Choose *Linux worker* or *Both* when training. *Both* splits large scenes (400+ images) into blocks trained in parallel on both machines.

## 9. Models pack & benchmarks
**Models & Benchmarks** installs an optional private Python environment with learned matchers (LightGlue family, LoFTR, MASt3R, VGGT) and segmentation (SAM 2, sky, people). The table shows each model's benchmark results on this machine: registration rate, pose accuracy, reprojection error, time, memory and licence. Select a model to see its strengths and weaknesses. Note: some models (SuperPoint, MASt3R, the VGGT-1B weights) have non-commercial licences.
