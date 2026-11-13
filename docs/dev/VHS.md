# VHS / old home video → splats

Status: **study done, plan below, not built yet** (2026-11-13). Roadmap item from 2026-11-06: "a restoration pipeline
for low-quality footage such as VHS home videos".

The goal is to make splats from home-video tapes that someone has already digitised to a video file. Real-world input
looks like the two sample files below: a capture card or a HandBrake re-encode, sometimes upscaled, sometimes
frame-rate converted, with tape artefacts baked in. The footage itself never goes in the repo; it stays in the
git-ignored `Sample datasets/` folder.

## 1. What the footage contains

Measured with ffprobe/ffmpeg (`idet`, `scene`) and a small OpenCV script that reads every 5th frame: per-row and
per-column means, Laplacian sharpness, a downscale → upscale PSNR test for real detail, and the frame-to-frame
difference.

### Main test clip: "lots of camera movement, multiple clips" (.mp4, 547 MB)

| Property | Finding | What it means for the pipeline |
|---|---|---|
| Container / codec | HEVC 1890×1068, 9.8 Mb/s, re-encoded by HandBrake 1.10 | The tape has already been through one capture plus one lossy re-encode. Expect blocking on top of tape noise. |
| Pixel aspect | **SAR 712:945** → display 4:3 (1424×1068 square pixels) | Pixels are not square. **Neither frame extractor applies SAR today** (`VideoFrameReader` in the Mac app, `frames.py` in splatprep), so the plain video path writes frames stretched 1.33× horizontally. COLMAP/GLOMAP models assume `fx = fy`, so this damages the solve on its own. Fix for every video, not just VHS. |
| Frame rate | 20 fps, 8961 frames, 7 min 28 s, no duplicate frames | Unusual (PAL is 25 fps); probably a frame-rate conversion on capture. Every frame is a unique image. |
| Interlacing | `idet`: 2413 / 2413 progressive, no combing | Already deinterlaced (or blended) before we got it. Deinterlacing has to be **detected per file**, not assumed. |
| Real detail | Downscale → upscale PSNR: 45–49 dB at 360 px wide, 39–41 dB at 288 lines, 45–47 dB at 576 lines | The 1890×1068 frame holds about **360–480 × 576** samples of real detail: tape luma bandwidth horizontally and PAL lines vertically. Working at ~768×576 (square pixels) loses nothing. Larger sizes only cost time and memory for the solver and trainer. |
| Borders | Black left border ~16 px and right border ~10 px (at 1890 wide); top is clean | Crop them. Detected from column means. |
| Head-switching band | The bottom ~24 rows: a dark smear 10–14 rows up, then black in the last 4 | Crop the bottom band (~2.5% of the height). Detected from row means plus per-row instability. |
| Tracking noise | Occasional single-line white dropout streaks (e.g. a ~360 px line at y≈258 in one frame) | Rare and short. A per-frame dropout score plus the robust losses in training cope. Frames with large dropouts are dropped. |
| Overlays | **No date/time stamp** visible in this file | Overlay detection (static, high-contrast pixels across a clip) is still worth having for other tapes, but it's lower priority. |
| Scene cuts | 11 cuts at 36.3, 82.8, 139.35, 139.95, 192.7, 244.45, 308.6, 311.95, 322.0, 359.45, 421.2 s → **12 clips**, two of them under 4 s | Splitting clips works with the ffmpeg-style scene score. Tiny clips (< ~5 s) get dropped. |
| Content | A beach with dunes, rocks and sea. People and a dog move through nearly every shot, often close up. Hand-held with fast pans; some shots point straight down at sand and water. | Most of what's static is sand (very little texture) and distant hills. Moving people cover a big share of many frames, so **transient masking or people masks are essential**. Water and sky must be masked (the sky masker already exists). |
| Blur | Laplacian variance median 14 (5–95%: 9–30); heavy motion blur on pans | Frame selection must favour sharp frames: windowed sharpest-of-N rather than a fixed rate. |
| Colour | Per-clip mean RGB varies (e.g. clip 0 warm 104/97/89, clip 1 cool 107/106/111, clip 2 dark 90/86/83) | Per-clip white balance and levels, then per-image appearance correction in training for the drift within a clip. |
| Zoom | None obvious in the sampled frames; needs a measurement from homography scale | Per-clip intrinsics by default. Use per-frame focal only when zoom is detected. |

Per-clip summary (sharpness = median Laplacian variance; higher is sharper):

| Clip | Time (s) | Sharpness | Notes from the contact sheet |
|---|---|---|---|
| 0 | 0 – 36 | 18.3 | Dune edge and beach, people walking. Wide shots, some panning. **Candidate.** |
| 1 | 36 – 83 | 11.9 | Beach toward the sea, then a bright sky pan. |
| 2 | 83 – 139 | 14.9 | Mostly close-ups of hands and sand in shallow water. Unusable for 3D. |
| 3 | 139.4 – 140.0 | 4.7 | 0.6 s glitch. Drop. |
| 4 | 140 – 193 | 13.0 | Dune edge and beach, the same place as clip 0, then a close-up dig. **Candidate, may merge with clip 0.** |
| 5 | 193 – 244 | 25.5 | Kneeling and digging in sand, close range. The sharpest clip, but the subject moves. |
| 6 | 244 – 309 | 14.6 | Open beach with the hill behind, people running. |
| 7 | 309 – 312 | 16.4 | 3 s. Drop. |
| 8 | 312 – 322 | 14.4 | 10 s. |
| 9 | 322 – 359 | 12.2 | Rocks behind the sand, people lying in the sand. **Candidate** (the rocks have texture). |
| 10 | 359 – 421 | 13.1 | Overcast beach with granite boulders, wide. **Candidate.** |
| 11 | 421 – 448 | 13.5 | The same overcast beach, closer. May merge with 10. |

### Second clip: "minimal lateral movement, mostly rotational" (.mov, 25 MB)

| Property | Finding |
|---|---|
| Format | H.264 720×576 (PAL), SAR 16:15 → 768×576 square pixels, flagged progressive at **60 fps** |
| Cadence | **3076 of 4265 frame-to-frame differences are ~0**: every unique frame is repeated three times, so it's really ~20 fps. Duplicates must be dropped before frame selection, otherwise "pick the sharpest in a window" just finds copies. |
| Interlacing | `idet`: 214 TFF and 4 BFF out of 4266, the rest progressive. A **few combed frames** → per-frame field-aware deinterlace. |
| Blue screen | From **56.25 s to the end (71 s)** it's a solid VCR blue screen | Detect and drop "no signal" frames (solid blue or black, near-zero variance). |
| Content | Garden with trees and a brick barbecue, a person walking toward the camera, then a slow pan across a beach | Mostly rotation. Rotation-only clips give poses but no depth. The solve should detect "no parallax" (a homography explains the motion) and say so, rather than produce a flat splat. |
| Other | Head-switching band at the bottom, a QuickTime timecode track | Same crop as above. |

The sample frames that went with this study were shared outside the repo, and none are committed.

## 2. Plan

The work lives mostly in **splatprep**, so the Mac app (which already runs splatprep for learned SfM) and the Linux
worker share it. This follows ARCHITECTURE "Extension points": a restoration stage between ingest and masks that only
rewrites `FrameSet` images, plus per-clip cameras.

### 2.1 Fix first, for every video: square pixels

Apply the stream's sample aspect ratio when extracting frames in `frames.py` and in the Mac `VideoFrameReader`
(resample to square pixels, keeping the height). This is a small, separate fix that also helps the plain video path,
so the comparison in §3 runs both with and without it.

### 2.2 `splatprep vhs` (new module `tools/splatprep/vhs/`)

1. **Analyse** (`task: vhsAnalyse`, fast, low-res decode): SAR, true cadence (duplicate runs), per-frame combing score,
   borders and head-switching band, no-signal frames, scene cuts → clips, per-clip sharpness, motion and parallax
   estimates, zoom estimate, and a usable/unusable verdict with a reason in plain words. Output is `vhs.json`, which
   the New Project sheet shows.
2. **Clean up** each kept frame:
   - drop duplicates and no-signal frames;
   - field-aware deinterlace only for combed frames (bwdif-style: rebuild the missing field from the field itself
     plus its temporal neighbours);
   - crop borders and the head-switching band;
   - resample to square pixels at the measured real detail, so no upscaling;
   - light temporal denoise (3–5 aligned frames) and deblocking;
   - per-clip colour normalisation (white balance and levels against the clip median);
   - mask static overlays such as date stamps when detected.
3. **Select frames per clip**: windowed sharpest frame, a minimum motion between picks (optical flow), and skipping
   frames whose motion is pure rotation relative to the last pick, so the picks carry parallax. Close-ups that fill
   the frame with a moving subject are skipped.
4. **Solve per clip**: one camera per clip (SIMPLE_RADIAL, square pixels after 2.2), or a focal length per frame when
   zoom is detected. Learned features at native resolution, with a higher keypoint budget for small images. Clips
   that fail or have no parallax get a reason.
5. **Merge**: register the solved clips against each other (retrieval pairs across clips → one mapper run with a
   camera per clip). Clips that join become one project. The rest become separate projects or are left out; the user
   chooses.

### 2.3 Training defaults for VHS projects

- per-image appearance correction on;
- SH degree 1 (tape colour can't support view-dependent colour, and higher bands fit noise);
- transient masking on (people and the dog), plus people masks from the existing masker where available;
- sky and water masks on;
- a lower resolution schedule to match the real detail.

All of these are existing settings (`train.appearanceCorrection`, `train.shDegree`, `train.transientMasking`, …), so
VHS is a preset, not new trainer code.

### 2.4 Interface

- A "Home video (VHS, old tapes)" tile on the Mac start page and the same entry in the Linux studio.
- In the New Project sheet, the detected clips: thumbnail, length, verdict ("good: lots of movement", "too short",
  "only turning, no depth", "mostly close-ups") and a tick box. A choice for joining clips into one project or making
  one project per clip.

### 2.5 Optional learned restoration

Upscaling and restoration models (Real-ESRGAN-style) are kept out of the default path. The real detail is ~480×576,
and inventing detail per frame gives inconsistent texture across views, which hurts multi-view fitting. If tried, it
goes in the Models pack as an optional model and is measured (§3) before it's offered.

## 3. How it will be measured

On the main clip's best clips, with the Mac CLI:

- **Plain**: today's video path (`--prepare` on the clip's time range) vs **VHS**: the new path on the same range.
- Report cameras registered / frames offered, held-out PSNR (`--eval`), and renders (`--render`) from a few training
  views and one novel view.
- Report clips that can't be solved and why.
- splatprep unit tests on synthetic video: a known SAR, injected 3:1 duplicates, combed frames, black borders, a
  bottom band, blue-screen frames, a burnt-in date stamp and cuts. The Linux suite keeps passing.
