# VHS / old home video → splats

Status: **built** (2026-11-13): the *Home video* source on the Mac and in SplatWork for Linux, `tools/splatprep/vhs.py`.
Roadmap item from 2026-11-06: "a restoration pipeline for low-quality footage such as VHS home videos". §1 is the
study of the sample footage, §2 the plan as built (with what changed on the way), §3 the measurements.

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

### Third clip: "field with car" (.mp4, 431 MB)

The same transfer chain as the main clip: HEVC 1890×1068 at SAR 712:945, 20 fps, 6 min 3 s. A short shot of a bird,
then a station wagon doing laps of a dry field while the camera mostly pans and zooms from the edge of the field.
There's a blue no-signal screen at 177.8–188.8 s and a burst of damaged frames around 9.4–10.9 s; one of them,
looked at closely, is all tearing and smeared blocks (comb score 0.79 against 0.5–0.6 for clean frames). The car
kicks up dust and fills much of many frames. The camera pans and zooms from a fixed spot, so there is depth only
where it walks.

The sample frames that went with this study were shared outside the repo, and none are committed.

## 2. As built

The work lives in splatprep (`tools/splatprep/vhs.py`), so the Mac (local splatprep) and Linux workers share it:
`task: frames` with a `vhs` object (PROTOCOL.md). The Mac's source is `SourceDescription.homeVideo`, and the Linux
studio's source kind is `homevideo`.

### 2.1 Square pixels, for every video

`frames.py` (`pixel_aspect`, `square_pixels`) and the Mac's `VideoFrameReader` (`pixelAspect`, applied in `render`)
resample stored widths to square pixels and keep the scan lines. This isn't specific to VHS: the plain video path
on the test clip placed **73 of 73** cameras with square pixels against **64 of 73** stretched (§3).

### 2.2 Analysis (one pass, ~95 s for the 7.5 min main clip)

Every frame is decoded at a quarter of the width and full height, which is enough for combing and borders. Each
frame is measured for:
- **repeats:** mean difference to the previous frame < 0.6 gray levels;
- **no signal:** a uniform frame;
- **comb score:** adjacent-line over same-field-line difference. Clean frames measure 0.5–0.6; above 0.7 counts as
  combed or tape-damaged. None of the three samples has real interlacing left, only damaged frames;
- **tracking-dropout streaks:** thin runs brighter or darker than the lines two above and two below;
- **sharpness:** Laplacian variance on the central 80 %;
- **colour statistics.**

KLT tracks on a 320-px thumbnail give:
- **camera travel:** the median feature motion;
- **zoom:** the homography scale over ~0.5 s, counted only where there is no parallax;
- **parallax:** the share of tracks that fit the epipolar geometry but not a homography;
- **rigidity:** the share of tracks that fit one rigid motion.

The per-frame data is turned into clips and properties:
- **Borders and the head-switching band:** from time-averaged column and row profiles. The main clip crops
  21 / 13 px at the sides and 19 rows at the bottom.
- **Cuts:** a jump well above the recent frame-to-frame change. Glitch bursts shorter than 1 s fold into one short
  clip.
- **Burnt-in overlays:** static, edgy pixels whose surroundings change.
- **Verdicts:** `good`, `rotationOnly` (parallax < 10 %), `static`, `blurry`, `moving` (rigidity < 55 %: people
  close up, water), `short` and `noSignal`.
- **Recommended clip:** the `good` clip with the best depth × length × sharpness × rigidity².
- **Cache:** the per-frame data is kept in `vhs-analysis.npz`, so finding the clips and preparing the frames decode
  the file once.

### 2.3 Frames

- **Picking:** per clip, a new frame each time the camera has moved a step, or 1.5 s have passed with some movement;
  within each stretch the sharpest frame wins. Sharpness counts only up to twice the clip median, because tearing
  scores as "sharp". The step adapts to the frame budget, which is shared by camera travel.
- **Clean-up order:**
  1. rebuild combed frames from one field;
  2. inpaint dropout streaks;
  3. crop;
  4. resample to square pixels at ≤ 576 lines (the main clip gives 768 × 576);
  5. motion-compensated temporal denoise: two neighbours each side, aligned by a homography and blended where they
     agree, so moving people don't ghost;
  6. one gray-world + levels transform per clip (gains within ±20 %, stretch ≤ 1.15×).
- **Frame metadata:** each frame carries `clip` and a `cameraGroup`: one per clip, plus a new group each time the lens
  has zoomed more than ~8 %.

### 2.4 Cameras: what changed after measuring

The plan was a camera per clip and joining clips that overlap. One joint solve of all ten `good` clips of the main
clip went like this:
- 561 frames, 58 min.
- It placed 526 frames, but stacked three clips of the overcast beach onto one spot, near a fourth clip of hands in
  water.
- One zoom group came out with a 2° field of view.
- Clip 13 alone solves cleanly in 4 min.

So now:
- **One clip by default:** the project uses the recommended clip. You tick more clips of the same place to join them.
- **Wrong joins are left out:** with several clips, splatprep's `check_clips` (sfm `clipTravel`) leaves out a clip
  whose solved cameras sit on one spot although the analysis saw the camera travel, or that got an impossible zoom.
  On the joint solve it leaves out exactly the three stacked clips and the 2° group.
- **Compared with the best clip, not the median (2026-11-14):** the first real project picked 9 clips. It was restored
  and solved on the Linux worker, and the solve placed only clip 6 properly (path 28.6), squashing five other clips
  onto it (paths 0.06–2.4). The splat was unusable (16.9 dB). Two things let it through:
  - `vhs.json` didn't come back from the worker, so neither the check nor the tape training settings ran. It now
    comes back, and a dataset whose frames carry a clip counts as home video too.
  - The check compared each clip with the median clip, which was itself a collapsed clip. It now compares with the
    best-placed clip. On that project's poses it keeps clip 6 alone, which is the clip that trains to 20.57 dB
    in §3.
- **Solver choice:** a single clip with one lens setting goes to **Object Capture** on the Mac, which did best (§3).
  Several clips or zoom groups go to splatprep (SuperPoint + LightGlue), which takes a camera per group.

### 2.5 Training defaults

`HomeVideoIngest.adjustTraining` on the Mac and `ProjectOptions.train_config` in the Linux studio set:
- SH degree ≤ 1;
- transient masking on;
- training size ≤ 1024 px.

These apply unless the user customised the settings. Accuracy is the same as with the default settings (§3), but
people and the dog are left out instead of ghosting.

### 2.6 Not done, and why

- **Learned upscaling or restoration** (Real-ESRGAN and the like): not used. The frames hold ~768 × 576 of real
  detail. Upscaling invents texture per frame that doesn't agree between views, and the denoise above already uses
  the real extra information (neighbouring frames). It could be an optional Models-pack step later, if a
  measurement shows a gain.
- **Linux clip list:** the Linux studio takes clip numbers and lists the clips in the project log. The Mac's list with
  pictures isn't mirrored yet.
- **Deinterlacing** is tested on synthetic combing only: none of the samples has interlacing left.

## 3. Results

All runs on the Mac CLI (M-series, the Balanced preset, 15k steps). Every 8th camera, in time order, was held out
of training and scored (`--eval`). Each path's held-out photos are its own frames, so the numbers compare
reconstruction quality per path, not identical images. The held-out photos contain the people, and the VHS
training default leaves people out, so that row is penalised where they stand.

### Clip 13 of the main clip (overcast beach with boulders, 384.7–421.2 s, cut out losslessly)

| Path | Frames | Cameras placed | Solve | Held-out PSNR |
|---|---|---|---|---|
| Plain video, before this work (stretched pixels, Object Capture) | 73 | 64 | 21 s | 18.55 dB |
| Plain video, square pixels (Object Capture) | 73 | 73 | 18 s | 19.09 dB |
| Home video, learned SfM (per-clip camera) | 75 | 72 | 4 min | 18.89 dB (one edge view 11.8) |
| Home video, learned SfM, default training settings | 75 | 72 | 4 min | 18.87 dB |
| **Home video, Object Capture (now the default for one clip)** | 75 | 72 | 31 s | **19.56 dB** |

The renders from a held-out camera matter more here than the PSNR:
- **Plain path:** sand blotches where people stood.
- **Home video path:** clean sand and rocks, with the people and the dog gone.

### Clip 6 of the main clip (open beach under the dunes, 239.0–308.4 s): the default flow

This is what a user gets: the whole 7.5 min file opened as home video with nothing changed. The analysis
recommends clip 6. It took 150 frames in two zoom groups, so the solve was learned SfM. The plain paths read a
lossless cut of the same clip.

| Path | Frames | Cameras placed | Prepare | Held-out PSNR |
|---|---|---|---|---|
| Plain video, before this work (stretched pixels, Object Capture) | 139 | — | — | **fails**: Object Capture error 6, twice |
| Plain video, square pixels (Object Capture) | 139 | 121 | 27 s | 19.36 dB |
| **Home video (default)** | 150 | 122 (18 off the camera path left out) | 12 min 20 s, incl. the 95 s analysis | **20.57 dB** |

Training took 7.8 min for home video and 13.9 min for plain (smaller frames). In held-out renders, the plain splat
has ghost people and a smeared dog where the family played; the home-video splat shows the clean beach, dunes and
sky. This is the main clip's usable splat.

### What can't be solved

- **The joint solve of all ten clips:** see §2.4.
- **Clip 2 (71.9–139.4 s):** hands and towels in shallow water, very shaky. It is now `moving` (rigidity 52 %), and
  it was the top-ranked clip before rigidity was added.
- **The second sample (garden and beach pan):** the camera mostly turns. The garden walk-up (0.4–34.7 s) is `good`
  with parallax 17 %; the pans are `rotationOnly`, and the last 15 s are a blue screen.
- **The third sample (field with car):** the camera pans and zooms from the edge of the field after a moving car. The
  analysis rates most clips `good` (parallax 27–37 %), but much of that parallax is the car, and every long clip
  zooms. It is not measured here; expect the fixed scenery (trees, fence) at best.
