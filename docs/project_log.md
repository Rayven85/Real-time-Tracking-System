# Machine Learning Navigation — Project Technical Log

**Project:** Real-time vehicle tracking on a miniature race track  
**Period:** April 2026  
**Student:** Rayven85  
**Supervisor:** Akshat  

---

## Project Overview

The goal is to build a top-down camera vision system that:
1. Detects whether a small vehicle is on or off the track
2. Reads road signs (STOP, speed limit 55)
3. Detects a light indicator on the track surface
4. Uses ArUco markers for camera calibration and vehicle localisation

Hardware: GoPro camera (overhead mount), miniature race track (white background, black rounded-rectangle track, STOP sign, 55 speed limit sign, circular LED indicator).

---

## Phase 1: Camera Selection

### Logitech USB Webcam (Rejected)
- Tested first: 1280×720 @ 24 FPS
- **Problem:** Even at maximum mount height, could not cover the full table surface
- Decision: Abandoned in favour of GoPro

### GoPro (Selected)
- Supervisor confirmed GoPro as the primary camera
- Connected via USB Webcam mode on Mac (requires GoPro Webcam software)
- USB Webcam mode supports: Wide, SuperView (not Linear)
- For offline analysis: all FOV modes available

---

## Phase 2: GoPro FOV Comparison

Script: `gopro_distortion.py`

Three modes tested by placing photos in:
- `gopro_samples/linear/`
- `gopro_samples/wide/`
- `gopro_samples/superview/`

The script overlays a green grid on each image — straight grid lines indicate low distortion.

### Results

| Mode | Table Coverage | Distortion | Grid Lines |
|------|---------------|------------|------------|
| Linear | Full | Minimal | Very straight |
| Wide | Full | Slight barrel | Mostly straight |
| SuperView | Full (but table small) | Severe barrel | Clearly curved |

**Conclusion:** Linear mode selected for offline analysis (best quality, no distortion correction needed).  
For USB Webcam real-time use: Wide mode required (Linear not supported in Webcam mode).

---

## Phase 3: Distortion Correction Tool

Script: `undistort.py`

### Why it was built
Wide and SuperView modes show barrel distortion (straight lines appear curved). A manual tuning tool was created to find correction parameters before camera calibration.

### How it works
- Loads images from `gopro_samples/{mode}/` or `RPi_samples/`
- Displays side-by-side: Original (left) | Undistorted (right)
- Green grid overlay on both sides — user tunes until grid lines are straight
- Three sliders: K1 (primary barrel correction), K2 (higher-order), Focal scale
- Press `s` to save result, `r` to reset, `q` to quit

### Usage
```bash
python undistort.py --mode wide
python undistort.py --mode superview
python undistort.py --mode rpi
```

### GoPro Wide calibration result (manual estimate)
- k1 = -0.462
- k2 = -0.054
- fx_scale = 23%

**Note:** These are manually estimated values. For production use, proper camera calibration with a checkerboard is recommended (see `calibrate_gopro.py`).

### RPi fisheye camera
- Separate mode (`--mode rpi`) using `cv2.fisheye` module
- Images from `RPi_samples/` — severe barrel distortion + pink colour cast
- Not selected for this project

---

## Phase 4: Camera Calibration Tool

Script: `calibrate_gopro.py`

### Purpose
Replace manual slider estimates with mathematically precise distortion coefficients using OpenCV's standard checkerboard calibration pipeline.

### How it works
1. Run script once to generate `calibration_checkerboard.png` (print on A4, mount on rigid board)
2. Photograph the checkerboard with GoPro Wide mode from 20–30 different angles/distances
3. Place images in `calib_images/`
4. Run script again — OpenCV detects inner corners, computes camera matrix K and distortion coefficients dist
5. Outputs `gopro_calib.npz` (K, dist, image size)
6. Reports reprojection error (good result: < 1.0 px)

### Checkerboard specification
- 10×7 squares → 9×6 inner corners (BOARD_W=9, BOARD_H=6)
- Square size: 25 mm (measure actual printed size and update SQUARE_MM)

---

## Phase 5: ArUco Marker System

### Why ArUco over AprilTag

| | AprilTag | ArUco |
|---|---|---|
| Origin | University of Michigan | University of Córdoba → OpenCV |
| Python usage | Requires separate `pupil-apriltags` library | Built into `cv2` — no extra install |
| Detection speed | Slower in Python | Faster |
| Documentation | Academic papers | Full OpenCV official docs |
| Robustness | Higher (far/blurry scenes) | Sufficient for close overhead capture |

**Decision:** ArUco chosen — zero extra dependencies, full OpenCV integration, sufficient accuracy for controlled indoor overhead scene.

### Marker generation

Script: `generate_markers.py`  
Dictionary: `DICT_4X4_50` (simple 4×4 pattern, easiest to detect at close range)

| Marker | ID | Purpose |
|--------|----|---------|
| corner_0.png | 0 | Table top-left corner |
| corner_1.png | 1 | Table top-right corner |
| corner_2.png | 2 | Table bottom-right corner |
| corner_3.png | 3 | Table bottom-left corner |
| car.png | 4 | Mounted on vehicle top |

Print size recommendations:
- Corner markers: 4–6 cm square
- Car marker: 3–4 cm square

### Why DICT_4X4_50
- Simplest pattern (16 data bits) → easiest to detect at close overhead distance
- 50 unique IDs — more than sufficient for this application
- Faster detection than 5×5 or 6×6 dictionaries

---

## Phase 6: Main Detection System

Script: `aruco_detect.py`

### Architecture

```
GoPro (USB Webcam, Wide FOV)
        ↓
 [Optional undistortion]
        ↓
 ArUco marker detection
        ↓
 4 corner markers found? → Perspective warp → Top-down view
                                    ↓
                          Car marker (ID=4) detected?
                                    ↓
                     Query saved track mask at car position
                                    ↓
                          ON TRACK / OFF TRACK
```

### Key features

**Camera discovery**  
`python aruco_detect.py` lists all available cameras with resolution.  
`python aruco_detect.py --source 0` opens a specific camera.

On Mac, GoPro via USB Webcam appears as index 0 but requires `cv2.CAP_AVFOUNDATION` backend and several warm-up frames before frames are readable.

**Perspective warp**  
Once all 4 corner markers (ID 0–3) are detected, `cv2.getPerspectiveTransform` computes a homography matrix M. Every frame, the raw image is warped into a 600×600 px top-down orthographic view. The warp matrix updates each frame so it self-corrects if the camera moves slightly.

**ON/OFF TRACK detection — key design decision**

*Problem:* The car marker (a black-and-white square) physically covers the track beneath it. Any pixel-based track detection at the car's position will read the marker pattern, not the track.

*Solution (supervisor's recommendation):*
1. **Before placing the car marker:** Press `c` to capture the track mask from the clean top-down view
2. The mask is saved to `track_mask.png`
3. **After placing the car marker:** The saved (unoccluded) mask is used for all subsequent checks

The detection checks a 20 px radius around the car's warped position against the saved mask. If >10% of pixels in that region are track pixels → ON TRACK.

A temporal debounce filter (8 frames, 60% majority) prevents flickering when the car is near the track edge.

**Keyboard controls**

| Key | Action |
|-----|--------|
| `w` | Toggle perspective warp (top-down view) |
| `c` | Capture track mask (do this before placing car marker) |
| `m` | Show track mask alongside top-down view (debug) |
| `d` | Show distortion comparison (original vs undistorted) |
| `s` | Save screenshot to `screenshots/` |
| `q` | Quit |

---

## Phase 7: Issues Encountered & Resolutions

| Issue | Cause | Resolution |
|-------|-------|------------|
| Logitech can't cover full table | Fixed focal length, insufficient height | Switched to GoPro |
| GoPro not found by OpenCV index scan | Mac requires `CAP_AVFOUNDATION` + warm-up frames | Added AVFoundation backend + 5 discard frames |
| Chinese text shows as `???` | OpenCV `putText` does not support Unicode/Chinese | Replaced all labels with English |
| ON/OFF TRACK always flickering | Sampling car marker's own black pixels | Added temporal debounce filter |
| ON/OFF TRACK always OFF | Erasing car marker region removed underlying track | Stopped erasing car region from mask |
| ON/OFF TRACK always OFF (2nd cause) | Car marker physically covers the track | Adopted pre-capture approach: scan track before placing marker |
| Undistortion makes image worse | Manual k1 estimate too aggressive for this mount height | Disabled undistortion (GoPro Wide distortion negligible at this height) |

---

## File Structure

```
TrackingSystem/
├── aruco_detect.py        # Main detection system (ArUco + warp + on/off track)
├── generate_markers.py    # Generate ArUco marker images for printing
├── gopro_distortion.py    # Compare GoPro FOV modes (Linear/Wide/SuperView)
├── undistort.py           # Manual distortion correction tuning tool
├── calibrate_gopro.py     # Checkerboard camera calibration tool
├── markers/               # Generated ArUco marker PNGs
├── gopro_samples/         # GoPro FOV comparison images
│   ├── linear/
│   ├── wide/
│   └── superview/
├── calib_images/          # Checkerboard calibration images (to be filled)
├── screenshots/           # Saved detection screenshots
└── track_mask.png         # Saved track mask (generated at runtime by pressing 'c')
```

---

## Next Steps

- [ ] Complete proper camera calibration using `calibrate_gopro.py` (checkerboard photos)
- [ ] Attach car marker to actual vehicle and test ON/OFF TRACK in motion
- [ ] Add road sign detection (STOP, speed limit 55) to the warped top-down view
- [ ] Test on the larger MDLS lab table (may require second camera + ArUco stitching)
- [ ] Explore GoPro USB Webcam mode stability for sustained real-time operation
