"""
Auto-labeller for track-specific YOLO training data
=====================================================
Reads 600x600 warped-view screenshots and generates YOLO-format .txt labels.
Auto-detects signs with HSV, then lets you adjust each box with the mouse.

Usage:
    python auto_label.py                    # labels train split
    python auto_label.py --split val        # labels val split

Controls (per image):
    Enter / Space  — accept all boxes and save
    e              — edit: redraw each box with mouse (drag to select, Space to confirm)
    d              — discard this image (no label saved)
    q              — quit
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

# ── ROI coordinates (x1, y1, x2, y2) in 600×600 warped view ─────────
ROI_STOP  = (305, 472, 402, 550)
ROI_55    = (468, 283, 568, 362)
ROI_LIGHT = (10,  252, 72,  315)

RED_RATIO_STOP = 0.10
RED_RATIO_55   = 0.04
GREEN_RATIO_LIGHT = 0.05   # HSV green pixel fraction → green light
RED_RATIO_LIGHT   = 0.05   # HSV red pixel fraction   → red light

COLOURS = {
    'stop':        (0,   60, 255),
    'speed_55':    (255, 80,   0),
    'light_off':   (120, 120, 120),
    'light_green': (0,  200,   0),
    'light_red':   (0,   0,  220),
}

# Auto-label boxes: class_id, x_center, y_center, width, height (normalised to 600×600)
AUTO_BOXES = {
    'stop':        (0, 0.5892, 0.8517, 0.1617, 0.1300),
    'speed_55':    (1, 0.8633, 0.5375, 0.1667, 0.1317),
    'light_off':   (2, 0.0683, 0.4725, 0.1033, 0.1050),
    'light_green': (3, 0.0683, 0.4725, 0.1033, 0.1050),
    'light_red':   (4, 0.0683, 0.4725, 0.1033, 0.1050),
}


# ── Helpers ──────────────────────────────────────────────────────────

def color_ratio(region, lower1, upper1, lower2=None, upper2=None):
    if region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower1, upper1)
    if lower2 is not None:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))
    return float(mask.sum()) / 255.0 / region.size


def detect_light_state(region):
    """Return 'light_green', 'light_red', or 'light_off'."""
    g = color_ratio(region, (40, 60, 60), (85, 255, 255))
    r = color_ratio(region, (0, 80, 80), (10, 255, 255), (160, 80, 80), (180, 255, 255))
    if g > GREEN_RATIO_LIGHT:
        return 'light_green'
    if r > RED_RATIO_LIGHT:
        return 'light_red'
    return 'light_off'


def detect_visible(img):
    visible = []
    x1, y1, x2, y2 = ROI_STOP
    if color_ratio(img[y1:y2, x1:x2], (0, 80, 80), (10, 255, 255),
                   (160, 80, 80), (180, 255, 255)) > RED_RATIO_STOP:
        visible.append('stop')
    x1, y1, x2, y2 = ROI_55
    if color_ratio(img[y1:y2, x1:x2], (0, 80, 80), (10, 255, 255),
                   (160, 80, 80), (180, 255, 255)) > RED_RATIO_55:
        visible.append('speed_55')
    x1, y1, x2, y2 = ROI_LIGHT
    visible.append(detect_light_state(img[y1:y2, x1:x2]))
    return visible


def box_to_pixels(xc, yc, w, h, W=600, H=600):
    x1 = int((xc - w / 2) * W)
    y1 = int((yc - h / 2) * H)
    x2 = int((xc + w / 2) * W)
    y2 = int((yc + h / 2) * H)
    return x1, y1, x2, y2


def pixels_to_norm(x1, y1, x2, y2, W=600, H=600):
    xc = (x1 + x2) / 2 / W
    yc = (y1 + y2) / 2 / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    return xc, yc, w, h


def draw_boxes(img, boxes):
    """boxes: {name: (cls_id, xc, yc, w, h)}"""
    out = img.copy()
    for name, (cls_id, xc, yc, w, h) in boxes.items():
        col = COLOURS[name]
        x1, y1, x2, y2 = box_to_pixels(xc, yc, w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, name, (x1 + 3, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(out, name, (x1 + 3, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    return out


def add_hint(img, text):
    out = img.copy()
    cv2.putText(out, text, (5, out.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
    cv2.putText(out, text, (5, out.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    return out


# ── Edit mode: redraw each box with mouse ────────────────────────────

def edit_boxes(img, boxes):
    """Let user redraw each visible box with selectROI. Returns updated boxes."""
    updated = dict(boxes)
    WIN = "Edit box (drag to draw, Space/Enter=confirm, c=cancel sign)"

    for name in list(updated.keys()):
        cls_id, xc, yc, w, h = updated[name]
        col = COLOURS[name]

        # Highlight current box + instruction
        canvas = draw_boxes(img, updated)
        x1, y1, x2, y2 = box_to_pixels(xc, yc, w, h)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 3)  # highlight in cyan
        canvas = add_hint(canvas, f"Redrawing [{name}]  drag=new box  Space=keep  c=remove sign")
        cv2.imshow(WIN, canvas)
        cv2.waitKey(300)

        roi = cv2.selectROI(WIN, canvas, showCrosshair=True, fromCenter=False)
        rx, ry, rw, rh = roi

        if rw > 5 and rh > 5:
            new_xc, new_yc, new_w, new_h = pixels_to_norm(rx, ry, rx + rw, ry + rh)
            updated[name] = (cls_id, new_xc, new_yc, new_w, new_h)
            print(f"    [{name}] updated: xc={new_xc:.3f} yc={new_yc:.3f} w={new_w:.3f} h={new_h:.3f}")
        else:
            # Empty selection = keep original (user pressed Space without drawing)
            print(f"    [{name}] kept original")

    cv2.destroyWindow(WIN)
    return updated


# ── Label I/O ────────────────────────────────────────────────────────

def write_label(label_path, boxes):
    lines = [f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
             for cls_id, xc, yc, w, h in boxes.values()]
    label_path.write_text("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--img-dir", default=None)
    args = parser.parse_args()

    img_dir   = Path(args.img_dir) if args.img_dir else Path("track_dataset/images") / args.split
    label_dir = Path("track_dataset/labels") / args.split

    if not img_dir.exists():
        print(f"Image folder not found: {img_dir}")
        print("Put your 600×600 warped-view screenshots there first.")
        return

    label_dir.mkdir(parents=True, exist_ok=True)

    exts = {'.jpg', '.jpeg', '.png'}
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)
    if not images:
        print(f"No images in {img_dir}")
        return

    print(f"Found {len(images)} images")
    print("Controls:")
    print("  Enter      — save and next")
    print("  1          — toggle STOP box on/off")
    print("  2          — toggle 55 box on/off")
    print("  3          — cycle light state (off → green → red → off)")
    print("  e          — edit: redraw boxes with mouse")
    print("  d          — discard this image")
    print("  q          — quit\n")

    # All sign names that are always proposed
    SIGN_BOXES   = ['stop', 'speed_55']
    LIGHT_STATES = ['light_off', 'light_green', 'light_red']

    accepted = discarded = skipped = 0
    WIN = "Auto-label preview"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 600, 660)

    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        label_path = label_dir / (img_path.stem + ".txt")

        # Auto-detect initial state
        auto = detect_visible(img)
        light_state = next((n for n in auto if n.startswith('light_')), 'light_off')
        show_stop  = 'stop'     in auto
        show_55    = 'speed_55' in auto

        # Mutable copies of box coords — edits persist across loop iterations
        custom_boxes = {k: v for k, v in AUTO_BOXES.items()}

        print(f"  [{i+1}/{len(images)}] {img_path.name}  auto={auto}")

        while True:
            # Build active boxes dict from custom_boxes (not AUTO_BOXES)
            boxes = {}
            if show_stop:  boxes['stop']     = custom_boxes['stop']
            if show_55:    boxes['speed_55'] = custom_boxes['speed_55']
            boxes[light_state] = custom_boxes[light_state]

            # Draw
            canvas = draw_boxes(img, boxes)
            # Status bar
            status = (f"STOP={'ON' if show_stop else 'off'}  "
                      f"55={'ON' if show_55 else 'off'}  "
                      f"LIGHT={light_state.upper()}")
            cv2.putText(canvas, status, (5, canvas.shape[0] - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(canvas, status, (5, canvas.shape[0] - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
            canvas = add_hint(canvas, "Enter=save  1=stop  2=55  3=light  e=edit  d=discard  q=quit")
            cv2.imshow(WIN, canvas)

            key = cv2.waitKey(0) & 0xFF

            if key in (13, 32):            # Enter — save
                write_label(label_path, boxes)
                accepted += 1
                print(f"    → saved: {list(boxes.keys())}")
                break

            elif key == ord('1'):          # toggle STOP
                show_stop = not show_stop

            elif key == ord('2'):          # toggle 55
                show_55 = not show_55

            elif key == ord('3'):          # cycle light state
                idx = LIGHT_STATES.index(light_state)
                light_state = LIGHT_STATES[(idx + 1) % len(LIGHT_STATES)]

            elif key == ord('e'):          # edit boxes with mouse
                boxes = edit_boxes(img, boxes)
                # persist edited coords back into custom_boxes
                for name, coords in boxes.items():
                    custom_boxes[name] = coords
                show_stop   = 'stop'     in boxes
                show_55     = 'speed_55' in boxes
                light_state = next((n for n in boxes if n.startswith('light_')), light_state)

            elif key == ord('d'):          # discard
                if label_path.exists():
                    label_path.unlink()
                discarded += 1
                print(f"    → discarded")
                break

            elif key == ord('q'):
                cv2.destroyAllWindows()
                _summary(accepted, discarded, skipped, len(images))
                return

    cv2.destroyAllWindows()
    _summary(accepted, discarded, skipped, len(images))


def _summary(accepted, discarded, skipped, total):
    print(f"\n{'='*40}")
    print(f"  Accepted : {accepted}")
    print(f"  Discarded: {discarded}")
    print(f"  Skipped  : {skipped}")
    print(f"  Total    : {total}")
    print(f"{'='*40}")
    print("Next: python train_track.py")


if __name__ == "__main__":
    main()
