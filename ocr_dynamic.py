import cv2
import numpy as np
import time
import easyocr
from maxarm_pc import MaxArmClient

# === CONFIG ===
PORT = "COM2165"      # MaxArm COM port
CAM_INDEX = 0         # camera index

# Robot Z positions
Z_FIXED = 180         # normal hover height
Z_TAP   = 75          # tap height
TAP_DOWN_TIME = 500   # ms
TAP_UP_TIME   = 500   # ms

# Arm off-screen position after tapping
OFFSCREEN_X = 100
OFFSCREEN_Y = -100
OFFSCREEN_Z = 200

# start position after homing (to the side of the phone)
HOME_X = 100
HOME_Y = -100
HOME_Z = 200

# OCR scale factor
OCR_SCALE = 1

SETUP_SEQUENCE = [
    {
        "name": "welcome_start",
        "targets": ["start"],
        "post_delay": 2.0,
        "optional": False,
    },
    {
        "name": "agree_to_all",
        "targets": ["agree to all (optional)"],
        "post_delay": 2.0,
        "optional": False,
    },
    {
        "name": "agree_button",
        "targets": ["agree"],
        "post_delay": 2.0,
    },
    {
        "name": "setup_manually",
        "targets": ["set up manually"],
        "post_delay": 2.0,
    },
    {
        "name": "skip_button",
        "targets": ["skip"],
        "post_delay": 2.0,
    },
    {
        "name": "skip_wifi",
        "targets": ["skip"],
        "post_delay": 2.0,
    },
    {
        "name": "skip_sim",
        "targets": ["skip"],
        "post_delay": 2.0,
    },
    {
        "name": "next_date",
        "targets": ["next"],
        "post_delay": 2.0,
    },
    {
        "name": "more1",
        "targets": ["more"],
        "post_delay": 2.0,
    },
    {
        "name": "more2",
        "targets": ["more"],
        "post_delay": 2.0,
    },
    {
        "name": "accept_google",
        "targets": ["accept"],
        "post_delay": 2.0,
    },
    {
        "name": "skip_protect",
        "targets": ["skip"],
        "post_delay": 2.0,
    },
    {
        "name": "skip_anyway",
        "targets": ["skip anyway"],
        "post_delay": 2.0,
    },
    {
        "name": "accept_ATT",
        "targets": ["accept"],
        "post_delay": 2.0,
    },
    {
        "name": "agree_samsung",
        "targets": ["agree"],
        "post_delay": 2.0,
    },
    {
        "name": "next_theme",
        "targets": ["next"],
        "post_delay": 4.0,
    },
    {
        "name": "finish",
        "targets": ["finish"],
        "post_delay": 2.0,
    }
]


def pixel_to_world(px, py, frame_w, frame_h):
    """
    Convert pixel (x,y) to robot workspace coordinates.
    X is linearly mapped then scaled toward center so we don't overshoot left/right.
    """

    # Normalize (0..1)
    nx = px / frame_w
    ny = py / frame_h

    # Compensate for 180° camera flip:
    nx = 1.0 - nx
    ny = 1.0 - ny

    # Workspace bounds
    X_MIN, X_MAX = -245, 245
    Y_MIN, Y_MAX = -245, -30

    # --- raw linear mapping (old behavior) ---
    x_raw = X_MIN + nx * (X_MAX - X_MIN)
    y_raw     = Y_MIN + ny * (Y_MAX - Y_MIN)

    # --- compress X and Y around center to fix overshoot ---
    X_SCALE = 0.2   # start with 0.7 (70%), tune this number
    X_OFFSET = 0.0  # if you later see a constant bias, you can tweak this
    Y_SCALE = 0.9
    Y_OFFSET = 0.0

    x = X_OFFSET + x_raw * X_SCALE
    y = Y_OFFSET + y_raw * Y_SCALE
    z = Z_FIXED

    return x, y, z



def tap(arm, x, y):
    """Move arm over target → down → tap → up → move off-screen."""
    print(f"TAP @ ({x:.1f}, {y:.1f})")

    # 1) Move over target at safe hover height
    arm.move_xyz(x, y, Z_FIXED, time_ms=500)
    time.sleep(0.5)

    # 2) Go straight down to tap height
    arm.move_xyz(x, y, Z_TAP, time_ms=TAP_DOWN_TIME)
    time.sleep(TAP_DOWN_TIME / 1000.0)  # 500 ms

    # 3) Go straight back up
    arm.move_xyz(x, y, Z_FIXED, time_ms=TAP_UP_TIME)
    time.sleep(TAP_UP_TIME / 1000.0)    # 500 ms

    # 4) Move away at safe height
    print("Moving off-screen...")
    arm.move_xyz(OFFSCREEN_X, OFFSCREEN_Y, OFFSCREEN_Z, time_ms=700)
    time.sleep(0.7)


def run_step(step, arm, reader, cap, max_wait=20):
    """
    Run one scripted step:
      - Take a frame
      - Run OCR once
      - If found, tap, then return True
      - If not found, wait 2 seconds and retry until max_wait
    Assumes the arm is already in OFFSCREEN position when this is called.
    """

    name       = step.get("name", "unnamed")
    targets    = step["targets"]
    post_delay = step.get("post_delay", 1.0)
    optional   = step.get("optional", False)

    print(f"\n=== STEP: {name} (targets: {targets}) ===")
    t0 = time.time()

    while time.time() - t0 < max_wait:
        # --- grab a fresh frame ---
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed, retrying in 2s...")
            time.sleep(2)
            continue

        frame = cv2.flip(frame, -1)

        # main phone crop (same as before)
        h, w, _ = frame.shape
        crop_left   = int(w * 0.35)
        crop_right  = int(w * 0.65)
        crop_top    = int(h * 0.05)
        crop_bottom = int(h * 0.90)
        frame_step  = frame[crop_top:crop_bottom, crop_left:crop_right]

        # ---------- SPECIAL CASE: "agree_button" -> bottom 20% only ----------
        if name == "agree_button" or name == "setup_manually" or name == "skip_button" or name == "skip_wifi" or name == "skip_sim" or name == "next_date" or name =="more1" or name =="more2" or name == "accept_google" or name == "skip_protect" or name == "agree_samsung" or name == "accept_ATT":
            h_s, w_s, _ = frame_step.shape
            start_y = int(h_s * 0.8)            # bottom 20%
            roi = frame_step[start_y:h_s, :, :] # region of interest for OCR

            cv2.imshow("OCR", frame_step)       # still show full area
            cv2.waitKey(1)

            result = run_ocr_on_frame(reader, roi, targets)

            if result:
                cx_roi, cy_roi, bbox_full, matched = result

                # shift ROI coords back into frame_step coords
                cx = cx_roi
                cy = cy_roi + start_y
                bbox_full[:, 1] += start_y
        else:
            # default: use whole phone crop for other steps
            cv2.imshow("OCR", frame_step)
            cv2.waitKey(1)

            result = run_ocr_on_frame(reader, frame_step, targets)

            if result:
                cx, cy, bbox_full, matched = result
        # --------------------------------------------------------------------

        if result:
            print(f"Found '{matched}' at ({cx}, {cy}) for step '{name}'")

            # Visual debug overlay on full phone crop
            cv2.polylines(frame_step, [bbox_full], True, (0, 255, 0), 2)
            cv2.circle(frame_step, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(
                frame_step,
                f"{name}: {matched}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            cv2.imshow("OCR", frame_step)
            cv2.waitKey(1)

            # Map pixel → world using the *full* phone-crop size
            h2, w2, _ = frame_step.shape
            x, y, z = pixel_to_world(cx, cy, w2, h2)

            tap(arm, x, y)
            time.sleep(post_delay)
            return True

        print(f"'{targets}' not found yet for step '{name}'. Retrying in 2s...")
        time.sleep(2)

    # Timed out
    msg = f"Step '{name}' timed out after {max_wait}s."
    if optional:
        print(msg + ' Marking as optional and continuing.')
        return False
    else:
        print(msg + ' Stopping sequence.')
        return False


def run_ocr_on_frame(reader, frame, targets):
    """
    OCR: find all matches for ANY target word, pick the one with the largest area.
    Returns: (cx_full, cy_full, bbox_full, matched_target) or None
    """
    h, w, _ = frame.shape

    # 1) Crop center region (small)
    x1, x2 = w // 8, 7 * w // 8
    y1, y2 = h // 8, 7 * h // 8
    crop = frame[y1:y2, x1:x2]

    # 2) Resize
    big = cv2.resize(crop, None, fx=OCR_SCALE, fy=OCR_SCALE)

    # 3) Sharpen
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)

    results = reader.readtext(sharp, detail=1)
    lower_targets = [t.lower() for t in targets]

    matches = []  # (area, cx_full, cy_full, pts_full, matched_target)

    for (bbox, text, conf) in results:
        txt = text.strip().lower()

        for target in lower_targets:
            if target in txt:
                pts_big = np.array(bbox, dtype=np.float32)
                pts_crop = pts_big / OCR_SCALE

                cx_crop = float(np.mean(pts_crop[:, 0]))
                cy_crop = float(np.mean(pts_crop[:, 1]))

                cx_full = int(x1 + cx_crop)
                cy_full = int(y1 + cy_crop)

                pts_full = np.zeros_like(pts_crop)
                pts_full[:, 0] = x1 + pts_crop[:, 0]
                pts_full[:, 1] = y1 + pts_crop[:, 1]

                area = cv2.contourArea(pts_full.astype(np.float32))

                matches.append((area, cx_full, cy_full, pts_full.astype(int), target))

    if not matches:
        return None

    matches.sort(reverse=True, key=lambda m: m[0])
    area, cx_full, cy_full, pts_full, matched_target = matches[0]

    return cx_full, cy_full, pts_full, matched_target


def main():
    arm = MaxArmClient(PORT)
    print("Homing...")
    arm.home(2000)

    print("Moving to start (side) position...")
    arm.move_xyz(HOME_X, HOME_Y, HOME_Z, time_ms=1000)
    time.sleep(1.0)

    # Camera setup
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # lower res to speed up OCR
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    reader = None  # lazy init

    print("Preview running. Press 's' to start sequence, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, -1)

        h, w, _ = frame.shape
        crop_left   = int(w * 0.35)
        crop_right  = int(w * 0.65)
        crop_top    = int(h * 0.05)
        crop_bottom = int(h * 0.90)
        frame_view  = frame[crop_top:crop_bottom, crop_left:crop_right]

        cv2.putText(
            frame_view,
            "Press 's' to start automation, 'q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("OCR", frame_view)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('s'):
            # Initialize OCR reader on first run
            if reader is None:
                print("Initializing OCR (this may take a moment)...")
                reader = easyocr.Reader(['en'])

            print("Starting setup sequence...")
            for step in SETUP_SEQUENCE:
                success = run_step(step, arm, reader, cap, max_wait=25)
                if not success and not step.get("optional", False):
                    print("Stopping automation due to failed required step.")
                    break

            print("Sequence finished. Press 's' to run again or 'q' to quit.")

    cap.release()
    cv2.destroyAllWindows()
    arm.close()


if __name__ == "__main__":
    main()
