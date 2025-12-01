import cv2
import numpy as np
import time
from maxarm_pc import MaxArmClient

# NEW: import your tap model helpers
from tap_model import load_tap_model, predict_tap

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


def pixel_to_world(px, py, frame_w, frame_h):
    """
    Convert pixel (x,y) in the phone-crop image to robot workspace coordinates.
    You may need to tune these bounds/scale for your setup.
    """

    # Normalize (0..1)
    nx = px / frame_w
    ny = py / frame_h

    # Compensate for 180° camera flip:
    nx = 1.0 - nx
    ny = 1.0 - ny

    # Workspace bounds (tune as needed)
    X_MIN, X_MAX = -245, 245
    Y_MIN, Y_MAX = -245, -30

    # Raw linear mapping
    x_raw = X_MIN + nx * (X_MAX - X_MIN)
    y_raw = Y_MIN + ny * (Y_MAX - Y_MIN)

    # Compress around center if needed
    X_SCALE = 0.2
    X_OFFSET = 0.0
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
    time.sleep(TAP_DOWN_TIME / 1000.0)

    # 3) Go straight back up
    arm.move_xyz(x, y, Z_FIXED, time_ms=TAP_UP_TIME)
    time.sleep(TAP_UP_TIME / 1000.0)

    # 4) Move away at safe height
    print("Moving off-screen...")
    arm.move_xyz(OFFSCREEN_X, OFFSCREEN_Y, OFFSCREEN_Z, time_ms=700)
    time.sleep(0.7)


def get_phone_crop(frame):
    """
    Crop the region where the phone screen is.
    Uses the same crop logic you had before.
    Returns (frame_step, crop_left, crop_top).
    """
    h, w, _ = frame.shape

    crop_left   = int(w * 0.35)
    crop_right  = int(w * 0.65)
    crop_top    = int(h * 0.05)
    crop_bottom = int(h * 0.90)

    frame_step = frame[crop_top:crop_bottom, crop_left:crop_right]

    return frame_step, crop_left, crop_top


def model_tap_once(arm, cap):
    """
    Capture one frame, run the PyTorch tap model on the phone crop,
    visualize the prediction, then move the arm to tap.
    """
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        return False

    # Flip like before (180°)
    frame = cv2.flip(frame, -1)

    # Crop to phone area
    frame_step, crop_left, crop_top = get_phone_crop(frame)

    # Use the tap model to predict tap position in the phone crop
    cx, cy = predict_tap(frame_step)

    print(f"Model predicted tap at phone-crop pixels ({cx:.1f}, {cy:.1f})")

    # Visualize prediction on the crop
    vis = frame_step.copy()
    cv2.circle(vis, (int(cx), int(cy)), 10, (0, 0, 255), -1)
    cv2.putText(
        vis,
        f"Pred tap: ({int(cx)}, {int(cy)})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Model View (phone crop)", vis)
    cv2.waitKey(1)

    # Map from phone-crop pixels to robot workspace
    h2, w2, _ = frame_step.shape
    x_world, y_world, z_world = pixel_to_world(cx, cy, w2, h2)

    # Tap with the arm
    tap(arm, x_world, y_world)

    return True


def main():
    # --- Init arm ---
    arm = MaxArmClient(PORT)
    print("Homing...")
    arm.home(2000)

    print("Moving to start (side) position...")
    arm.move_xyz(HOME_X, HOME_Y, HOME_Z, time_ms=1000)
    time.sleep(1.0)

    # --- Init camera ---
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # --- Load tap model ---
    load_tap_model()

    print("Preview running.")
    print("  's' = single model-based tap")
    print("  'a' = auto mode (several taps in a row)")
    print("  'q' = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, -1)

        # Show cropped phone area for preview
        frame_step, _, _ = get_phone_crop(frame)

        cv2.putText(
            frame_step,
            "Press 's' = single tap, 'a' = auto, 'q' = quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Preview (phone crop)", frame_step)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # One tap at current screen
        if key == ord('s'):
            print("\n[Single tap] Capturing and tapping using model...")
            model_tap_once(arm, cap)

        # Auto mode: do N taps in sequence using the model
        if key == ord('a'):
            print("\n[Auto mode] Running multiple taps...")
            NUM_STEPS = 15   # adjust as you like
            for i in range(NUM_STEPS):
                print(f"\nAuto step {i+1}/{NUM_STEPS}")
                ok = model_tap_once(arm, cap)
                if not ok:
                    print("Camera error, stopping auto mode.")
                    break
                # Give the phone time to animate to next screen
                time.sleep(2.0)

    cap.release()
    cv2.destroyAllWindows()
    arm.close()


if __name__ == "__main__":
    main()
