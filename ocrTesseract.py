import cv2
import pytesseract

# ---------------- CONFIG ----------------

# Path to Tesseract on Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

TARGET_TEXT = "settings"
MIN_CONF = 50  # confidence threshold for displaying text


# ---------------- ZOOM ----------------

def apply_zoom(frame, zoom_factor: float):
    """
    Simple digital zoom around center.
    zoom_factor = 1.0 -> no zoom
    """
    if zoom_factor <= 1.0:
        return frame

    h, w = frame.shape[:2]
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)

    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    cropped = frame[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return zoomed


# ---------------- OCR ----------------

def run_ocr_on_frame(frame):
    """
    Run OCR on the entire frame and draw bounding boxes.
    Highlights the word "settings" if detected.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Slight upscale to help Tesseract see small text
    scale = 1.0
    gray_up = cv2.resize(
        gray,
        (int(gray.shape[1] * scale), int(gray.shape[0] * scale)),
        interpolation=cv2.INTER_CUBIC
    )

    gray_up = cv2.bilateralFilter(gray_up, 9, 75, 75)
    gray_up = cv2.adaptiveThreshold(
        gray_up,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    config = (
        r"--oem 3 --psm 6 "
        r"-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    )

    data = pytesseract.image_to_data(
        gray_up,
        output_type=pytesseract.Output.DICT,
        config=config
    )

    annotated = frame.copy()
    found_target = False

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except ValueError:
            conf = -1

        if conf < MIN_CONF:
            continue

        x = int(data["left"][i] / scale)
        y = int(data["top"][i] / scale)
        w = int(data["width"][i] / scale)
        h = int(data["height"][i] / scale)

        if TARGET_TEXT.lower() in text.lower():
            color = (0, 255, 0)
            found_target = True
        else:
            color = (0, 165, 255)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            annotated,
            f"{text} ({int(conf)})",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    if found_target:
        print("✅ 'Settings' detected in this screenshot.")
    else:
        print("❌ 'Settings' NOT detected in this screenshot.")

    return annotated


# ---------------- MAIN LOOP ----------------

def main():
    # Try different indexes if 0 isn't your phone / webcam.
    # Force DirectShow backend to avoid MSMF issues.
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Could not open camera at index 0 with CAP_DSHOW.")
        print("   Try changing VideoCapture(0, ...) to 1 or 2, etc.")
        return

    # Start with safer 720p. Once stable, you can try 1920x1080 again.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    zoom_factor = 1.0
    failed_grabs = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            failed_grabs += 1
            if failed_grabs == 1:
                print("⚠ Failed to grab frame from camera.")
            if failed_grabs > 30:
                print("❌ Too many failed frames, exiting.")
                break
            continue
        else:
            failed_grabs = 0

        zoomed = apply_zoom(frame, zoom_factor)

        hud = zoomed.copy()
        cv2.putText(
            hud,
            f"Zoom: {zoom_factor:.1f}x | q=quit  c=capture+OCR  z=zoom+  x=zoom-",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Live Camera", hud)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("z"):
            zoom_factor = min(3.0, zoom_factor + 0.2)

        if key == ord("x"):
            zoom_factor = max(1.0, zoom_factor - 0.2)

        if key == ord("c"):
            screenshot = zoomed.copy()
            annotated = run_ocr_on_frame(screenshot)
            cv2.imshow("OCR Result", annotated)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
