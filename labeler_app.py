import os
import json
from io import BytesIO
from datetime import datetime

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)


def save_image_and_label(img: Image.Image, tap_x, tap_y, screen_label: str | None = None):
    """
    Save the image and label metadata.
    tap_x, tap_y are in original image pixel coordinates.
    We also save normalized coords [0,1] for training.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_filename = f"{ts}.png"
    img_path = os.path.join(IMAGES_DIR, img_filename)
    label_path = os.path.join(LABELS_DIR, f"{ts}.json")

    # Save image
    img.save(img_path)

    w, h = img.size
    label = {
        "image_filename": img_filename,
        "tap_x": float(tap_x),
        "tap_y": float(tap_y),
        "tap_x_norm": float(tap_x / w),
        "tap_y_norm": float(tap_y / h),
        "img_width": int(w),
        "img_height": int(h),
        "screen_label": screen_label or "",
    }

    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(label, f, indent=2)

    return img_path, label_path


def main():
    st.title("Phone Setup AI – Click-to-Tap Labeler")

    st.write(
        "Upload phone screen screenshots, click where the robot should tap, "
        "and we’ll store training data for a tap prediction model."
    )

    uploaded_files = st.file_uploader(
        "Upload screenshots",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "images" not in st.session_state:
        st.session_state.images = []

    # When new files are uploaded, reset state
    if uploaded_files:
        # Only convert to images once
        if not st.session_state.images:
            for uf in uploaded_files:
                image = Image.open(BytesIO(uf.read())).convert("RGB")
                st.session_state.images.append(image)
            st.session_state.current_index = 0

    if not st.session_state.images:
        st.info("Upload some images to get started.")
        return

    idx = st.session_state.current_index
    total = len(st.session_state.images)
    st.write(f"Image {idx + 1} of {total}")

    img = st.session_state.images[idx]

    # Optional: enter screen label (e.g. 'google_terms', 'wifi_skip')
    screen_label = st.text_input("Screen label (optional, e.g. 'google_terms')", key=f"label_{idx}")

    st.write("Click once where you want the robot to tap (draw a point).")

    w, h = img.size

    # Use drawable canvas with a point tool
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 1.0)",
        stroke_width=5,
        stroke_color="#ff0000",
        background_image=img,
        update_streamlit=True,
        height=h,
        width=w,
        drawing_mode="point",
        key=f"canvas_{idx}",
    )

    tap_coords = None
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        if objects:
            # Use the last drawn point
            last_obj = objects[-1]
            # For a point, 'left' and 'top' are its coordinates
            tap_x = last_obj.get("left", None)
            tap_y = last_obj.get("top", None)
            if tap_x is not None and tap_y is not None:
                tap_coords = (tap_x, tap_y)
                st.write(f"Selected tap coords (pixels): x={tap_x:.1f}, y={tap_y:.1f}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Save label & Next"):
            if tap_coords is None:
                st.warning("Click on the image to select a tap point before saving.")
            else:
                tap_x, tap_y = tap_coords
                img_path, label_path = save_image_and_label(
                    img,
                    tap_x=tap_x,
                    tap_y=tap_y,
                    screen_label=screen_label.strip() or None,
                )
                st.success(f"Saved label: {os.path.basename(label_path)}")

                if idx < total - 1:
                    st.session_state.current_index += 1
                else:
                    st.success("All uploaded images have been labeled.")
    with col2:
        if st.button("Skip this image"):
            if idx < total - 1:
                st.session_state.current_index += 1
            else:
                st.success("No more images.")


if __name__ == "__main__":
    main()
