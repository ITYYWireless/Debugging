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

    return img_path, label_path, label


def main():
    st.title("Phone Setup AI – Click-to-Tap Labeler")

    st.write(
        "Upload phone screenshots, **click once** where the robot should tap, "
        "and we'll save images + tap coordinates for training."
    )

    uploaded_files = st.file_uploader(
        "Upload screenshots",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if "images" not in st.session_state:
        st.session_state.images = []
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    # Load uploaded files into memory once
    if uploaded_files and not st.session_state.images:
        for uf in uploaded_files:
            image = Image.open(BytesIO(uf.read())).convert("RGB")
            st.session_state.images.append(image)
        st.session_state.current_index = 0

    if not st.session_state.images:
        st.info("Upload some images to get started.")
        return

    idx = st.session_state.current_index
    total = len(st.session_state.images)
    img = st.session_state.images[idx]
    w, h = img.size

    st.write(f"Image {idx + 1} of {total}  |  size: {w} x {h}")

    screen_label = st.text_input(
        "Screen label (optional, e.g. 'StartIOS14', 'GoogleTerms')",
        key=f"screen_label_{idx}",
    )

    st.markdown("### Click on the image once to choose tap point")

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 1.0)",
        stroke_width=5,
        stroke_color="#ff0000",
        background_image=img,
        update_streamlit=True,
        height=h,
        width=w,
        drawing_mode="point",      # we just want points
        key=f"canvas_{idx}",
    )

    tap_x = tap_y = None
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        if objects:
            last_obj = objects[-1]
            tap_x = last_obj.get("left", None)
            tap_y = last_obj.get("top", None)

    if tap_x is not None and tap_y is not None:
        st.write(f"Selected tap coords: x = **{tap_x:.1f}**, y = **{tap_y:.1f}**")
    else:
        st.warning("Click once on the image to set the tap point.")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬅ Previous", disabled=(idx == 0)):
            if idx > 0:
                st.session_state.current_index -= 1
                st.experimental_rerun()

    with col2:
        if st.button("Save label for this image"):
            if tap_x is None or tap_y is None:
                st.error("You need to click on the image first.")
            else:
                img_path, label_path, label = save_image_and_label(
                    img,
                    tap_x=tap_x,
                    tap_y=tap_y,
                    screen_label=screen_label.strip() or None,
                )
                st.success(f"Saved label: {os.path.basename(label_path)}")
                st.json(label)

    with col3:
        if st.button("Next ➡", disabled=(idx == total - 1)):
            if idx < total - 1:
                st.session_state.current_index += 1
                st.experimental_rerun()

    st.markdown("---")
    st.write("Images go to:")
    st.code(IMAGES_DIR)
    st.write("Labels (JSON) go to:")
    st.code(LABELS_DIR)


if __name__ == "__main__":
    main()
