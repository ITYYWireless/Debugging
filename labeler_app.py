import os
import json
from io import BytesIO
from datetime import datetime

import streamlit as st
from PIL import Image, ImageDraw

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
    st.title("Phone Setup AI – Tap Point Labeler (Slider Version)")

    st.write(
        "Upload phone screen screenshots, choose where the robot should tap "
        "using X/Y sliders, and we'll store training data for a tap prediction model."
    )

    uploaded_files = st.file_uploader(
        "Upload screenshots",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    # Session state for images and index
    if "images" not in st.session_state:
        st.session_state.images = []
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    # When new files are uploaded, reset images list once
    if uploaded_files:
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
    img = st.session_state.images[idx]
    w, h = img.size

    st.write(f"Image {idx + 1} of {total}  |  size: {w} x {h}")

    # Optional screen label (e.g. 'google_terms', 'wifi_skip')
    screen_label = st.text_input(
        "Screen label (optional, e.g. 'google_terms')",
        key=f"screen_label_{idx}",
    )

    st.markdown("### Choose tap coordinates (pixels)")

    # Defaults: center of the image
    default_x = w // 2
    default_y = h // 2

    # Use session_state to remember choices per image if you navigate
    if f"x_{idx}" not in st.session_state:
        st.session_state[f"x_{idx}"] = default_x
    if f"y_{idx}" not in st.session_state:
        st.session_state[f"y_{idx}"] = default_y

    tap_x = st.slider(
        "Tap X (horizontal, 0 = left)",
        min_value=0,
        max_value=w - 1,
        value=st.session_state[f"x_{idx}"],
        key=f"x_slider_{idx}",
    )
    tap_y = st.slider(
        "Tap Y (vertical, 0 = top)",
        min_value=0,
        max_value=h - 1,
        value=st.session_state[f"y_{idx}"],
        key=f"y_slider_{idx}",
    )

    # Update stored values
    st.session_state[f"x_{idx}"] = tap_x
    st.session_state[f"y_{idx}"] = tap_y

    st.write(f"Selected tap coords: x = **{tap_x}**, y = **{tap_y}**")

    # Draw preview with a red dot
    preview_img = img.copy()
    draw = ImageDraw.Draw(preview_img)
    r = max(4, int(min(w, h) * 0.01))  # radius of dot
    left_up = (tap_x - r, tap_y - r)
    right_down = (tap_x + r, tap_y + r)
    draw.ellipse([left_up, right_down], fill="red")

    st.markdown("### Preview (with tap point in red)")
    st.image(preview_img, use_column_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬅ Previous", disabled=(idx == 0)):
            if idx > 0:
                st.session_state.current_index -= 1
                st.experimental_rerun()

    with col2:
        if st.button("Save label for this image"):
            img_path, label_path = save_image_and_label(
                img,
                tap_x=tap_x,
                tap_y=tap_y,
                screen_label=screen_label.strip() or None,
            )
            st.success(f"Saved label: {os.path.basename(label_path)}")

    with col3:
        if st.button("Next ➡", disabled=(idx == total - 1)):
            if idx < total - 1:
                st.session_state.current_index += 1
                st.experimental_rerun()

    st.markdown("---")
    st.write(
        "You can label all images in one go. When you're done, you'll have "
        "PNG images in `data/images/` and JSON labels in `data/labels/` "
        "ready for the training script."
    )


if __name__ == "__main__":
    main()
