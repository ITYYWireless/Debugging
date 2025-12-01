"""
tap_model.py

Load a trained PyTorch model that predicts where to tap on a phone
screenshot, and provide a helper to get tap coordinates from an
OpenCV frame (BGR numpy array).
"""

import os
from typing import Tuple

import cv2
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

# If tap_model.py is in your project root, this is fine:
BASE_DIR = os.path.dirname(__file__)

# If tap_model.py is inside a "robot" folder, uncomment this instead:
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "tap_regressor.pt")

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------

tap_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tap_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

tap_model: nn.Module | None = None  # will be initialized by load_tap_model()


# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------

def load_tap_model() -> None:
    """
    Load the trained tap regressor model from disk into global tap_model.

    Call this once at program startup, before calling predict_tap().
    """
    global tap_model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Tap model file not found at {MODEL_PATH}. "
            f"Make sure you trained it (train_tap_model.py) "
            f"and that the path is correct."
        )

    print(f"[tap_model] Loading model from {MODEL_PATH} on device {tap_device}...")

    # Create the same architecture we used in training
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state = torch.load(MODEL_PATH, map_location=tap_device)
    model.load_state_dict(state)

    model.to(tap_device)
    model.eval()

    tap_model = model
    print("[tap_model] Model loaded and ready.")


# -----------------------------------------------------------------------------
# Predict tap point
# -----------------------------------------------------------------------------

def predict_tap(frame_step) -> Tuple[float, float]:
    """
    Predict tap coordinates for a given phone crop.

    Parameters
    ----------
    frame_step : np.ndarray
        OpenCV BGR image (H, W, 3) of the phone region.

    Returns
    -------
    (cx, cy) : tuple of float
        Predicted tap position in pixel coordinates of `frame_step`.
    """
    if tap_model is None:
        raise RuntimeError(
            "tap_model is not loaded. Call load_tap_model() once at startup."
        )

    # Convert BGR (OpenCV) -> RGB (PIL) for torchvision transforms
    rgb = cv2.cvtColor(frame_step, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    x = tap_transform(pil_img).unsqueeze(0).to(tap_device)  # (1, 3, 224, 224)

    with torch.no_grad():
        out = tap_model(x)[0]  # shape: (2,)
        x_norm, y_norm = out.tolist()

    # Clamp to [0, 1] just in case
    x_norm = max(0.0, min(1.0, x_norm))
    y_norm = max(0.0, min(1.0, y_norm))

    h, w, _ = frame_step.shape
    cx = x_norm * w
    cy = y_norm * h

    return cx, cy


# -----------------------------------------------------------------------------
# Optional: quick CLI test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Simple test: load a random image from data/images and print predicted coords.
    Adjust paths as needed if you want to use this.
    """
    import glob

    load_tap_model()

    images_dir = os.path.join(BASE_DIR, "data", "images")
    paths = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(
        os.path.join(images_dir, "*.jpg")
    )

    if not paths:
        print("[tap_model] No images found in data/images for test.")
    else:
        test_path = paths[0]
        print(f"[tap_model] Testing on {test_path}")
        img = cv2.imread(test_path)
        cx, cy = predict_tap(img)
        print(f"[tap_model] Predicted tap at (cx={cx:.1f}, cy={cy:.1f})")
