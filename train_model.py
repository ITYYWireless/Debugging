import os
import json
from glob import glob

from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

# --- Paths (match labeler_app.py) ---
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


class TapDataset(Dataset):
    """
    Dataset that loads images and tap coordinates (normalized 0..1)
    from our label JSON files.
    """

    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.samples = []

        for label_path in glob(os.path.join(labels_dir, "*.json")):
            with open(label_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            img_filename = meta["image_filename"]
            img_path = os.path.join(images_dir, img_filename)

            if not os.path.exists(img_path):
                print(f"WARNING: image not found for label {label_path}")
                continue

            x_norm = float(meta["tap_x_norm"])
            y_norm = float(meta["tap_y_norm"])
            self.samples.append((img_path, (x_norm, y_norm)))

        print(f"Loaded {len(self.samples)} labeled samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, (x_norm, y_norm) = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        target = torch.tensor([x_norm, y_norm], dtype=torch.float32)  # (2,)
        return img, target


def main():
    # Basic image transforms for ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = TapDataset(IMAGES_DIR, LABELS_DIR, transform=transform)

    if len(dataset) < 2:
        print("Not enough data yet (need at least ~20+ labeled images).")
        return

    # Train/val split
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ResNet18 backbone with 2-output head (x_norm, y_norm)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.MSELoss()  # regression loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 15
    for epoch in range(EPOCHS):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)           # (B, 2)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        avg_train_loss = train_loss / train_size

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)

        avg_val_loss = val_loss / val_size
        print(f"Epoch {epoch+1}/{EPOCHS}  "
              f"train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}")

    # Save model weights
    save_path = os.path.join(MODELS_DIR, "tap_regressor.pt")
    torch.save(model.state_dict(), save_path)
    print("Saved model to:", save_path)


if __name__ == "__main__":
    main()
