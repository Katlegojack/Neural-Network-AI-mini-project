import base64
import gzip
import io
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from django.conf import settings
from PIL import Image, ImageOps


MODEL_FILES = {
    "naive": "naive_sequential_mnist.pt",
    "ewc": "ewc_mnist.pt",
    "replay": "experience_replay_mnist.pt",
}

MODEL_LABELS = {
    "naive": "Naive Sequential",
    "ewc": "EWC",
    "replay": "Experience Replay",
}


class MNISTNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


def model_dir() -> Path:
    return settings.REPO_ROOT / "models"


def missing_models():
    return [
        MODEL_LABELS[key]
        for key, filename in MODEL_FILES.items()
        if not (model_dir() / filename).exists()
    ]


@lru_cache(maxsize=3)
def load_model(model_key: str):
    path = model_dir() / MODEL_FILES[model_key]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model checkpoint: {path}. "
            "Run 'python manage.py prepare_models' first."
        )

    model = MNISTNetwork()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _crop_and_center(image: Image.Image) -> Image.Image:
    arr = np.asarray(image, dtype=np.uint8)
    mask = arr > 20

    if not mask.any():
        return image.resize((28, 28), Image.Resampling.LANCZOS)

    ys, xs = np.where(mask)
    left, right = xs.min(), xs.max() + 1
    top, bottom = ys.min(), ys.max() + 1

    cropped = image.crop((left, top, right, bottom))

    width, height = cropped.size
    scale = 20.0 / max(width, height)
    resized = cropped.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("L", (28, 28), color=0)
    x = (28 - resized.width) // 2
    y = (28 - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def preprocess_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file).convert("L")

    # MNIST uses a bright digit on a dark background.
    # Most uploaded handwritten images use the opposite convention.
    if np.asarray(image).mean() > 127:
        image = ImageOps.invert(image)

    image = _crop_and_center(image)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor, image


def _load_idx_images(path: Path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Invalid MNIST image file.")
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def _load_idx_labels(path: Path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid MNIST label file.")
        return np.frombuffer(f.read(), dtype=np.uint8)


def get_mnist_sample(digit: int):
    images = _load_idx_images(settings.REPO_ROOT / "t10k-images-idx3-ubyte.gz")
    labels = _load_idx_labels(settings.REPO_ROOT / "t10k-labels-idx1-ubyte.gz")

    index = int(np.where(labels == digit)[0][0])
    arr = images[index]
    image = Image.fromarray(arr, mode="L")
    tensor = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return tensor, image


def image_to_data_url(image: Image.Image) -> str:
    enlarged = image.resize((280, 280), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    enlarged.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@torch.no_grad()
def predict(model_key: str, tensor: torch.Tensor):
    model = load_model(model_key)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]

    confidence, prediction = torch.max(probabilities, dim=0)
    top_values, top_indices = torch.topk(probabilities, k=3)

    return {
        "model_key": model_key,
        "model_name": MODEL_LABELS[model_key],
        "prediction": int(prediction.item()),
        "confidence": float(confidence.item() * 100),
        "top3": [
            {
                "digit": int(idx.item()),
                "confidence": float(value.item() * 100),
            }
            for value, idx in zip(top_values, top_indices)
        ],
    }
