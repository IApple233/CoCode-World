"""Explicit, hash-bound reference images for held-out appearance editing."""
from pathlib import Path

import numpy as np
from PIL import Image

from cocode_world.io import sha256_file

TARGET_REFERENCE = "target_split_screen_frame_000000"
EXTERNAL_REFERENCE = "external_benchmark_image"


def validate_external_reference(record):
    if record.get("reference_image_source") != EXTERNAL_REFERENCE:
        raise ValueError("Expected an explicitly declared external benchmark image")
    path = Path(record.get("reference_image_path", ""))
    digest = record.get("reference_image_sha256", "")
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"External reference must be an existing absolute path: {path}")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("External reference requires a SHA256 digest")
    return path


def load_external_reference(record):
    path = validate_external_reference(record)
    if sha256_file(path) != record["reference_image_sha256"]:
        raise ValueError(f"External reference hash mismatch: {path}")
    source_canvas = record.get("source_canvas", [record.get("height", 480), record.get("width", 1664)])
    expected_size = (int(source_canvas[1]), int(source_canvas[0]))
    if expected_size not in ((1664, 480), (2560, 720)):
        raise ValueError(f"Unsupported external reference canvas: {expected_size}")
    with Image.open(path) as image:
        if image.size != expected_size:
            raise ValueError(f"External reference must be {expected_size[0]}x{expected_size[1]}, got {image.size}")
        return np.array(image.convert("RGB"))
