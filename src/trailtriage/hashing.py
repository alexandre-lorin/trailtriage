from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageOps


def features(path: Path) -> dict:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((512, 512))
        phash = str(imagehash.phash(image))
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        laplacian = (
            -4 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        return {
            "phash": phash,
            "sharpness_score": float(laplacian.var()) if laplacian.size else 0.0,
            "contrast": float(gray.std()),
        }


def distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()
