import numpy as np
from PIL import Image
from hopfield_model import IMG_SIZE


def image_to_pattern(img_array: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    img = img_array.astype(float) / 255.0
    binary = np.where(img >= threshold, 1.0, -1.0)
    return binary.flatten()


def pattern_to_image(pattern: np.ndarray, shape) -> np.ndarray:
    pattern = np.asarray(pattern, dtype=float).flatten()
    img = pattern.reshape(shape)
    img = (img + 1.0) / 2.0
    img = img * 255.0
    return img.astype("uint8")


def load_image_as_array(path: str) -> np.ndarray:
    img = Image.open(path).convert("L").resize(IMG_SIZE)
    return np.array(img, dtype=float)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return int(np.sum(a != b))