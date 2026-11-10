"""Frame comparison: exact pixel equality, difference stats and SSIM (numpy only)."""

import io

import numpy as np
from PIL import Image


def load_gray(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=np.float64)


def load_rgb(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.int16)


def _box(x: np.ndarray, k: int) -> np.ndarray:
    """Mean over k x k windows ('valid' region) via 2D cumulative sums."""
    c = np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


def ssim(a: np.ndarray, b: np.ndarray, win: int = 7, data_range: float = 255.0) -> float:
    """Mean SSIM (Wang et al. 2004) with a uniform win x win window on grayscale images."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_a, mu_b = _box(a, win), _box(b, win)
    var_a = _box(a * a, win) - mu_a ** 2
    var_b = _box(b * b, win) - mu_b ** 2
    cov = _box(a * b, win) - mu_a * mu_b
    n = win * win
    var_a, var_b, cov = var_a * n / (n - 1), var_b * n / (n - 1), cov * n / (n - 1)
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(s.mean())


def compare(png_a: bytes, png_b: bytes) -> dict:
    """Identity, pixel-difference stats and SSIM between two PNG frames."""
    ra, rb = load_rgb(png_a), load_rgb(png_b)
    if ra.shape != rb.shape:
        return {"same_size": False, "identical": False, "ssim": 0.0}
    d = np.abs(ra - rb)
    ident = bool(d.max() == 0)
    return {
        "same_size": True,
        "identical": ident,
        "n_diff_pixels": int((d.max(axis=2) > 0).sum()),
        "max_abs_diff": int(d.max()),
        "mean_abs_diff": float(d.mean()),
        "ssim": 1.0 if ident else ssim(load_gray(png_a), load_gray(png_b)),
    }


def changed(png_a: bytes, png_b: bytes, min_diff_pixels: int = 50) -> bool:
    """True when the frames differ visibly (more than a handful of pixels)."""
    c = compare(png_a, png_b)
    return (not c["same_size"]) or c["n_diff_pixels"] >= min_diff_pixels
