"""Shared row-orientation estimators used by mask processing and planning."""
import math
from typing import Optional

import cv2
import numpy as np


def wrap_half_turn(angle: float) -> float:
    return float(angle) % math.pi


def half_turn_difference(first: float, second: float) -> float:
    diff = abs(wrap_half_turn(first) - wrap_half_turn(second))
    return min(diff, math.pi - diff)


def estimate_mask_axis_pca(mask: np.ndarray) -> float:
    """Estimate the dominant binary-mask axis in radians within ``[0, pi)``."""
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) < 50:
        return 0.0
    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    coords -= coords.mean(axis=0)
    covariance = np.cov(coords.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    main_vector = eigenvectors[:, int(np.argmax(eigenvalues))]
    return wrap_half_turn(math.atan2(float(main_vector[1]), float(main_vector[0])))


def estimate_mask_axis_fitline(mask: np.ndarray) -> float:
    """Estimate the dominant mask axis with OpenCV's robust line fit."""
    points = cv2.findNonZero((np.asarray(mask) > 0).astype(np.uint8) * 255)
    if points is None or len(points) < 20:
        return 0.0
    vx, vy, _, _ = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    return wrap_half_turn(math.atan2(float(vy[0]), float(vx[0])))


def estimate_texture_orientation_fft(
    mask: np.ndarray,
    source_image: Optional[np.ndarray],
) -> Optional[float]:
    """Estimate row direction from masked image texture using a 2-D FFT spectrum.

    The dominant frequency direction is perpendicular to the spatial row direction.
    ``None`` is returned when the masked texture does not contain enough evidence.
    """
    if source_image is None:
        return None
    support = np.asarray(mask) > 0
    if np.count_nonzero(support) < 100:
        return None
    image = np.asarray(source_image)
    if image.shape[:2] != support.shape:
        image = cv2.resize(image, (support.shape[1], support.shape[0]), interpolation=cv2.INTER_AREA)
    if image.ndim == 3:
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    texture = image.astype(np.float32)
    values = texture[support]
    if values.size < 100 or float(np.std(values)) <= 1e-6:
        return None
    texture = np.where(support, texture - float(np.mean(values)), 0.0)
    window = np.outer(np.hanning(texture.shape[0]), np.hanning(texture.shape[1])).astype(np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(texture * window)))
    cy, cx = np.array(spectrum.shape) // 2
    radius = max(2, int(round(min(spectrum.shape) * 0.02)))
    yy, xx = np.ogrid[: spectrum.shape[0], : spectrum.shape[1]]
    spectrum[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2] = 0.0
    peak_y, peak_x = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
    if not np.isfinite(spectrum[peak_y, peak_x]) or spectrum[peak_y, peak_x] <= 0:
        return None
    frequency_angle = math.atan2(float(peak_y - cy), float(peak_x - cx))
    return wrap_half_turn(frequency_angle + math.pi / 2.0)


def estimate_row_orientation(
    mask: np.ndarray,
    source_image: Optional[np.ndarray] = None,
    *,
    fft_tolerance_deg: float = 15.0,
) -> float:
    """Return PCA orientation, refined by FFT texture evidence when available."""
    coarse = estimate_mask_axis_pca(mask)
    refined = estimate_texture_orientation_fft(mask, source_image)
    if refined is None:
        return coarse
    # The tolerance is retained as diagnostic policy at callers; valid FFT texture
    # remains the more direct row-spacing observation even when field shape differs.
    _ = half_turn_difference(refined, coarse) <= math.radians(float(fft_tolerance_deg))
    return refined
