"""DECISION-005: reproducible GeoTIFF RGB and radiometric preprocessing."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


RASTER_PREPROCESSING_VERSION = "rgb-percentile-v1"


class EmptyRasterWindowError(ValueError):
    """The requested window contains no valid pixel in any selected RGB band."""


def _band_names(color_interpretations: Sequence[Any]) -> list[str]:
    names = []
    for item in color_interpretations or ():
        name = getattr(item, "name", None) or str(item)
        names.append(str(name).lower().split(".")[-1])
    return names


def select_rgb_band_indexes(count: int, color_interpretations: Sequence[Any] = ()) -> list[int]:
    count = int(count)
    if count <= 0:
        raise ValueError("raster has no bands")
    names = _band_names(color_interpretations)
    if all(name in names for name in ("red", "green", "blue")):
        return [names.index(name) + 1 for name in ("red", "green", "blue")]
    indexes = list(range(1, min(3, count) + 1))
    while len(indexes) < 3:
        indexes.append(indexes[-1])
    return indexes


def validate_preprocessing_config(config: Mapping[str, Any] | None) -> dict:
    cfg = dict(config or {})
    mode = str(cfg.get("mode", "percentile_per_band"))
    if mode not in {"percentile_per_band", "uint8_identity"}:
        raise ValueError(f"unsupported raster preprocessing mode: {mode}")
    lower = float(cfg.get("lower_percentile", 2.0))
    upper = float(cfg.get("upper_percentile", 98.0))
    if not 0.0 <= lower < upper <= 100.0:
        raise ValueError("raster preprocessing percentiles must satisfy 0 <= lower < upper <= 100")
    return {
        "mode": mode,
        "lower_percentile": lower,
        "upper_percentile": upper,
    }


def normalise_raster_bands(
    bands: np.ndarray,
    valid_masks: np.ndarray | None = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply the versioned, NoData-aware radiometric input contract.

    DECISION-005 requires native uint8 values to remain byte-exact and every
    non-uint8 scaling range to be recorded for later audit.
    """
    values = np.asarray(bands)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError("RGB raster data must have shape (3, height, width)")
    cfg = validate_preprocessing_config(config)
    if valid_masks is None:
        valid = np.ones(values.shape, dtype=bool)
    else:
        masks = np.asarray(valid_masks)
        if masks.shape != values.shape:
            raise ValueError("raster validity masks must match RGB band shape")
        valid = masks > 0
    valid &= np.isfinite(values)

    valid_counts = np.count_nonzero(valid, axis=(1, 2))
    if np.all(valid_counts == 0):
        raise EmptyRasterWindowError("RGB window contains no valid pixels")

    output = np.zeros(values.shape, dtype=np.uint8)
    band_metadata = []
    # Native 8-bit RGB is already in the model's expected numeric range.
    identity = values.dtype == np.uint8
    for index in range(3):
        source = values[index]
        band_valid = valid[index]
        samples = source[band_valid]
        if samples.size == 0:
            raise ValueError(f"RGB band {index + 1} contains no valid pixels")
        if identity:
            lower, upper = 0.0, 255.0
            scaled = source.astype(np.uint8, copy=True)
        else:
            lower, upper = np.percentile(
                samples.astype(np.float64, copy=False),
                [cfg["lower_percentile"], cfg["upper_percentile"]],
            )
            lower, upper = float(lower), float(upper)
            if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
                raise ValueError(f"RGB band {index + 1} has no usable radiometric range")
            scaled = np.clip(
                (source.astype(np.float64) - lower) * (255.0 / (upper - lower)),
                0.0,
                255.0,
            ).astype(np.uint8)
        scaled[~band_valid] = 0
        output[index] = scaled
        band_metadata.append({
            "band": index + 1,
            "lower": lower,
            "upper": upper,
            "valid_pixel_count": int(samples.size),
        })

    metadata = {
        "version": RASTER_PREPROCESSING_VERSION,
        "mode": "uint8_identity" if identity else cfg["mode"],
        "input_dtype": str(values.dtype),
        "lower_percentile": cfg["lower_percentile"],
        "upper_percentile": cfg["upper_percentile"],
        "bands": band_metadata,
    }
    return np.ascontiguousarray(np.moveaxis(output, 0, -1)), metadata


def read_rgb_raster(
    dataset,
    *,
    window=None,
    out_shape=None,
    resampling=None,
    config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict]:
    """Read RGB by declared color interpretation and preserve audit metadata.

    This is the only GeoTIFF-to-model RGB boundary under DECISION-005.
    """
    indexes = select_rgb_band_indexes(dataset.count, getattr(dataset, "colorinterp", ()))
    read_kwargs = {}
    if window is not None:
        read_kwargs["window"] = window
    if out_shape is not None:
        read_kwargs["out_shape"] = (3, int(out_shape[0]), int(out_shape[1]))
    if resampling is not None:
        read_kwargs["resampling"] = resampling
    bands = dataset.read(indexes, **read_kwargs)
    masks = dataset.read_masks(indexes, **read_kwargs)
    rgb, metadata = normalise_raster_bands(bands, masks, config)
    metadata.update({
        "source_band_indexes": indexes,
        "source_color_interpretation": _band_names(getattr(dataset, "colorinterp", ())),
        "source_nodata": [
            None if value is None or not np.isfinite(float(value)) else float(value)
            for value in (getattr(dataset, "nodatavals", ()) or ())
        ],
    })
    return rgb, metadata
