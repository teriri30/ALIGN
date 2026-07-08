"""Canonical 2-D harvester geometry shared by display and footprint validation."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Tuple

import numpy as np


FOOTPRINT_MODEL = "planar_rigid_footprint_v2"
FOOTPRINT_REFERENCE_POINT = "track_contact_center"
FOOTPRINT_REQUIRED_KEYS = (
    "cutter_width_m",
    "cutter_forward_offset_m",
    "track_width_m",
    "track_gauge_m",
    "wheelbase_m",
    "track_length_m",
    "turn_radius_m",
)


def _positive(params: Mapping[str, Any], key: str) -> float:
    try:
        value = float(params[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be measured and explicitly configured") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be a finite positive value")
    return value


def _rectangle(x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    return np.asarray(
        ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
        dtype=np.float64,
    )


def planar_geometry_m(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Return local metre-space polygons around the track-contact centre.

    Local +x is the machine forward direction and +y is its right side.
    The route reference point is the midpoint between both track contact
    rectangles. This is a mathematical planar model, not a CAD envelope.
    """
    missing = [key for key in FOOTPRINT_REQUIRED_KEYS if key not in params]
    if missing:
        raise ValueError(
            "missing harvester geometry parameters: " + ", ".join(missing)
        )
    cutter_width = _positive(params, "cutter_width_m")
    cutter_forward_offset = _positive(params, "cutter_forward_offset_m")
    track_width = _positive(params, "track_width_m")
    track_gauge = float(params["track_gauge_m"])
    if not math.isfinite(track_gauge) or track_gauge < 0:
        raise ValueError("track_gauge_m must be finite and non-negative")
    wheelbase = _positive(params, "wheelbase_m")
    track_length = _positive(params, "track_length_m")
    turn_radius = _positive(params, "turn_radius_m")

    half_track_length = track_length * 0.5
    half_track_width = track_width * 0.5
    half_gauge = track_gauge * 0.5
    body_length = max(wheelbase, track_length)
    half_body_length = body_length * 0.5
    half_body_width = (track_gauge + track_width) * 0.5
    half_cutter_width = cutter_width * 0.5
    visual_header_depth = max(0.16, min(0.26, cutter_width * 0.10))
    visual_header_front = half_body_length + visual_header_depth

    return {
        "model": FOOTPRINT_MODEL,
        "reference_point": FOOTPRINT_REFERENCE_POINT,
        "left_track": _rectangle(
            -half_track_length, half_track_length,
            -half_gauge - half_track_width, -half_gauge + half_track_width,
        ),
        "right_track": _rectangle(
            -half_track_length, half_track_length,
            half_gauge - half_track_width, half_gauge + half_track_width,
        ),
        "body": _rectangle(
            -half_body_length, half_body_length,
            -half_body_width, half_body_width,
        ),
        "cutter_line": np.asarray(
            (
                (cutter_forward_offset, -half_cutter_width),
                (cutter_forward_offset, half_cutter_width),
            ),
            dtype=np.float64,
        ),
        "cutter_visual": _rectangle(
            half_body_length,
            visual_header_front,
            -half_cutter_width,
            half_cutter_width,
        ),
        "cutter_visual_edge": np.asarray(
            (
                (visual_header_front, -half_cutter_width),
                (visual_header_front, half_cutter_width),
            ),
            dtype=np.float64,
        ),
        "dimensions_m": {
            "cutter_width_m": cutter_width,
            "cutter_forward_offset_m": cutter_forward_offset,
            "track_width_m": track_width,
            "track_gauge_m": track_gauge,
            "wheelbase_m": wheelbase,
            "track_length_m": track_length,
            "turn_radius_m": turn_radius,
            "body_length_m": body_length,
            "body_width_m": track_gauge + track_width,
        },
    }


def transform_polygon(
    polygon_m: np.ndarray,
    center_px: Tuple[float, float],
    heading_rad: float,
    pixels_per_meter: float,
) -> np.ndarray:
    """Transform one local metre polygon into image pixel coordinates."""
    scale = float(pixels_per_meter)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("pixels_per_meter must be finite and positive")
    c, s = math.cos(float(heading_rad)), math.sin(float(heading_rad))
    rotation = np.asarray(((c, -s), (s, c)), dtype=np.float64)
    center = np.asarray(center_px, dtype=np.float64)
    return np.asarray(polygon_m, dtype=np.float64) * scale @ rotation.T + center


def planar_geometry_px(
    params: Mapping[str, Any],
    center_px: Tuple[float, float],
    heading_rad: float,
    pixels_per_meter: float,
) -> Dict[str, Any]:
    geometry = planar_geometry_m(params)
    result = {
        key: transform_polygon(value, center_px, heading_rad, pixels_per_meter)
        for key, value in geometry.items()
        if isinstance(value, np.ndarray)
    }
    result.update({
        "model": geometry["model"],
        "reference_point": geometry["reference_point"],
        "dimensions_m": geometry["dimensions_m"],
    })
    return result
