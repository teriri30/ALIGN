"""DECISION-012 canonical pose sampling for validation and animation."""

from __future__ import annotations

import math
from typing import Callable, Iterable, Tuple

import numpy as np


Point = Tuple[float, float]
DistanceFunction = Callable[[Point, Point], float]


def build_polyline_metric(points: Iterable[Point], distance_fn: DistanceFunction):
    values = np.asarray(list(points), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or len(values) < 2:
        raise ValueError("a path segment requires at least two 2-D points")
    values = values[:, :2]
    measured = []
    kept = [values[0]]
    for point in values[1:]:
        previous = kept[-1]
        distance = float(distance_fn(tuple(previous), tuple(point)))
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("path distance must be finite and non-negative")
        if distance <= 1e-9:
            continue
        kept.append(point)
        measured.append(distance)
    if len(kept) < 2:
        raise ValueError("path segment has no measurable length")
    values = np.asarray(kept, dtype=np.float64)
    lengths = np.asarray(measured, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return values, lengths, cumulative


def point_at_distance(points, lengths, cumulative, distance: float) -> np.ndarray:
    distance = max(0.0, min(float(distance), float(cumulative[-1])))
    index = int(np.searchsorted(cumulative, distance, side="right") - 1)
    index = max(0, min(index, len(lengths) - 1))
    length = float(lengths[index])
    ratio = 0.0 if length <= 1e-9 else (
        distance - float(cumulative[index])
    ) / length
    return points[index] + ratio * (points[index + 1] - points[index])


def pose_at_distance(
    points,
    lengths,
    cumulative,
    distance: float,
    heading_window: float,
    *,
    reversing: bool = False,
):
    """Return the canonical centre position and tangent heading."""
    distance = max(0.0, min(float(distance), float(cumulative[-1])))
    window = max(1e-6, float(heading_window))
    center = point_at_distance(points, lengths, cumulative, distance)
    before = point_at_distance(
        points, lengths, cumulative, max(0.0, distance - window)
    )
    after = point_at_distance(
        points, lengths, cumulative, min(float(cumulative[-1]), distance + window)
    )
    direction = after - before
    heading = math.atan2(float(direction[1]), float(direction[0]))
    if reversing:
        heading = (heading + math.pi) % (2.0 * math.pi)
    return float(center[0]), float(center[1]), float(heading)


def sample_polyline_poses(
    points: Iterable[Point],
    distance_fn: DistanceFunction,
    spacing: float,
    heading_window: float,
    *,
    reversing: bool = False,
):
    values, lengths, cumulative = build_polyline_metric(points, distance_fn)
    total = float(cumulative[-1])
    step = max(1e-6, float(spacing))
    distances = list(np.arange(0.0, total, step))
    if not distances or total - distances[-1] > 1e-9:
        distances.append(total)
    return [
        pose_at_distance(
            values, lengths, cumulative, distance, heading_window,
            reversing=reversing,
        )
        for distance in distances
    ]
