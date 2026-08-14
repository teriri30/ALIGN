"""Reproducible paper experiment plans, metrics, and artifact export."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np

from config import save_json_atomic


EXPERIMENT_SCHEMA = "reui.paper-experiment.v1"


def default_experiment_plan(model_paths: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Freeze one input while comparing the currently implemented mask stages.

    Turn geometry is implemented and can be exercised by the planner, but turn
    optimisation/strategy comparison is outside the current paper scope. The
    runner still accepts explicit ``path_cases`` for future offline studies;
    the default plan deliberately contains none.
    """
    path_cases: list[Dict[str, Any]] = []
    return {
        "mask_cases": [
            {"id": "raw_mask_geometry", "label": "raw mask geometry", "mode": "raw"},
            {"id": "band_geometry_reconstruction", "label": "band geometry reconstruction", "mode": "regularized_body"},
            {"id": "body_residual_recovery", "label": "body residual recovery", "mode": "body_residual"},
            {"id": "full_mask_regularization", "label": "full mask regularization", "mode": "full"},
        ],
        "path_cases": path_cases,
        # DECISION-014: source-image model comparisons remain disabled.
        "model_cases": [],
        "model_comparison": {"status": "disabled", "reason": "independent_model_inference_not_implemented"},
        "path_comparison": {
            "status": "disabled",
            "reason": "turn_strategy_experiment_not_in_current_scope",
        },
        "comparison_policy": {
            "same_inputs_required": True,
            "supervised_metrics_require_ground_truth": True,
            "failed_cases_are_preserved": True,
            "interactive_state_is_read_only": True,
            "path_strategy_comparison": "disabled",
        },
    }


def segmentation_metrics(predicted: np.ndarray, ground_truth: Optional[np.ndarray]) -> Dict[str, float]:
    if ground_truth is None:
        return {}
    pred = np.asarray(predicted) > 0
    truth = np.asarray(ground_truth) > 0
    if pred.shape != truth.shape:
        raise ValueError("ground-truth mask shape does not match prediction")
    intersection = int(np.count_nonzero(pred & truth))
    union = int(np.count_nonzero(pred | truth))
    pred_count = int(np.count_nonzero(pred))
    truth_count = int(np.count_nonzero(truth))
    return {
        "intersection_over_union": intersection / union if union else 1.0,
        "dice": 2.0 * intersection / (pred_count + truth_count) if pred_count + truth_count else 1.0,
        "precision": intersection / pred_count if pred_count else (1.0 if truth_count == 0 else 0.0),
        "recall": intersection / truth_count if truth_count else 1.0,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _case_dir_name(case_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_id)).strip("._")
    return cleaned or "case"


def _write_image(path: Path, image: np.ndarray, *, rgb: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image)
    if array.ndim == 2:
        array = (array > 0).astype(np.uint8) * 255
    elif rgb:
        array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), array):
        raise OSError(f"failed to write image: {path}")


def _metric_rows(records: Sequence[Mapping[str, Any]]) -> list[dict]:
    rows = []
    for record in records:
        base = {
            "case_id": str(record.get("case_id", "")),
            "module": str(record.get("module", "")),
            "status": str(record.get("status", "")),
            "error": str(record.get("error", "") or ""),
        }
        metrics = record.get("metrics") or {}
        if not metrics:
            rows.append({**base, "metric": "", "value": "", "unit": ""})
        for metric, value in metrics.items():
            if isinstance(value, (int, float, np.number)) and np.isfinite(float(value)):
                rows.append({**base, "metric": str(metric), "value": value, "unit": _metric_unit(metric)})
    return rows


def _metric_unit(metric: str) -> str:
    if metric.endswith("_pct"):
        return "%"
    if metric.endswith("_m2"):
        return "m²"
    if metric.endswith("_m"):
        return "m"
    if metric.endswith("_s") or metric == "runtime_seconds":
        return "s"
    return ""


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _segment_metric_rows(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[str]]:
    """Flatten canonical footprint segment metrics for audit-friendly export."""
    rows = []
    metric_names = set()
    for record in records:
        for segment in record.get("segment_metrics") or []:
            if not isinstance(segment, Mapping):
                continue
            row = {
                "case_id": str(record.get("case_id", "")),
                "module": str(record.get("module", "")),
                "status": str(record.get("status", "")),
                "error": str(record.get("error", "") or ""),
                "segment_index": segment.get("segment_index", ""),
                "segment_type": str(segment.get("segment_type", "")),
            }
            for key, value in segment.items():
                if key in {"segment_index", "segment_type"}:
                    continue
                if isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and np.isfinite(float(value)):
                    row[str(key)] = float(value)
                    metric_names.add(str(key))
            rows.append(row)
    fields = ["case_id", "module", "status", "error", "segment_index", "segment_type", *sorted(metric_names)]
    return rows, fields

_REQUIRED_SEGMENT_TYPES = (
    "work", "turn_approach", "turn", "turn_reverse", "turn_aux",
    "entry", "exit", "unload",
)
_SEGMENT_SUMMARY_METRICS = (
    "length_m",
    "track_swept_area_m2",
    "track_body_overlap_area_m2",
    "track_headland_overlap_area_m2",
    "track_uncertain_overlap_area_m2",
    "track_forbidden_overlap_area_m2",
    "outside_field_area_m2",
    "turn_cost_m2",
    "agronomic_damage_cost_m2",
)

def _segment_type_summary_metrics(segment_metrics: Sequence[Mapping[str, Any]]) -> dict:
    """Aggregate canonical segment metrics with explicit zero-filled categories."""
    summary = {}
    segments = [item for item in segment_metrics or [] if isinstance(item, Mapping)]
    for segment_type in _REQUIRED_SEGMENT_TYPES:
        matching = [
            item for item in segments
            if str(item.get("segment_type", "")) == segment_type
        ]
        prefix = f"{segment_type}_"
        for metric in _SEGMENT_SUMMARY_METRICS:
            values = [item.get(metric) for item in matching]
            if not matching:
                summary[prefix + metric] = 0.0
                continue
            if any(value is None for value in values):
                summary[prefix + metric] = None
                continue
            numeric = []
            for value in values:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(number):
                    numeric.append(number)
            summary[prefix + metric] = float(sum(numeric))
        track_area = summary[prefix + "track_swept_area_m2"]
        outside_area = summary[prefix + "outside_field_area_m2"]
        summary[prefix + "outside_field_pct"] = (
            float(outside_area / track_area * 100.0)
            if isinstance(track_area, (int, float))
            and isinstance(outside_area, (int, float))
            and track_area > 0.0
            else (None if outside_area is None or track_area is None else 0.0)
        )
    return summary


def _wide_rows(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[str]]:
    metric_names = sorted({str(key) for record in records for key in (record.get("metrics") or {})})
    fields = ["case_id", "module", "status", "error", *metric_names]
    rows = []
    for record in records:
        row = {name: "" for name in fields}
        row.update({
            "case_id": record.get("case_id", ""),
            "module": record.get("module", ""),
            "status": record.get("status", ""),
            "error": record.get("error", "") or "",
        })
        for name, value in (record.get("metrics") or {}).items():
            if isinstance(value, (int, float, np.number)) and np.isfinite(float(value)):
                row[str(name)] = value
        rows.append(row)
    return rows, fields


def _draw_metric_figure(path: Path, records: Sequence[Mapping[str, Any]], module: str) -> None:
    selected = [record for record in records if record.get("module") == module and record.get("status") == "completed"]
    metric_names = [name for name in ("intersection_over_union", "dice", "crop_mask_geometric_coverage_pct", "rolling_canopy_pct", "total_length_m", "runtime_seconds") if any(name in (record.get("metrics") or {}) for record in selected)]
    width, row_height = 1100, 52
    height = max(180, 90 + row_height * max(1, len(selected)) * max(1, len(metric_names)))
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, f"{module.upper()} COMPARISON", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    y = 85
    for metric in metric_names:
        values = [float((record.get("metrics") or {}).get(metric, 0.0)) for record in selected]
        maximum = max(values) if values else 1.0
        maximum = maximum if maximum > 0 else 1.0
        cv2.putText(canvas, metric, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        y += 24
        for record, value in zip(selected, values):
            label = str(record.get("case_id", ""))[:28]
            cv2.putText(canvas, label, (45, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
            bar_x = 285
            bar_width = int(650 * value / maximum)
            cv2.rectangle(canvas, (bar_x, y), (bar_x + max(1, bar_width), y + 22), (70, 110, 180), -1)
            cv2.putText(canvas, f"{value:.4g}", (950, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            y += row_height
        y += 12
    if not metric_names:
        cv2.putText(canvas, "No completed numeric records", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 1, cv2.LINE_AA)
    _write_image(path, canvas)


def _snapshot_state(state) -> SimpleNamespace:
    names = (
        "harvester_params", "turn_strategy", "field_boundary", "field_area_m2",
        "forbidden_regions", "forbidden_regions_confirmed", "entry_point",
        "exit_point", "unload_points", "mask_offset_x", "mask_offset_y",
        "mask_raw",
        "source_sha256", "current_model_path", "model_sha256",
    )
    values = {}
    for name in names:
        value = getattr(state, name, None)
        if isinstance(value, np.ndarray):
            value = value.copy()
        else:
            value = deepcopy(value)
        values[name] = value
    return SimpleNamespace(**values)


def _numeric_metrics(values: Mapping[str, Any]) -> Dict[str, float]:
    result = {}
    for key, value in values.items():
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            number = float(value)
            if np.isfinite(number):
                result[str(key)] = number
    return result


def _raw_mask_result(raw_mask: np.ndarray, processed_result: Mapping[str, Any]) -> Dict[str, Any]:
    """Raw-mask baseline reuses detected geometry but removes regularized area changes."""
    result = dict(processed_result)
    raw = (np.asarray(raw_mask) > 0).astype(np.uint8) * 255
    result["processed_mask"] = raw
    result["headland_mask"] = np.zeros_like(raw)
    result["uncertain_residual_mask"] = np.zeros_like(raw)
    result["planning_support_mask"] = raw
    diagnostics = dict(result.get("diagnostics") or {})
    diagnostics.update({
        "experiment_baseline": "raw_mask_geometry",
        "raw_area_px": int(np.count_nonzero(raw)),
        "processed_area_px": int(np.count_nonzero(raw)),
        "iou_to_raw": 1.0,
    })
    result["diagnostics"] = diagnostics
    return result


def _intermediate_mask_result(
    case_id: str,
    processed_result: Mapping[str, Any],
    *,
    raw_mask: np.ndarray,
) -> Dict[str, Any]:
    result = dict(processed_result)
    regularized = processed_result.get("regularized_body_mask")
    residual = processed_result.get("body_residual_mask")
    if not isinstance(regularized, np.ndarray):
        raise ValueError("regularized body evidence is unavailable")
    mask = (regularized > 0).astype(np.uint8) * 255
    if case_id == "body_residual_recovery":
        if not isinstance(residual, np.ndarray):
            raise ValueError("body residual evidence is unavailable")
        mask = cv2.bitwise_or(mask, (residual > 0).astype(np.uint8) * 255)
    result["processed_mask"] = mask
    result["headland_mask"] = np.zeros_like(mask)
    result["uncertain_residual_mask"] = np.zeros_like(mask)
    result["planning_support_mask"] = mask
    diagnostics = dict(result.get("diagnostics") or {})
    raw_bool = np.asarray(raw_mask) > 0
    mask_bool = mask > 0
    intersection = int(np.count_nonzero(raw_bool & mask_bool))
    union = int(np.count_nonzero(raw_bool | mask_bool))
    raw_area = int(np.count_nonzero(raw_bool))
    mask_area = int(np.count_nonzero(mask_bool))
    diagnostics.update({
        "experiment_baseline": case_id,
        "raw_area_px": raw_area,
        "processed_area_px": mask_area,
        "area_ratio": float(mask_area / max(raw_area, 1)),
        "precision_to_raw": float(intersection / max(mask_area, 1)),
        "recall_to_raw": float(intersection / max(raw_area, 1)),
        "iou_to_raw": float(intersection / max(union, 1)),
    })
    result["diagnostics"] = diagnostics
    return result


def _render_case_preview(
    source_image: np.ndarray,
    mask_result: Mapping[str, Any],
    path_result: Optional[Mapping[str, Any]],
    offset_xy=(0.0, 0.0),
) -> np.ndarray:
    image = np.asarray(source_image).copy()
    mask = mask_result.get("processed_mask")
    if isinstance(mask, np.ndarray):
        x0, y0 = (int(round(float(value))) for value in offset_xy)
        x1 = min(image.shape[1], x0 + mask.shape[1])
        y1 = min(image.shape[0], y0 + mask.shape[0])
        if x1 > max(0, x0) and y1 > max(0, y0):
            dst_x0, dst_y0 = max(0, x0), max(0, y0)
            src_x0, src_y0 = dst_x0 - x0, dst_y0 - y0
            local = mask[src_y0:src_y0 + (y1 - dst_y0), src_x0:src_x0 + (x1 - dst_x0)] > 0
            roi = image[dst_y0:y1, dst_x0:x1]
            roi[local] = (
                roi[local].astype(np.float32) * 0.65
                + np.asarray([60, 190, 80]) * 0.35
            ).astype(np.uint8)
    if path_result:
        for segment in path_result.get("full_path") or []:
            points = np.rint(np.asarray(segment, dtype=np.float64)).astype(np.int32)
            if points.ndim == 2 and len(points) > 1:
                cv2.polylines(image, [points], False, (250, 90, 40), 2, cv2.LINE_AA)
    return image


class PaperExperimentRunner:
    """DECISION-014: run comparable cases against an immutable project snapshot."""

    def __init__(
        self,
        *,
        raw_mask: np.ndarray,
        source_image: np.ndarray,
        state,
        geo,
        output_dir: str,
        mask_config: Optional[Mapping[str, Any]] = None,
        path_config: Optional[Mapping[str, Any]] = None,
        ground_truth_mask: Optional[np.ndarray] = None,
        process_mask_fn=None,
        plan_path_fn=None,
        progress_callback=None,
        cancel_callback=None,
    ):
        self.raw_mask = (np.asarray(raw_mask) > 0).astype(np.uint8) * 255
        self.source_image = np.asarray(source_image).copy()
        self.state = _snapshot_state(state)
        self.geo = geo
        self.output_dir = str(output_dir)
        self.mask_config = dict(mask_config or {})
        self.path_config = dict(path_config or {})
        self.ground_truth_mask = ground_truth_mask
        self.progress_callback = progress_callback
        self.cancel_callback = cancel_callback
        if process_mask_fn is None:
            from mask_processor import process_mask
            process_mask_fn = process_mask
        if plan_path_fn is None:
            from path_planner import plan_path
            plan_path_fn = plan_path
        self.process_mask_fn = process_mask_fn
        self.plan_path_fn = plan_path_fn

    def _check_cancelled(self):
        if callable(self.cancel_callback) and self.cancel_callback():
            raise RuntimeError("task cancelled")

    def _progress(self, completed: int, total: int, message: str):
        if callable(self.progress_callback):
            self.progress_callback(completed / max(total, 1), message)

    def run(self, *, plan: Optional[Mapping[str, Any]] = None) -> str:
        plan = dict(plan or default_experiment_plan([]))
        mask_cases = list(plan.get("mask_cases") or [])
        path_cases = list(plan.get("path_cases") or [])
        total = 1 + len(mask_cases) + len(path_cases)
        records = []
        self._check_cancelled()
        started = time.perf_counter()
        full_result = self.process_mask_fn(
            self.raw_mask.copy(), geo=self.geo, state=self.state,
            config=dict(self.mask_config),
        )
        if not isinstance(full_result, Mapping) or not isinstance(full_result.get("processed_mask"), np.ndarray):
            raise RuntimeError("full mask regularization did not produce a valid result")
        completed = 1
        self._progress(completed, total, "完整掩膜处理完成")

        mask_results = {
            "raw_mask_geometry": _raw_mask_result(self.raw_mask, full_result),
            "full_mask_regularization": dict(full_result),
        }
        for case in mask_cases:
            self._check_cancelled()
            case_id = str(case.get("id", "mask_case"))
            case_started = time.perf_counter()
            try:
                if case_id not in mask_results:
                    mask_results[case_id] = _intermediate_mask_result(
                        case_id,
                        full_result,
                        raw_mask=self.raw_mask,
                    )
                result = mask_results[case_id]
                metrics = _numeric_metrics(result.get("diagnostics") or {})
                metrics.update(segmentation_metrics(result["processed_mask"], self.ground_truth_mask))
                metrics["runtime_seconds"] = float(time.perf_counter() - case_started)
                records.append({
                    "case_id": case_id, "module": "mask", "status": "completed",
                    "metrics": metrics, "parameters": dict(case),
                    "mask": result["processed_mask"],
                    "preview": _render_case_preview(
                        self.source_image,
                        result,
                        None,
                        (
                            getattr(self.state, "mask_offset_x", 0) or 0,
                            getattr(self.state, "mask_offset_y", 0) or 0,
                        ),
                    ),
                })
            except Exception as exc:
                records.append({"case_id": case_id, "module": "mask", "status": "failed", "error": str(exc), "metrics": {}, "parameters": dict(case)})
            completed += 1
            self._progress(completed, total, f"掩膜实验：{case_id}")

        active_mask = mask_results["full_mask_regularization"]
        processed = active_mask["processed_mask"]
        for case in path_cases:
            self._check_cancelled()
            case_id = str(case.get("id", "path_case"))
            strategy = str(case.get("strategy", case_id))
            case_started = time.perf_counter()
            try:
                config = dict(self.path_config)
                config.update({
                    "turn_strategy": strategy,
                    "headland_mask": active_mask.get("headland_mask"),
                    "planning_support_mask": active_mask.get("planning_support_mask"),
                    "uncertain_mask": active_mask.get("uncertain_residual_mask"),
                    "forbidden_mask": active_mask.get("forbidden_mask"),
                })
                result = self.plan_path_fn(
                    active_mask.get("wide_bands") or [], processed,
                    float(active_mask.get("main_angle", 0.0)),
                    geo=self.geo, state=self.state, config=config,
                    detected_mask=self.raw_mask, planned_mask=processed,
                )
                metrics = _numeric_metrics(result.get("validation") or {})
                turn_assessment = result.get("turn_assessment") or {}
                metrics["turn_count"] = float(len(turn_assessment.get("turn_decisions") or []))
                # DECISION-015: preserve per-turn attribution in experiment metrics.
                metrics["headland_turn_track_overlap_area_m2"] = float(
                    (result.get("validation") or {}).get("headland_turn_track_overlap_area_m2", 0.0) or 0.0
                )
                metrics["turn_cost_m2"] = float(
                    (result.get("validation") or {}).get("turn_cost_m2", 0.0) or 0.0
                )
                metrics["track_rolling_visit_once_area_m2"] = float(
                    (result.get("validation") or {}).get("track_rolling_visit_once_area_m2", 0.0) or 0.0
                )
                metrics["track_rolling_visit_twice_or_more_area_m2"] = float(
                    (result.get("validation") or {}).get("track_rolling_visit_twice_or_more_area_m2", 0.0) or 0.0
                )
                metrics["runtime_seconds"] = float(time.perf_counter() - case_started)
                segment_metrics = list((result.get("validation") or {}).get("segment_metrics") or [])
                metrics.update(_segment_type_summary_metrics(segment_metrics))
                turn_semantics = {
                    "requested_strategy": turn_assessment.get("user_requested_strategy", strategy),
                    "candidate_strategy": turn_assessment.get("candidate_strategy", strategy),
                    "effective_strategies": list(turn_assessment.get("effective_strategies") or []),
                    "fallback_reasons": list(turn_assessment.get("fallback_reasons") or []),
                }
                records.append({
                    "case_id": case_id, "module": "path", "status": "completed",
                    "metrics": metrics,
                    "segment_metrics": segment_metrics,
                    "parameters": {
                        **dict(case),
                        "selected_strategy": turn_assessment.get("selected_candidate_strategy"),
                        "requested_strategy": turn_semantics["requested_strategy"],
                        "candidate_strategy": turn_semantics["candidate_strategy"],
                        "effective_strategies": turn_semantics["effective_strategies"],
                        "fallback_reasons": turn_semantics["fallback_reasons"],
                    },
                    "turn_semantics": turn_semantics,
                    "mask": processed,
                    "preview": _render_case_preview(
                        self.source_image,
                        active_mask,
                        result,
                        (
                            getattr(self.state, "mask_offset_x", 0) or 0,
                            getattr(self.state, "mask_offset_y", 0) or 0,
                        ),
                    ),
                })
            except Exception as exc:
                records.append({"case_id": case_id, "module": "path", "status": "failed", "error": str(exc), "metrics": {}, "parameters": dict(case)})
            completed += 1
            self._progress(completed, total, f"路径实验：{case_id}")

        inputs = {
            "source_sha256": str(getattr(self.state, "source_sha256", "") or ""),
            "model_path": str(getattr(self.state, "current_model_path", "") or ""),
            "model_sha256": str(getattr(self.state, "model_sha256", "") or ""),
            "raw_mask_sha256": hashlib.sha256(self.raw_mask.tobytes()).hexdigest(),
            "mask_config": self.mask_config,
            "path_config": self.path_config,
            "harvester_params": getattr(self.state, "harvester_params", {}),
            "runner_seconds": float(time.perf_counter() - started),
        }
        return export_experiment_bundle(self.output_dir, records, experiment_inputs=inputs, plan=plan)


def _export_experiment_bundle_base(
    output_dir: str,
    records: Sequence[Mapping[str, Any]],
    *,
    experiment_inputs: Optional[Mapping[str, Any]] = None,
    plan: Optional[Mapping[str, Any]] = None,
) -> str:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    serial_records = []
    for record in records:
        item = dict(record)
        case_dir = root / "cases" / _case_dir_name(item.get("case_id", ""))
        preview = item.pop("preview", None)
        mask = item.pop("mask", None)
        if isinstance(preview, np.ndarray):
            _write_image(case_dir / "preview.png", preview, rgb=True)
        if isinstance(mask, np.ndarray):
            _write_image(case_dir / "mask.png", mask)
        case_dir.mkdir(parents=True, exist_ok=True)
        save_json_atomic(str(case_dir / "result.json"), _jsonable(item))
        serial_records.append(_jsonable(item))

    long_rows = _metric_rows(records)
    _write_csv(root / "results_long.csv", long_rows, ["case_id", "module", "status", "error", "metric", "value", "unit"])
    wide_rows, wide_fields = _wide_rows(records)
    _write_csv(root / "results_wide.csv", wide_rows, wide_fields)
    modules = sorted({str(record.get("module", "unknown")) for record in records})
    for module in modules:
        module_rows = [row for row in wide_rows if row.get("module") == module]
        _write_csv(root / "tables" / f"{module}_comparison.csv", module_rows, wide_fields)
        _draw_metric_figure(root / "figures" / f"{module}_metrics.png", records, module)

    completed = sum(record.get("status") == "completed" for record in records)
    failed = sum(record.get("status") == "failed" for record in records)
    manifest = {
        "schema": EXPERIMENT_SCHEMA,
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": _jsonable(dict(experiment_inputs or {})),
        "plan": _jsonable(dict(plan or {})),
        "completed_count": completed,
        "failed_count": failed,
        "records": serial_records,
    }
    save_json_atomic(str(root / "experiment_manifest.json"), manifest)
    (root / "README.md").write_text(
        "# reUI 论文实验结果\n\n"
        "- `results_long.csv`：长表，适合统计软件和绘图。\n"
        "- `results_wide.csv`：宽表，适合直接整理论文表格。\n"
        "- `path_segments.csv`：按 case 和实际 segment_type 导出的逐段足迹指标。\n"
        "- `tables/`：按模块拆分的对比表。\n"
        "- `figures/`：自动生成的初步指标图，可作为论文制图底稿。\n"
        "- `cases/`：每个实验组的掩膜、路径预览和完整结果。\n\n"
        "监督分割指标仅在提供人工真值掩膜时产生。预计碾压率和几何覆盖率是二维影像平面指标，不等同于实际产量损失或收获率。\n",
        encoding="utf-8",
    )
    return str(root)


def export_experiment_bundle(
    output_dir: str,
    records: Sequence[Mapping[str, Any]],
    *,
    experiment_inputs: Optional[Mapping[str, Any]] = None,
    plan: Optional[Mapping[str, Any]] = None,
) -> str:
    """Export the aggregate bundle and canonical per-segment footprint metrics."""
    result = _export_experiment_bundle_base(
        output_dir,
        records,
        experiment_inputs=experiment_inputs,
        plan=plan,
    )
    segment_rows, segment_fields = _segment_metric_rows(records)
    _write_csv(Path(result) / "path_segments.csv", segment_rows, segment_fields)
    return result
