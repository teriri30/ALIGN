"""模型加载 + 分块推理 + 后台推理运行器"""
import os, math, threading, time, traceback
from typing import Optional, List, Tuple, Any, Dict
import numpy as np
import cv2

from config import Config, AppLogger
from state import AppState
from geo import GeoUtils
from task_lifecycle import TaskToken, make_task_token, result_is_admissible
from runtime_paths import resolve_model_path
# 掩膜处理和路径规划由独立步骤负责，不在推理运行器中调用


# v6 keeps the v5 evidence contract and makes the selected inference runtime
# explicit (GPU/FP16 when available) while using the configured batch size.
# The native pass remains authoritative; runtime details are part of the
# provenance so a result is never silently treated as equivalent after the
# execution contract changes.
#
# v5 keeps the v4 colour contract and adds a 2x contextual evidence pass for
# native orthomosaics.  A 640 px source tile in the 12 m field data is only
# about one four-row band wide, so it cannot expose both instance boundaries.
# The native pass remains authoritative for detail; the contextual pass is
# downsampled to 640 only to preserve between-band evidence and recover weak
# detections close to native support.  Both scales and their fusion parameters
# are recorded in provenance, so v4 caches must not be reused.
#
# v4 records both sides of the color contract explicitly. ``read_rgb_raster``
# and the in-app preview are RGB, while Ultralytics 8.x interprets numpy arrays
# as OpenCV BGR and flips them to RGB in ``BasePredictor.preprocess``.  The
# adapter in ``ModelEngine.predict`` therefore performs exactly one RGB->BGR
# conversion at the external-library boundary.  v3 passed RGB directly and its
# cached masks must be recomputed.
# DECISION-005: keep application arrays RGB and adapt once at the Ultralytics boundary.
INFERENCE_PIPELINE_VERSION = "source-tif-sahi-v8-overlap-fusion-context-conservative"


INFERENCE_DEFAULT_BATCH_SIZE = 8
INFERENCE_MAX_BATCH_SIZE = 32
# Context evidence is auxiliary to the native-resolution pass.  Requiring
# repeated support in most overlapping context tiles prevents one weak
# 1280px response from expanding the semantic mask across an orthomosaic.
INFERENCE_DEFAULT_CONTEXT_SUPPORT_RATIO = 0.80
INFERENCE_DEFAULT_RECOVERY_CONF = 0.20
INFERENCE_DEFAULT_MASK_THRESHOLD = 0.50
INFERENCE_DEFAULT_TILE_EDGE_WEIGHT = 0.05


def _gaussian_tile_weight(
    height: int,
    width: int,
    *,
    edge_floor: float = INFERENCE_DEFAULT_TILE_EDGE_WEIGHT,
) -> np.ndarray:
    """Return a smooth centre-weight for one source tile.

    The native GeoTIFF path previously pasted each tile with a hard maximum,
    which made the rectangular tile grid visible in the final mask.  A
    normalized Gaussian keeps the tile centre authoritative and makes the
    overlap area an evidence-weighted blend.  The small positive floor keeps
    edge pixels represented when a neighbouring tile has no valid prediction.
    """
    height = max(1, int(height))
    width = max(1, int(width))
    floor = min(max(float(edge_floor), 0.0), 1.0)
    gy, gx = np.ogrid[:height, :width]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    sx = max(width / 4.5, 1.0)
    sy = max(height / 4.5, 1.0)
    weight = np.exp(
        -(((gx - cx) ** 2) / (2.0 * sx * sx)
          + ((gy - cy) ** 2) / (2.0 * sy * sy))
    ).astype(np.float32)
    weight -= float(weight.min())
    peak = float(weight.max())
    if peak > 1e-10:
        weight /= peak
    return (weight * (1.0 - floor) + floor).astype(np.float32, copy=False)


def _probability_to_mask(
    probability: np.ndarray,
    *,
    threshold: float = INFERENCE_DEFAULT_MASK_THRESHOLD,
) -> np.ndarray:
    """Convert a fused probability map with one explicit threshold."""
    value = np.asarray(probability, dtype=np.float32)
    finite = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
    return (finite >= float(threshold)).astype(np.uint8) * 255


def _model_config(cfg=None) -> dict:
    source = cfg if cfg is not None else Config()
    if hasattr(source, "section"):
        return source.section("model")
    return dict(source or {})


def configured_inference_batch_size(cfg=None) -> int:
    """Return the bounded batch size used by formal tile inference.

    ``inference_batch_size`` is preferred.  ``eval_batch_size`` remains a
    compatibility fallback for existing user configurations, which already
    carry the value 8 in the shipped configuration.  The value changes the
    execution grouping only; it is recorded in inference provenance.
    """
    model_cfg = _model_config(cfg)
    raw = model_cfg.get("inference_batch_size", model_cfg.get(
        "eval_batch_size", INFERENCE_DEFAULT_BATCH_SIZE
    ))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = INFERENCE_DEFAULT_BATCH_SIZE
    return max(1, min(INFERENCE_MAX_BATCH_SIZE, value))


def resolve_inference_runtime(cfg=None) -> dict:
    """Resolve the reproducible device/precision policy for model inference."""
    model_cfg = _model_config(cfg)
    requested_device = str(model_cfg.get("inference_device", "auto") or "auto").strip().lower()
    requested_precision = str(
        model_cfg.get("inference_precision", "auto") or "auto"
    ).strip().lower()
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if requested_device == "cpu":
        device = "cpu"
    elif requested_device in {"cuda", "cuda:0", "0"}:
        device = "cuda:0" if cuda_available else "cpu"
    else:
        device = "cuda:0" if cuda_available else "cpu"

    use_half = (
        device.startswith("cuda")
        and requested_precision in {"auto", "fp16", "half"}
    )
    return {
        "device": device,
        "precision": "fp16" if use_half else "fp32",
    }


def build_inference_config(cfg=None, engine=None, batch_size=None) -> dict:
    """Build the exact execution settings bound into inference provenance."""
    runtime = None
    if engine is not None:
        runtime_getter = getattr(engine, "runtime_config", None)
        if callable(runtime_getter):
            try:
                runtime = dict(runtime_getter() or {})
            except Exception:
                runtime = None
    runtime = runtime or resolve_inference_runtime(cfg)
    return {
        "capture_size": int((cfg or Config()).TILE_CAPTURE_SIZE),
        "overlap": float((cfg or Config()).TILE_OVERLAP),
        "conf": float((cfg or Config()).MODEL_CONF),
        "iou": float((cfg or Config()).MODEL_IOU),
        "batch_size": int(
            configured_inference_batch_size(cfg)
            if batch_size is None
            else max(1, min(INFERENCE_MAX_BATCH_SIZE, int(batch_size)))
        ),
        "device": str(runtime.get("device", "cpu")),
        "precision": str(runtime.get("precision", "fp32")),
    }


def _result_mask_stack(result, target_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return aligned instance masks and confidences from one YOLO result.

    DECISION-009: the semantic union is not the only useful evidence.  Keeping
    the stack until contact boundaries are measured prevents adjacent model
    instances from being irreversibly flattened by ``np.max``.
    """
    height, width = [max(1, int(value)) for value in target_shape[:2]]
    if result is None or getattr(result, "masks", None) is None:
        return (
            np.zeros((0, height, width), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    masks = np.asarray(result.masks.data.cpu().numpy(), dtype=np.float32)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3 or masks.shape[0] == 0:
        return (
            np.zeros((0, height, width), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    if masks.shape[1:] != (height, width):
        masks = np.stack([
            cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
            for mask in masks
        ]).astype(np.float32, copy=False)
    boxes = getattr(result, "boxes", None)
    box_conf = getattr(boxes, "conf", None) if boxes is not None else None
    if box_conf is None:
        confidences = np.ones((masks.shape[0],), dtype=np.float32)
    else:
        confidences = np.asarray(box_conf.cpu().numpy(), dtype=np.float32).reshape(-1)
        if confidences.size != masks.shape[0]:
            confidences = np.ones((masks.shape[0],), dtype=np.float32)
    return masks, confidences


def _contact_boundary_from_instances(
    masks: np.ndarray,
    confidences: np.ndarray,
    *,
    minimum_confidence: float,
    contact_radius_px: int = 4,
    border_guard_px: int = 6,
) -> np.ndarray:
    """Extract only boundaries supported by two touching YOLO instances.

    Ordinary outer object contours are deliberately excluded.  This makes the
    evidence useful for separating a four-row band from an adjacent uniform
    block without drawing every tile edge into the semantic mask.
    """
    if masks.ndim != 3 or masks.shape[0] < 2:
        shape = masks.shape[1:] if masks.ndim == 3 else (1, 1)
        return np.zeros(shape, dtype=np.uint8)
    keep = np.asarray(confidences, dtype=np.float32) >= float(minimum_confidence)
    binary = np.asarray(masks[keep] >= 0.5, dtype=np.uint8)
    if binary.shape[0] < 2:
        return np.zeros(masks.shape[1:], dtype=np.uint8)
    radius = max(1, int(contact_radius_px))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    boundary = np.zeros(binary.shape[1:], dtype=np.uint8)
    union = np.any(binary > 0, axis=0).astype(np.uint8)
    for index, instance in enumerate(binary):
        others = np.any(
            np.delete(binary, index, axis=0) > 0,
            axis=0,
        ).astype(np.uint8)
        if not np.any(others):
            continue
        edge = cv2.morphologyEx(instance, cv2.MORPH_GRADIENT, edge_kernel)
        touching = cv2.dilate(others, kernel, iterations=1)
        boundary[(edge > 0) & (touching > 0) & (union > 0)] = 255
    guard = max(0, int(border_guard_px))
    if guard:
        boundary[:guard, :] = 0
        boundary[-guard:, :] = 0
        boundary[:, :guard] = 0
        boundary[:, -guard:] = 0
    return boundary


def _axis_angle_from_mask(mask: np.ndarray) -> float:
    points = np.column_stack(np.nonzero(np.asarray(mask) > 0))
    if points.shape[0] < 2:
        return 0.0
    xy = points[:, ::-1].astype(np.float64)
    xy -= np.mean(xy, axis=0, keepdims=True)
    covariance = np.cov(xy, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    return float(math.atan2(float(axis[1]), float(axis[0])) % math.pi)


def _filter_context_separators(
    candidate: np.ndarray,
    reference_mask: np.ndarray,
    *,
    meters_per_pixel: float,
    minimum_length_m: float,
    angle_tolerance_deg: float = 25.0,
) -> np.ndarray:
    """Keep long context boundaries parallel to the field work direction."""
    binary = (np.asarray(candidate) > 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros_like(binary, dtype=np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    main_angle = _axis_angle_from_mask(reference_mask)
    tolerance = math.radians(max(0.0, float(angle_tolerance_deg)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    filtered = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, count):
        # Keep the same connected-component pixels and PCA rule, but limit
        # each membership scan to its component bounding box.  Scanning the
        # full native orthomosaic once per component made this post-processing
        # dominate large-source inference time.
        x, y, width, height = [int(value) for value in stats[label, :4]]
        if width <= 0 or height <= 0:
            continue
        component_labels = labels[y:y + height, x:x + width]
        component = component_labels == label
        points_yx = np.column_stack(np.nonzero(component))
        if points_yx.shape[0] < 2:
            continue
        points = points_yx[:, ::-1].astype(np.float64)
        centered = points - np.mean(points, axis=0, keepdims=True)
        covariance = np.cov(centered, rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        projections = centered @ axis
        length_m = float(np.ptp(projections)) * float(meters_per_pixel)
        angle = float(math.atan2(float(axis[1]), float(axis[0])) % math.pi)
        diff = abs(angle - main_angle) % math.pi
        diff = min(diff, math.pi - diff)
        if length_m >= float(minimum_length_m) and diff <= tolerance:
            filtered_roi = filtered[y:y + height, x:x + width]
            filtered_roi[component] = 255
    return filtered


def _fuse_context_evidence(
    primary_mask: np.ndarray,
    support_votes: np.ndarray,
    boundary_votes: np.ndarray,
    coverage_votes: np.ndarray,
    *,
    meters_per_pixel: float,
    support_ratio: float = INFERENCE_DEFAULT_CONTEXT_SUPPORT_RATIO,
    recovery_distance_m: float = 0.75,
    separator_ratio: float = 0.30,
    separator_min_votes: int = 2,
    separator_min_length_m: float = 1.50,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Fuse conservative contextual recovery and instance separation evidence."""
    primary = np.where(np.asarray(primary_mask) > 0, 255, 0).astype(np.uint8)
    coverage = np.asarray(coverage_votes, dtype=np.float32)
    valid = coverage > 0
    support_fraction = np.zeros_like(coverage, dtype=np.float32)
    boundary_fraction = np.zeros_like(coverage, dtype=np.float32)
    np.divide(support_votes, coverage, out=support_fraction, where=valid)
    np.divide(boundary_votes, coverage, out=boundary_fraction, where=valid)

    contextual_support = support_fraction >= float(support_ratio)
    gap = (primary == 0).astype(np.uint8)
    distance_px = cv2.distanceTransform(gap, cv2.DIST_L2, 5)
    recovery_radius_px = max(
        1.0,
        float(recovery_distance_m) / max(float(meters_per_pixel), 1e-9),
    )
    recovered = contextual_support & (primary == 0) & (distance_px <= recovery_radius_px)
    combined = primary.copy()
    combined[recovered] = 255

    separator_candidate = (
        (np.asarray(boundary_votes) >= max(1, int(separator_min_votes)))
        & (boundary_fraction >= float(separator_ratio))
    ).astype(np.uint8) * 255
    separator = _filter_context_separators(
        separator_candidate,
        combined,
        meters_per_pixel=float(meters_per_pixel),
        minimum_length_m=float(separator_min_length_m),
    )
    if np.any(separator):
        separator = cv2.dilate(
            separator,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        combined[separator > 0] = 0
    diagnostics = {
        "context_recovered_pixels": int(np.count_nonzero(recovered)),
        "context_separator_pixels": int(np.count_nonzero(separator)),
    }
    return combined, diagnostics


# ─── ModelEngine ────────────────────────────────────────────
class ModelEngine:
    def __init__(self):
        self._model = None
        self._model_path = ""
        self._loaded = False
        self._runtime_config = resolve_inference_runtime()
        self.log = AppLogger()

    def load(self, model_path: str) -> bool:
        if model_path and not os.path.isabs(model_path):
            model_path = str(resolve_model_path(model_path))
        if not os.path.exists(model_path):
            self.log.error(f"模型文件不存在: {model_path}")
            return False
        try:
            from ultralytics import YOLO
            self.log.info(f"正在加载模型: {model_path}")
            self._model = YOLO(model_path)
            self._model_path = model_path
            self._loaded = True
            self._runtime_config = resolve_inference_runtime()
            self.log.info(
                "推理运行时: "
                f"device={self._runtime_config['device']}, "
                f"precision={self._runtime_config['precision']}"
            )
            self.log.info("模型加载成功")
            return True
        except Exception as e:
            self.log.error(f"模型加载失败: {e}")
            self._loaded = False
            return False

    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    def unload(self):
        """Release the active model when project history restores 'no model'."""
        model = self._model
        self._model = None
        self._model_path = ""
        self._loaded = False
        # Model switching is deliberately serialized by the GUI, but the
        # previous engine can still hold a large CPU/GPU allocation until its
        # last Python reference is collected.  Release it explicitly after
        # clearing the engine identity; this is a resource cleanup, not a
        # change to inference semantics.
        try:
            del model
            import gc

            gc.collect()
            import torch

            if bool(torch.cuda.is_available()):
                torch.cuda.empty_cache()
        except Exception:
            # CPU-only/frozen environments may not ship torch or CUDA.  The
            # model identity is already cleared, so cleanup remains best
            # effort and must never turn a successful switch into a failure.
            pass

    def runtime_config(self) -> dict:
        """Return the device/precision settings used by ``predict``."""
        return dict(self._runtime_config or resolve_inference_runtime())

    def predict(self, image: np.ndarray, conf: float = 0.5, iou: float = 0.5) -> Optional[Any]:
        if not self.is_loaded(): return None
        try:
            # Ultralytics 8.x treats numpy inputs as BGR and reverses the last
            # channel in BasePredictor.preprocess.  ALIGN owns an RGB contract,
            # so adapt once here for both single-image and batched-list calls.
            # Tensor inputs are left untouched because Ultralytics documents
            # them as already preprocessed BCHW RGB tensors.
            if isinstance(image, np.ndarray) and image.ndim >= 3 and image.shape[-1] == 3:
                model_input = np.ascontiguousarray(image[..., ::-1])
            elif isinstance(image, (list, tuple)):
                model_input = [
                    np.ascontiguousarray(item[..., ::-1])
                    if isinstance(item, np.ndarray) and item.ndim >= 3 and item.shape[-1] == 3
                    else item
                    for item in image
                ]
            else:
                model_input = image
            runtime = self.runtime_config()
            return self._model.predict(
                model_input,
                imgsz=640,
                conf=conf,
                iou=iou,
                retina_masks=True,
                verbose=False,
                device=str(runtime.get("device", "cpu")),
                half=str(runtime.get("precision", "fp32")) == "fp16",
            )
        except Exception as e:
            self.log.error(f"推理失败: {e}")
            return None

    def get_model_name(self) -> str:
        return os.path.basename(self._model_path) if self._model_path else "unknown"

# ─── 模型目录扫描 ──────────────────────────────────────────
def scan_model_dir(directory: str) -> List[Dict]:
    """扫描目录下的 .pt 模型文件，返回模型列表。
    支持扫描 yoloduibi/runs/ 下的训练权重。
    """
    models = []
    if not os.path.isdir(directory):
        return models

    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.pt'):
                full = os.path.join(root, f)
                # 从路径推断模型名
                rel = os.path.relpath(full, directory)
                name = rel.replace('\\', '/')
                models.append({"name": name, "path": full})
    return models


# ─── TiledInference ─────────────────────────────────────────
class TiledInference:
    INFER_SIZE = 640
    DEFAULT_CAPTURE_SIZE = 640
    DEFAULT_OVERLAP = 0.40          # 默认重叠率 0.40，用于降低切片边缘漏检
    DEFAULT_CONF = 0.25             # 默认模型置信度阈值 0.25
    DEFAULT_IOU = 0.45              # 默认 NMS IoU 阈值 0.45

    def __init__(self, engine: ModelEngine):
        self.engine = engine
        self.log = AppLogger()
        self._gauss_cache = {}  # 高斯权重缓存
        self._last_diagnostics: Dict[str, Any] = {}

    def _get_gauss_weight(self, sz: int) -> np.ndarray:
        """获取或生成高斯权重图：中心 1.0，边缘渐近 0，用于无缝融合"""
        if sz in self._gauss_cache:
            return self._gauss_cache[sz]
        self._gauss_cache[sz] = _gaussian_tile_weight(sz, sz)
        return self._gauss_cache[sz]

    def run(self, full_image: np.ndarray, progress_cb=None,
            capture_sz=DEFAULT_CAPTURE_SIZE, overlap=DEFAULT_OVERLAP,
            erode=False, conf=DEFAULT_CONF, iou=DEFAULT_IOU,
            batch_size: int = 4,
            mask_threshold: float = INFERENCE_DEFAULT_MASK_THRESHOLD
            ) -> Optional[np.ndarray]:
        """SAHI 三步：滑窗切块 -> 单块概率推理 -> 高斯加权融合
        Args:
            capture_sz: 从原图裁切的块尺寸，默认固定 640，避免额外缩放损失
            overlap: 重叠率（0~1），默认 0.40，用于降低切片边缘漏检
            erode: 是否执行防粘连腐蚀（细线建议 False）
            conf: 模型置信度阈值，默认 0.25
            iou: NMS 阈值，默认 0.45
            mask_threshold: 融合概率的固定掩膜阈值，默认 0.50
        """

        self._last_diagnostics = {}
        if not self.engine.is_loaded():
            self._last_diagnostics = {}
            return None

        H, W = full_image.shape[:2]
        capture_sz = int(capture_sz)
        overlap = min(max(float(overlap), 0.0), 0.95)
        infer_size = self.INFER_SIZE
        native = (capture_sz == infer_size)

        capture_stride = max(1, int(round(capture_sz * (1 - overlap))))
        n_cols = max(1, math.ceil((W - capture_sz) / capture_stride) + 1)
        n_rows = max(1, math.ceil((H - capture_sz) / capture_stride) + 1)
        total = n_rows * n_cols

        self.log.info(f"最大精度分块推理: {n_rows}x{n_cols}={total}块, "
                      f"{'原生640' if native else str(capture_sz)+'px'}, "
                      f"重叠率 {overlap*100:.0f}%, conf={conf}, iou={iou}")

        # 概率加权累加器（float32）
        prob_acc = np.zeros((H, W), dtype=np.float32)
        weight_acc = np.zeros((H, W), dtype=np.float32)
        gauss_w = self._get_gauss_weight(capture_sz)
        processed = 0
        batch_images = []
        batch_meta = []
        tile_trace: List[Dict[str, Any]] = []
        next_tile_index = 0

        def flush_batch():
            nonlocal processed, batch_images, batch_meta
            if not batch_images:
                return
            results = self.engine.predict(batch_images, conf=conf, iou=iou)
            if results is None:
                # Retry one-by-one after a batch failure, but never convert a
                # failed tile into valid background data.
                results = []
                for image, meta in zip(batch_images, batch_meta):
                    x0, y0 = int(meta[3]), int(meta[4])
                    single = self.engine.predict(image, conf=conf, iou=iou)
                    if not single:
                        raise RuntimeError(
                            f"tile inference failed at pixel window ({x0}, {y0})"
                        )
                    results.append(single[0])
            results = list(results)
            if len(results) != len(batch_meta):
                raise RuntimeError(
                    "tile inference failed: model result count does not match input tiles"
                )
            for result, (tile_index, row_index, col_index, x0, y0, ah, aw) in zip(results, batch_meta):
                if result is None:
                    raise RuntimeError(
                        f"tile inference failed at pixel window ({x0}, {y0})"
                    )
                tile_prob = np.zeros((capture_sz, capture_sz), dtype=np.float32)
                masks, confidences = _result_mask_stack(
                    result,
                    (capture_sz, capture_sz),
                )
                accepted = confidences >= float(conf)
                if np.any(accepted):
                    tile_prob = np.max(masks[accepted], axis=0).astype(
                        np.float32,
                        copy=False,
                    )
                roi_w = gauss_w[:ah, :aw]
                prob_acc[y0:y0+ah, x0:x0+aw] += tile_prob[:ah, :aw] * roi_w
                weight_acc[y0:y0+ah, x0:x0+aw] += roi_w
                tile_trace.append({
                    "tile_index": int(tile_index),
                    "row": int(row_index),
                    "column": int(col_index),
                    "source_window_xywh": [int(x0), int(y0), int(aw), int(ah)],
                    "output_window_xywh": [int(x0), int(y0), int(aw), int(ah)],
                    "accepted_instance_count": int(np.count_nonzero(accepted)),
                    "max_confidence": float(np.max(confidences)) if confidences.size else 0.0,
                    "tile_probability_pixels": int(
                        np.count_nonzero(tile_prob[:ah, :aw] >= float(mask_threshold))
                    ),
                    "skipped_nodata": False,
                })
                processed += 1
            if progress_cb:
                progress_cb(processed / total)
            batch_images = []
            batch_meta = []

        for row in range(n_rows):
            for col in range(n_cols):
                x0 = min(col * capture_stride, W - capture_sz)
                y0 = min(row * capture_stride, H - capture_sz)
                x0, y0 = max(0, x0), max(0, y0)

                tile_raw = full_image[y0:y0 + capture_sz, x0:x0 + capture_sz]
                if tile_raw.shape[0] < capture_sz or tile_raw.shape[1] < capture_sz:
                    tile_raw = cv2.copyMakeBorder(tile_raw, 0, capture_sz - tile_raw.shape[0],
                                                  0, capture_sz - tile_raw.shape[1],
                                                  cv2.BORDER_REFLECT)

                # 原生尺寸直接推理，无需缩放
                if native:
                    tile_infer = tile_raw
                else:
                    tile_infer = cv2.resize(tile_raw, (infer_size, infer_size),
                                            interpolation=cv2.INTER_LINEAR)

                # ``read_rgb_raster`` and the preview loader both return RGB.
                # Keep the internal array RGB; ModelEngine performs the single
                # Ultralytics-specific RGB->BGR boundary adaptation.
                ah = min(capture_sz, H - y0)
                aw = min(capture_sz, W - x0)
                batch_images.append(np.ascontiguousarray(tile_infer))
                batch_meta.append((next_tile_index, row, col, x0, y0, ah, aw))
                next_tile_index += 1
                if len(batch_images) >= max(1, int(batch_size)):
                    flush_batch()
        flush_batch()

        # 归一化概率 → 固定阈值。逐块 Otsu 会让相邻切片使用不同的掩膜
        # 标准，且无法表达跨切片的重叠证据。
        np.divide(
            prob_acc,
            weight_acc,
            out=prob_acc,
            where=weight_acc > 1e-10,
        )
        mask = _probability_to_mask(prob_acc, threshold=mask_threshold)
        self._last_diagnostics = {
            "capture_size": int(capture_sz),
            "overlap": float(overlap),
            "stride": int(capture_stride),
            "rows": int(n_rows),
            "columns": int(n_cols),
            "tile_count": int(total),
            "confidence_threshold": float(conf),
            "iou_threshold": float(iou),
            "mask_threshold": float(mask_threshold),
            "native_tile_input": bool(native),
            "tile_fusion": "gaussian_probability_mean",
            "tile_trace": tile_trace,
        }
        self.log.info(
            f"概率融合完成, threshold={float(mask_threshold):.2f}, "
            f"非零像素: {np.count_nonzero(mask)}"
        )

        # 可选轻度腐蚀（默认关闭）
        if erode:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.erode(mask, kernel, iterations=1)
            self.log.info(f"轻度腐蚀完成，剩余非零像素: {np.count_nonzero(mask)}")

        return mask

# ─── InferenceRunner ────────────────────────────────────────
class InferenceRunner:
    def __init__(
        self,
        state: AppState,
        engine: ModelEngine,
        geo: GeoUtils,
        token: Optional[TaskToken] = None,
    ):
        self.state = state
        self.engine = engine
        self.geo = geo
        self.cfg = Config()
        self.log = AppLogger()
        self._thread: Optional[threading.Thread] = None
        # Library callers may omit a token; capture identity at construction so
        # late results still fail closed after a project/model switch.
        self.token = token or make_task_token(state, "inference")
        self._cancel_event = threading.Event()
        self._cache_tif_path = str(getattr(state, "tif_path", "") or "")

    def cancel(self):
        """Invalidate this runner; in-flight work must not commit its result."""
        self._cancel_event.set()
        if self.token is not None:
            self.token.invalidate()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _is_active(self) -> bool:
        if self._cancel_event.is_set():
            return False
        # Fail closed: late results without a captured task token are rejected.
        if self.token is None:
            return False
        return result_is_admissible(self.token, self.state)

    def _update(self, **kwargs) -> bool:
        """Update state only while this task still owns the current project."""
        if not self._is_active():
            return False
        self.state.safe_update(**kwargs)
        return True

    def _cleanup(self, **kwargs) -> bool:
        """Clear task flags only when this task still owns the project."""
        if self.token is None or not self.token.belongs_to(self.state):
            return False
        self.state.safe_update(**kwargs)
        return True

    def start(self, full_image: np.ndarray, field_boundary: Optional[List] = None):
        if not self._is_active():
            return
        if self.state.inference_running: return
        self._cache_tif_path = str(getattr(self.state, "tif_path", "") or "")
        self.state.safe_update(
            mask_raw=None,
            inference_original_mask=None,
            inference_done=False,
            inference_provenance={},
            inference_running=True,
            inference_progress=0.0,
            status_message="正在运行 AI 推理...",
        )
        self._thread = threading.Thread(target=self._run, args=(full_image, field_boundary), daemon=True)
        self._thread.start()

    def start_from_tif(self, tif_path: str, display_shape: Tuple[int, int],
                       field_boundary: Optional[List] = None, downsample: int = 1):
        """DECISION-004: infer from source GeoTIFF pixels, never the UI preview.

        The preview is a display artifact. Using it for formal inference would
        discard narrow-row evidence and invalidate physical downstream results.
        Oversized crops therefore stream source-resolution tiles.
        """
        if not self._is_active() or self.state.inference_running:
            return
        self._cache_tif_path = str(tif_path or "")
        self.state.safe_update(
            mask_raw=None,
            inference_original_mask=None,
            inference_done=False,
            inference_provenance={},
            inference_running=True,
            inference_progress=0.0,
            status_message="正在读取原始影像执行 AI 推理...",
        )
        self._thread = threading.Thread(
            target=self._run_from_tif,
            args=(tif_path, display_shape, field_boundary, max(1, int(downsample or 1))),
            daemon=True,
        )
        self._thread.start()

    def _run_source_tif_tiles(self, src, x0_src: int, y0_src: int, crop_w: int, crop_h: int,
                              out_w: int, out_h: int) -> Tuple[np.ndarray, dict]:
        """DECISION-004: stream native source tiles without whole-crop scaling.

        The configured crop limit selects the memory strategy, not a lower
        scientific resolution. A failed tile fails the complete inference.
        """
        from rasterio.windows import Window
        from raster_preprocessing import EmptyRasterWindowError, read_rgb_raster

        capture_sz = int(self.cfg.TILE_CAPTURE_SIZE)
        overlap = min(max(float(self.cfg.TILE_OVERLAP), 0.0), 0.95)
        model_cfg = self.cfg.section("model")
        primary_conf = float(self.cfg.MODEL_CONF)
        configured_recovery_conf = float(
            model_cfg.get(
                "orthomosaic_recovery_conf_threshold",
                INFERENCE_DEFAULT_RECOVERY_CONF,
            )
        )
        recovery_conf = min(
            primary_conf,
            max(configured_recovery_conf, INFERENCE_DEFAULT_RECOVERY_CONF),
        )
        tile_mask_threshold = min(
            max(
                float(model_cfg.get(
                    "orthomosaic_primary_mask_threshold",
                    INFERENCE_DEFAULT_MASK_THRESHOLD,
                )),
                0.05,
            ),
            0.95,
        )
        tile_edge_weight_floor = min(
            max(
                float(model_cfg.get(
                    "orthomosaic_tile_edge_weight_floor",
                    INFERENCE_DEFAULT_TILE_EDGE_WEIGHT,
                )),
                0.0,
            ),
            1.0,
        )
        context_enabled = bool(
            model_cfg.get("orthomosaic_context_enabled", True)
            and getattr(src, "transform", None) is not None
        )
        context_multiplier = max(
            1,
            int(model_cfg.get("orthomosaic_context_capture_multiplier", 2)),
        )
        batch_size = configured_inference_batch_size(self.cfg)
        stride = max(1, int(round(capture_sz * (1.0 - overlap))))
        n_cols = max(1, math.ceil((crop_w - capture_sz) / stride) + 1)
        n_rows = max(1, math.ceil((crop_h - capture_sz) / stride) + 1)
        total = n_rows * n_cols
        # DECISION-004/009: native tiles remain authoritative, while their
        # overlapping evidence is blended with centre weights before the
        # semantic mask is formed; tile edges are never hard-pasted.
        out_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        tile_probability_acc = np.zeros((out_h, out_w), dtype=np.float32)
        tile_weight_acc = np.zeros((out_h, out_w), dtype=np.float32)
        tile_blend_weight = _gaussian_tile_weight(
            capture_sz,
            capture_sz,
            edge_floor=tile_edge_weight_floor,
        )
        preprocessing = {
            "streamed_source_tiles": True,
            "tile_capture_size": int(capture_sz),
            "tile_overlap": float(overlap),
            "tile_count": int(total),
            "primary_conf_threshold": float(primary_conf),
            "configured_recovery_conf_threshold": float(configured_recovery_conf),
            "effective_recovery_conf_threshold": float(recovery_conf),
            "primary_mask_threshold": float(tile_mask_threshold),
            "tile_fusion": "gaussian_probability_mean",
            "tile_edge_weight_floor": float(tile_edge_weight_floor),
            "batch_size": int(batch_size),
            "context_enabled": bool(context_enabled and context_multiplier > 1),
            "tile_preprocessing": [],
            "tile_trace": [],
            "context_tile_trace": [],
        }
        processed = 0
        valid_tile_count = 0
        skipped_nodata_count = 0
        batch_images = []
        batch_meta = []
        next_tile_index = 0

        def flush_batch():
            nonlocal processed, batch_images, batch_meta
            if not batch_images:
                return
            results = self.engine.predict(
                batch_images,
                conf=float(primary_conf),
                iou=float(self.cfg.MODEL_IOU),
            )
            if results is None:
                results = []
                for image, meta in zip(batch_images, batch_meta):
                    single = self.engine.predict(
                        image,
                        conf=float(primary_conf),
                        iou=float(self.cfg.MODEL_IOU),
                    )
                    if not single:
                        raise RuntimeError(f"tile inference failed at source window ({meta[3]}, {meta[4]})")
                    results.append(single[0])
            results = list(results)
            if len(results) != len(batch_meta):
                raise RuntimeError("tile inference failed: model result count does not match input tiles")
            for result, (tile_index, row_index, col_index, sx0, sy0, sw, sh) in zip(results, batch_meta):
                if result is None:
                    raise RuntimeError(f"tile inference failed at source window ({sx0}, {sy0})")
                masks, confidences = _result_mask_stack(
                    result,
                    (capture_sz, capture_sz),
                )
                accepted = confidences >= float(primary_conf)
                if np.any(accepted):
                    tile_probability = np.max(masks[accepted], axis=0).astype(
                        np.float32,
                        copy=False,
                    )
                else:
                    tile_probability = np.zeros(
                        (capture_sz, capture_sz),
                        dtype=np.float32,
                    )
                dx0 = int(math.floor((sx0 / max(1, crop_w)) * out_w))
                dy0 = int(math.floor((sy0 / max(1, crop_h)) * out_h))
                dx1 = int(math.ceil(((sx0 + sw) / max(1, crop_w)) * out_w))
                dy1 = int(math.ceil(((sy0 + sh) / max(1, crop_h)) * out_h))
                dx0, dy0 = max(0, dx0), max(0, dy0)
                dx1, dy1 = min(out_w, dx1), min(out_h, dy1)
                if dx1 > dx0 and dy1 > dy0:
                    target_size = (dx1 - dx0, dy1 - dy0)
                    projected_probability = cv2.resize(
                        tile_probability[:sh, :sw],
                        target_size,
                        interpolation=cv2.INTER_LINEAR,
                    )
                    projected_weight = cv2.resize(
                        tile_blend_weight[:sh, :sw],
                        target_size,
                        interpolation=cv2.INTER_LINEAR,
                    )
                    tile_probability_acc[dy0:dy1, dx0:dx1] += (
                        projected_probability * projected_weight
                    )
                    tile_weight_acc[dy0:dy1, dx0:dx1] += projected_weight
                preprocessing["tile_trace"].append({
                    "tile_index": int(tile_index),
                    "row": int(row_index),
                    "column": int(col_index),
                    "source_window_xywh": [
                        int(x0_src + sx0), int(y0_src + sy0), int(sw), int(sh)
                    ],
                    "source_crop_window_xywh": [int(sx0), int(sy0), int(sw), int(sh)],
                    "output_window_xywh": [
                        int(dx0), int(dy0), int(max(0, dx1 - dx0)), int(max(0, dy1 - dy0))
                    ],
                    "accepted_instance_count": int(np.count_nonzero(accepted)),
                    "max_confidence": float(np.max(confidences)) if confidences.size else 0.0,
                    "tile_probability_pixels": int(
                        np.count_nonzero(tile_probability[:sh, :sw] >= tile_mask_threshold)
                    ),
                    "skipped_nodata": False,
                })
                processed += 1
            self._update(
                inference_progress=0.04 + (processed / max(1, total)) * 0.60,
                status_message=f"原始影像流式切片推理中... {int(processed / max(1, total) * 100)}%",
            )
            batch_images = []
            batch_meta = []

        self.log.info(
            f"原始分辨率流式切片推理: {n_rows}x{n_cols}={total}块, "
            f"tile={capture_sz}px, overlap={overlap * 100:.0f}%"
        )
        for row in range(n_rows):
            for col in range(n_cols):
                sx0 = min(col * stride, crop_w - capture_sz)
                sy0 = min(row * stride, crop_h - capture_sz)
                sx0, sy0 = max(0, sx0), max(0, sy0)
                sw = min(capture_sz, crop_w - sx0)
                sh = min(capture_sz, crop_h - sy0)
                try:
                    tile_img, tile_pre = read_rgb_raster(
                        src,
                        window=Window(x0_src + sx0, y0_src + sy0, sw, sh),
                        config=self.cfg.section("raster_preprocessing"),
                    )
                except EmptyRasterWindowError:
                    skipped_nodata_count += 1
                    preprocessing["tile_trace"].append({
                        "tile_index": int(next_tile_index),
                        "row": int(row),
                        "column": int(col),
                        "source_window_xywh": [
                            int(x0_src + sx0), int(y0_src + sy0), int(sw), int(sh)
                        ],
                        "source_crop_window_xywh": [int(sx0), int(sy0), int(sw), int(sh)],
                        "accepted_instance_count": 0,
                        "max_confidence": 0.0,
                        "tile_probability_pixels": 0,
                        "skipped_nodata": True,
                    })
                    next_tile_index += 1
                    processed += 1
                    continue
                valid_tile_count += 1
                if len(preprocessing["tile_preprocessing"]) < 3:
                    preprocessing["tile_preprocessing"].append(tile_pre)
                if tile_img.shape[0] < capture_sz or tile_img.shape[1] < capture_sz:
                    tile_img = cv2.copyMakeBorder(
                        tile_img,
                        0,
                        capture_sz - tile_img.shape[0],
                        0,
                        capture_sz - tile_img.shape[1],
                        cv2.BORDER_REFLECT,
                    )
                # ``read_rgb_raster`` returns RGB. ModelEngine performs the
                # single conversion required by Ultralytics' BGR ndarray API.
                batch_images.append(np.ascontiguousarray(tile_img))
                batch_meta.append((next_tile_index, row, col, sx0, sy0, sw, sh))
                next_tile_index += 1
                if len(batch_images) >= batch_size:
                    flush_batch()
        flush_batch()
        preprocessing["valid_tile_count"] = int(valid_tile_count)
        preprocessing["skipped_nodata_tile_count"] = int(skipped_nodata_count)
        if valid_tile_count == 0:
            raise RuntimeError("selected source crop contains no valid RGB pixels")
        if skipped_nodata_count:
            self.log.info(
                f"source-tile inference skipped {skipped_nodata_count}/{total} all-NoData tiles"
            )
        np.divide(
            tile_probability_acc,
            tile_weight_acc,
            out=tile_probability_acc,
            where=tile_weight_acc > 1e-10,
        )
        out_mask = _probability_to_mask(
            tile_probability_acc,
            threshold=tile_mask_threshold,
        )
        preprocessing["primary_mask_pixels"] = int(np.count_nonzero(out_mask))
        self.log.info(
            "原始切片概率融合完成: "
            f"threshold={tile_mask_threshold:.2f}, "
            f"非零像素={np.count_nonzero(out_mask)}"
        )
        if context_enabled and context_multiplier > 1:
            context_capture = int(capture_sz * context_multiplier)
            context_overlap = min(
                max(float(model_cfg.get("orthomosaic_context_overlap", 0.50)), 0.0),
                0.90,
            )
            context_stride = max(
                1,
                int(round(context_capture * (1.0 - context_overlap))),
            )
            context_cols = max(
                1,
                math.ceil((crop_w - context_capture) / context_stride) + 1,
            )
            context_rows = max(
                1,
                math.ceil((crop_h - context_capture) / context_stride) + 1,
            )
            context_total = context_rows * context_cols
            support_votes = np.zeros((out_h, out_w), dtype=np.uint16)
            boundary_votes = np.zeros((out_h, out_w), dtype=np.uint16)
            coverage_votes = np.zeros((out_h, out_w), dtype=np.uint16)
            context_images = []
            context_meta = []
            context_processed = 0
            context_valid = 0
            context_skipped = 0
            context_tile_index = 0

            def flush_context_batch():
                nonlocal context_images, context_meta, context_processed
                if not context_images:
                    return
                results = self.engine.predict(
                    context_images,
                    conf=float(recovery_conf),
                    iou=float(self.cfg.MODEL_IOU),
                )
                if results is None:
                    results = []
                    for image, meta in zip(context_images, context_meta):
                        single = self.engine.predict(
                            image,
                            conf=float(recovery_conf),
                            iou=float(self.cfg.MODEL_IOU),
                        )
                        if not single:
                            raise RuntimeError(
                                f"context inference failed at source window ({meta[3]}, {meta[4]})"
                            )
                        results.append(single[0])
                results = list(results)
                if len(results) != len(context_meta):
                    raise RuntimeError(
                        "context inference failed: model result count does not match input tiles"
                    )
                for result, (tile_index, row_index, col_index, sx0, sy0, sw, sh) in zip(results, context_meta):
                    if result is None:
                        raise RuntimeError(
                            f"context inference failed at source window ({sx0}, {sy0})"
                        )
                    masks, confidences = _result_mask_stack(
                        result,
                        (TiledInference.INFER_SIZE, TiledInference.INFER_SIZE),
                    )
                    accepted = confidences >= float(recovery_conf)
                    if np.any(accepted):
                        tile_support = (
                            np.max(masks[accepted], axis=0) >= 0.5
                        ).astype(np.uint8) * 255
                    else:
                        tile_support = np.zeros(
                            (TiledInference.INFER_SIZE, TiledInference.INFER_SIZE),
                            dtype=np.uint8,
                        )
                    tile_boundary = _contact_boundary_from_instances(
                        masks,
                        confidences,
                        minimum_confidence=float(primary_conf),
                    )
                    dx0 = int(math.floor((sx0 / max(1, crop_w)) * out_w))
                    dy0 = int(math.floor((sy0 / max(1, crop_h)) * out_h))
                    dx1 = int(math.ceil(((sx0 + sw) / max(1, crop_w)) * out_w))
                    dy1 = int(math.ceil(((sy0 + sh) / max(1, crop_h)) * out_h))
                    dx0, dy0 = max(0, dx0), max(0, dy0)
                    dx1, dy1 = min(out_w, dx1), min(out_h, dy1)
                    if dx1 > dx0 and dy1 > dy0:
                        size = (dx1 - dx0, dy1 - dy0)
                        projected_support = cv2.resize(
                            tile_support,
                            size,
                            interpolation=cv2.INTER_NEAREST,
                        )
                        projected_boundary = cv2.resize(
                            tile_boundary,
                            size,
                            interpolation=cv2.INTER_NEAREST,
                        )
                        support_votes[dy0:dy1, dx0:dx1] += (
                            projected_support > 0
                        ).astype(np.uint16)
                        boundary_votes[dy0:dy1, dx0:dx1] += (
                            projected_boundary > 0
                        ).astype(np.uint16)
                        coverage_votes[dy0:dy1, dx0:dx1] += 1
                    preprocessing["context_tile_trace"].append({
                        "tile_index": int(tile_index),
                        "row": int(row_index),
                        "column": int(col_index),
                        "source_window_xywh": [
                            int(x0_src + sx0), int(y0_src + sy0), int(sw), int(sh)
                        ],
                        "source_crop_window_xywh": [int(sx0), int(sy0), int(sw), int(sh)],
                        "output_window_xywh": [
                            int(dx0), int(dy0), int(max(0, dx1 - dx0)), int(max(0, dy1 - dy0))
                        ],
                        "accepted_instance_count": int(np.count_nonzero(accepted)),
                        "max_confidence": float(np.max(confidences)) if confidences.size else 0.0,
                        "support_pixels": int(np.count_nonzero(tile_support)),
                        "boundary_pixels": int(np.count_nonzero(tile_boundary)),
                        "skipped_nodata": False,
                    })
                    context_processed += 1
                self._update(
                    inference_progress=(
                        0.64
                        + (context_processed / max(1, context_total)) * 0.24
                    ),
                    status_message=(
                        "正在提取跨切片实例边界... "
                        f"{int(context_processed / max(1, context_total) * 100)}%"
                    ),
                )
                context_images = []
                context_meta = []

            for row in range(context_rows):
                for col in range(context_cols):
                    sx0 = min(col * context_stride, crop_w - context_capture)
                    sy0 = min(row * context_stride, crop_h - context_capture)
                    sx0, sy0 = max(0, sx0), max(0, sy0)
                    sw = min(context_capture, crop_w - sx0)
                    sh = min(context_capture, crop_h - sy0)
                    try:
                        tile_img, _tile_pre = read_rgb_raster(
                            src,
                            window=Window(x0_src + sx0, y0_src + sy0, sw, sh),
                            config=self.cfg.section("raster_preprocessing"),
                        )
                    except EmptyRasterWindowError:
                        context_skipped += 1
                        preprocessing["context_tile_trace"].append({
                            "tile_index": int(context_tile_index),
                            "row": int(row),
                            "column": int(col),
                            "source_window_xywh": [
                                int(x0_src + sx0), int(y0_src + sy0), int(sw), int(sh)
                            ],
                            "source_crop_window_xywh": [int(sx0), int(sy0), int(sw), int(sh)],
                            "accepted_instance_count": 0,
                            "max_confidence": 0.0,
                            "support_pixels": 0,
                            "boundary_pixels": 0,
                            "skipped_nodata": True,
                        })
                        context_tile_index += 1
                        context_processed += 1
                        continue
                    context_valid += 1
                    if tile_img.shape[0] < context_capture or tile_img.shape[1] < context_capture:
                        tile_img = cv2.copyMakeBorder(
                            tile_img,
                            0,
                            context_capture - tile_img.shape[0],
                            0,
                            context_capture - tile_img.shape[1],
                            cv2.BORDER_REFLECT,
                        )
                    tile_img = cv2.resize(
                        tile_img,
                        (TiledInference.INFER_SIZE, TiledInference.INFER_SIZE),
                        interpolation=cv2.INTER_AREA,
                    )
                    context_images.append(np.ascontiguousarray(tile_img))
                    context_meta.append((context_tile_index, row, col, sx0, sy0, sw, sh))
                    context_tile_index += 1
                    if len(context_images) >= batch_size:
                        flush_context_batch()
            flush_context_batch()

            source_x_m = math.hypot(float(src.transform.a), float(src.transform.d))
            source_y_m = math.hypot(float(src.transform.b), float(src.transform.e))
            display_x_m = source_x_m * float(crop_w) / max(float(out_w), 1.0)
            display_y_m = source_y_m * float(crop_h) / max(float(out_h), 1.0)
            display_mpp = math.sqrt(max(display_x_m * display_y_m, 1e-12))
            out_mask, context_diagnostics = _fuse_context_evidence(
                out_mask,
                support_votes,
                boundary_votes,
                coverage_votes,
                meters_per_pixel=display_mpp,
                support_ratio=float(
                    model_cfg.get(
                        "orthomosaic_context_support_ratio",
                        INFERENCE_DEFAULT_CONTEXT_SUPPORT_RATIO,
                    )
                ),
                recovery_distance_m=float(model_cfg.get("orthomosaic_context_recovery_distance_m", 0.75)),
                separator_ratio=float(model_cfg.get("orthomosaic_context_separator_ratio", 0.30)),
                separator_min_votes=int(model_cfg.get("orthomosaic_context_separator_min_votes", 2)),
                separator_min_length_m=float(model_cfg.get("orthomosaic_context_separator_min_length_m", 1.50)),
            )
            preprocessing.update({
                "context_capture_size": int(context_capture),
                "context_resize_to": int(TiledInference.INFER_SIZE),
                "context_overlap": float(context_overlap),
                "context_tile_count": int(context_total),
                "context_valid_tile_count": int(context_valid),
                "context_skipped_nodata_tile_count": int(context_skipped),
                "context_display_meters_per_pixel": float(display_mpp),
                **context_diagnostics,
            })
            self.log.info(
                "context evidence fusion complete: "
                f"tiles={context_valid}/{context_total}, "
                f"recovered={context_diagnostics['context_recovered_pixels']}px, "
                f"separators={context_diagnostics['context_separator_pixels']}px"
            )
        return out_mask, preprocessing

    def _run_from_tif(self, tif_path: str, display_shape: Tuple[int, int],
                      field_boundary: Optional[List] = None, downsample: int = 1):
        succeeded = False
        started_at = time.perf_counter()
        try:
            import rasterio
            from rasterio.windows import Window
            from raster_preprocessing import read_rgb_raster

            if not field_boundary or len(field_boundary) < 3:
                self._update(status_message="推理失败：请先圈选田块", inference_running=False)
                return

            display_h, display_w = [max(1, int(v)) for v in display_shape[:2]]
            pts_display = np.asarray(field_boundary, dtype=np.float64)
            pad_display = 200
            # DECISION-020: the inference crop window must keep enough outside
            # context. A field circled tightly (e.g. ~1100 m2 with 25-32
            # vertices) previously shrank the crop to "field fills the whole
            # view"; with no soil/edge reference the model fused rows into one
            # connected blob (行间混连) and mask regularization failed with
            # "种植带重构失败". The window therefore expands to at least
            # min_infer_window_display_px (centered on the field bbox, clamped
            # to the display image). The field mask still clips the result, so
            # outside context never enters the output.
            min_window_display_px = max(
                1,
                int(
                    self.cfg.section("model").get(
                        "min_infer_window_display_px",
                        4000,
                    )
                ),
            )
            bmin_d = pts_display.min(axis=0)
            bmax_d = pts_display.max(axis=0)
            bcenter_d = (bmin_d + bmax_d) * 0.5
            half_x = max(
                (bmax_d[0] - bmin_d[0]) * 0.5 + pad_display,
                min_window_display_px * 0.5,
            )
            half_y = max(
                (bmax_d[1] - bmin_d[1]) * 0.5 + pad_display,
                min_window_display_px * 0.5,
            )
            x0_d = max(0, int(math.floor(bcenter_d[0] - half_x)))
            y0_d = max(0, int(math.floor(bcenter_d[1] - half_y)))
            x1_d = min(display_w, int(math.ceil(bcenter_d[0] + half_x)) + 1)
            y1_d = min(display_h, int(math.ceil(bcenter_d[1] + half_y)) + 1)

            self._update(inference_progress=0.03, status_message="正在读取原始分辨率田块裁剪...")
            with rasterio.open(tif_path) as src:
                scale_x = float(src.width) / float(display_w)
                scale_y = float(src.height) / float(display_h)
                pts_source = np.rint(
                    pts_display * np.asarray([scale_x, scale_y], dtype=np.float64)
                ).astype(np.int32)
                x0_src = max(0, int(math.floor(x0_d * scale_x)))
                y0_src = max(0, int(math.floor(y0_d * scale_y)))
                x1_src = min(src.width, int(math.ceil(x1_d * scale_x)))
                y1_src = min(src.height, int(math.ceil(y1_d * scale_y)))
                if x1_src <= x0_src or y1_src <= y0_src:
                    self._update(status_message="推理失败：田块范围无效", inference_running=False)
                    return
                out_w = max(1, x1_d - x0_d)
                out_h = max(1, y1_d - y0_d)
                crop_w = int(x1_src - x0_src)
                crop_h = int(y1_src - y0_src)
                crop_pixels = crop_w * crop_h
                max_crop_pixels = int(
                    Config().section("model").get(
                        "max_source_crop_pixels",
                        36_000_000,
                    )
                )
                if max_crop_pixels > 0 and crop_pixels > max_crop_pixels:
                    self.log.warning(
                        f"原始裁剪区域过大 ({crop_w}x{crop_h}, {crop_pixels / 1_000_000:.1f}MP)，"
                        "改用原始分辨率流式切片推理，避免整体下采样导致掩膜变粗"
                    )
                    self._update(
                        status_message=(
                            "圈选区域原始分辨率较大，正在按 640px 原始切片流式推理..."
                        )
                    )
                    crop_mask_src, preprocessing = self._run_source_tif_tiles(
                        src,
                        x0_src,
                        y0_src,
                        crop_w,
                        crop_h,
                        out_w,
                        out_h,
                    )
                    crop_img_shape = crop_mask_src.shape
                    local_mask_pts = pts_display.copy()
                    local_mask_pts[:, 0] -= x0_d
                    local_mask_pts[:, 1] -= y0_d
                else:
                    crop_img, preprocessing = read_rgb_raster(
                        src,
                        window=Window(x0_src, y0_src, x1_src - x0_src, y1_src - y0_src),
                        config=self.cfg.section("raster_preprocessing"),
                    )
                    crop_img_shape = crop_img.shape[:2]
                    local_mask_pts = pts_source.copy()
                    local_mask_pts[:, 0] -= x0_src
                    local_mask_pts[:, 1] -= y0_src

            crop_field_mask = np.zeros(crop_img_shape, dtype=np.uint8)
            # DECISION-020: clip the mask to the field polygon EXPANDED
            # outward so a tightly circled field keeps the surrounding
            # structure the mask regularization needs.  Band polygons and
            # centrelines are still clipped to the original boundary in
            # row_geometry.regularize_crop_mask.
            analysis_pts = local_mask_pts
            analysis_expand_m = float(
                self.cfg.section("model").get(
                    "field_analysis_expand_m",
                    6.0,
                )
            )
            # DECISION-020 hardening: the expansion must never be
            # disabled by a config quirk; clamp a sensible minimum and
            # report the resulting polygon in the journal.
            analysis_expand_m = max(4.0, analysis_expand_m)
            from row_geometry import expand_polygon_px

            transform = getattr(src, "transform", None)
            if transform is not None:
                display_mpp = math.hypot(
                    float(transform.a), float(transform.d)
                ) * scale_x
            else:
                display_mpp = 0.021  # display GSD fallback
            expand_px = analysis_expand_m / display_mpp
            analysis_pts = expand_polygon_px(
                analysis_pts,
                expand_px,
                crop_img_shape[0],
                crop_img_shape[1],
            )
            self.log.info(
                f"mask clip polygon: expand_m={analysis_expand_m:.1f} "
                f"expand_px={expand_px:.0f} pts_in={len(local_mask_pts)} "
                f"pts_out={len(analysis_pts)} bbox_out="
                f"({float(np.min(analysis_pts[:, 0])):.0f},"
                f"{float(np.min(analysis_pts[:, 1])):.0f},"
                f"{float(np.max(analysis_pts[:, 0])):.0f},"
                f"{float(np.max(analysis_pts[:, 1])):.0f})"
            )
            cv2.fillPoly(crop_field_mask, [np.rint(analysis_pts).astype(np.int32)], 255)
            self.log.info(
                f"原始分辨率田块裁剪尺寸: {crop_img_shape[1]}x{crop_img_shape[0]}, "
                f"display_scale=({scale_x:.9f},{scale_y:.9f})"
            )
            if crop_pixels <= max_crop_pixels or max_crop_pixels <= 0:
                def progress_cb(pct: float):
                    self._update(
                        inference_progress=0.04 + pct * 0.84,
                        status_message=f"原始影像切片推理中... {int(pct * 100)}%",
                    )

                tiled_inference = TiledInference(self.engine)
                crop_mask_src = tiled_inference.run(
                    crop_img,
                    progress_cb=progress_cb,
                    capture_sz=int(self.cfg.TILE_CAPTURE_SIZE),
                    overlap=float(self.cfg.TILE_OVERLAP),
                    erode=False,
                    conf=float(self.cfg.MODEL_CONF),
                    iou=float(self.cfg.MODEL_IOU),
                    batch_size=configured_inference_batch_size(self.cfg),
                )
                preprocessing = dict(preprocessing or {})
                preprocessing["tiled_inference"] = dict(
                    getattr(tiled_inference, "_last_diagnostics", {}) or {}
                )
            if crop_mask_src is not None:
                self._update(inference_progress=0.92, status_message="正在裁剪田块范围...")
                clipped = cv2.bitwise_and(crop_mask_src, crop_mask_src, mask=crop_field_mask)
                if np.count_nonzero(clipped) > 0:
                    crop_mask_src = clipped
            if crop_mask_src is None or np.count_nonzero(crop_mask_src) == 0:
                self._update(status_message="推理失败：掩膜为空", inference_running=False)
                return

            if crop_mask_src.shape[:2] == (out_h, out_w):
                mask = crop_mask_src
            else:
                mask = cv2.resize(crop_mask_src, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            boundary_meta = [[float(px), float(py)] for px, py in (self.state.field_boundary or [])]
            from provenance import file_sha256, make_stage_record
            source_sha256 = str(getattr(self.state, "source_sha256", "") or "")
            if not source_sha256:
                source_sha256 = file_sha256(tif_path)
            model_path = str(getattr(self.state, "current_model_path", "") or "")
            model_sha256 = str(getattr(self.state, "model_sha256", "") or "")
            if model_path and not model_sha256:
                model_sha256 = file_sha256(model_path)
            inference_config = build_inference_config(self.cfg, self.engine)
            provenance_inputs = {
                "source_sha256": source_sha256,
                "model_sha256": model_sha256,
                "field_boundary": boundary_meta,
                "source_crop_window": [x0_src, y0_src, crop_w, crop_h],
                "inference_config": inference_config,
                "preprocessing_config": self.cfg.section("raster_preprocessing"),
                "preprocessing": preprocessing,
            }
            inference_provenance = make_stage_record(
                "inference", INFERENCE_PIPELINE_VERSION, provenance_inputs, mask
            )
            cache_meta = {
                "model": self.state.current_model_name or "",
                "model_path": self.state.current_model_path or "",
                "strategy": INFERENCE_PIPELINE_VERSION,
                "model_sha256": model_sha256,
                "source_sha256": source_sha256,
                "inference_provenance": inference_provenance,
                "field_boundary": boundary_meta,
                "field_area_m2": float(self.state.field_area_m2 or 0.0),
                "source_crop_shape": [int(crop_mask_src.shape[0]), int(crop_mask_src.shape[1])],
                "display_downsample": int(downsample),
                "display_scale_x": float(scale_x),
                "display_scale_y": float(scale_y),
            }
            if not self._is_active():
                return
            from cache import save_mask, save_project_state
            cache_tif_path = self._cache_tif_path
            if not cache_tif_path or save_mask(
                cache_tif_path,
                mask,
                x0_d,
                y0_d,
                meta=cache_meta,
                commit_callback=self._is_active,
            ) is False:
                return
            if not self._is_active() or save_mask(
                cache_tif_path,
                mask,
                x0_d,
                y0_d,
                suffix="_best",
                meta=cache_meta,
                commit_callback=self._is_active,
            ) is False:
                return
            non_zero = np.count_nonzero(mask)
            inference_runtime = {
                "stage": "inference",
                "seconds": float(time.perf_counter() - started_at),
                "source_crop_shape": [int(crop_h), int(crop_w)],
                "display_mask_shape": [int(out_h), int(out_w)],
            }
            self._update(
                mask_raw=mask,
                inference_original_mask=mask,
                mask_offset_x=x0_d,
                mask_offset_y=y0_d,
                _mask_overlay_dirty=True,
                inference_done=True,
                cache_model_matches_active_engine=True,
                inference_provenance=inference_provenance,
                raster_preprocessing=preprocessing,
                source_sha256=source_sha256,
                model_sha256=model_sha256,
                inference_runtime=inference_runtime,
                inference_progress=1.0,
                status_message=f"AI 识别完成，原始影像推理掩膜已保存（非零像素: {non_zero}）。请执行「掩膜处理」。",
            )
            self.log.info(
                f"inference mask commit: nonzero={non_zero} admitted={self._is_active()}"
            )
            if not self._is_active():
                return
            save_project_state(
                cache_tif_path,
                self.state,
                stage="inference",
                commit_callback=self._is_active,
            )
            succeeded = True
            self.log.info(f"原始影像推理完成: source_mask={crop_mask_src.shape}, display_mask={mask.shape}, offset=({x0_d},{y0_d})")
        except Exception as e:
            self.log.error(f"原始影像推理异常: {traceback.format_exc()}")
            self._update(status_message=f"推理失败: {e}")
        finally:
            self._cleanup(
                inference_running=False,
                inference_progress=1.0,
                inference_done=succeeded,
            )
            if succeeded and self._is_active():
                from workflow import WorkflowUpdater
                WorkflowUpdater.advance(self.state, "INFERENCE_DONE")

    def _run(self, image: np.ndarray, field_boundary: Optional[List] = None):
        succeeded = False
        started_at = time.perf_counter()
        try:
            def progress_cb(pct: float):
                self._update(
                    inference_progress=0.04 + pct * 0.84,
                    status_message=f"切片推理中... {int(pct * 100)}%",
                )

            H, W = image.shape[:2]
            crop_mask = None
            x0 = y0 = 0

            if field_boundary and len(field_boundary) >= 3:
                pts = np.array(field_boundary, dtype=np.int32)
                pad = 200
                bx, by, bw, bh = cv2.boundingRect(pts)
                x0, y0 = bx, by
                x1, y1 = bx + bw, by + bh
                y0, y1 = max(0, y0 - pad), min(H, y1 + pad)
                x0, x1 = max(0, x0 - pad), min(W, x1 + pad)

                crop_img = image[y0:y1, x0:x1]
                crop_field_mask = np.zeros(crop_img.shape[:2], dtype=np.uint8)
                local_pts = pts.copy()
                local_pts[:, 0] -= x0
                local_pts[:, 1] -= y0
                analysis_pts = local_pts
                analysis_expand_m = float(
                    self.cfg.section("model").get(
                        "field_analysis_expand_m",
                        6.0,
                    )
                )
                if analysis_expand_m > 0:
                    from row_geometry import expand_polygon_px

                    # Preview path has no exact georeference; use the
                    # standard display GSD (0.021 m/px) as an approximation.
                    analysis_pts = expand_polygon_px(
                        analysis_pts,
                        analysis_expand_m / 0.021,
                        crop_img.shape[0],
                        crop_img.shape[1],
                    )
                cv2.fillPoly(crop_field_mask, [np.rint(analysis_pts).astype(np.int32)], 255)
                H_c, W_c = crop_img.shape[:2]
                self.log.info(f"田块裁剪尺寸: {W_c}x{H_c}")

                tiled = TiledInference(self.engine)
                crop_mask = tiled.run(
                    crop_img,
                    progress_cb=progress_cb,
                    capture_sz=int(self.cfg.TILE_CAPTURE_SIZE),
                    overlap=float(self.cfg.TILE_OVERLAP),
                    erode=False,
                    conf=float(self.cfg.MODEL_CONF),
                    iou=float(self.cfg.MODEL_IOU),
                    batch_size=configured_inference_batch_size(self.cfg),
                )
                if crop_mask is not None:
                    self._update(
                        inference_progress=0.92,
                        status_message="正在裁剪田块范围...",
                    )
                    clipped_mask = cv2.bitwise_and(crop_mask, crop_mask, mask=crop_field_mask)
                    if np.count_nonzero(clipped_mask) > 0:
                        crop_mask = clipped_mask
                    elif np.count_nonzero(crop_mask) > 0:
                        self.log.warning("裁剪后的田块掩膜为空，保留原始推理结果")

                if crop_mask is not None:
                    mask = crop_mask
                else:
                    mask = np.zeros((H, W), dtype=np.uint8)
            else:
                self.log.info("未圈选田块，跳过全图推理")
                mask = np.zeros((H, W), dtype=np.uint8)

            if mask is None or np.count_nonzero(mask) == 0:
                self._update(
                    status_message="推理失败：掩膜为空",
                    inference_running=False,
                )
                return

            boundary_meta = [[float(px), float(py)] for px, py in (self.state.field_boundary or [])]
            self._update(
                inference_progress=0.97,
                status_message="正在保存项目推理结果...",
            )
            from provenance import array_sha256, file_sha256, make_stage_record
            model_path = str(getattr(self.state, "current_model_path", "") or "")
            model_sha256 = str(getattr(self.state, "model_sha256", "") or "")
            if model_path and not model_sha256:
                model_sha256 = file_sha256(model_path)
            provenance_inputs = {
                "source_array_sha256": array_sha256(image),
                "model_sha256": model_sha256,
                "field_boundary": boundary_meta,
                "inference_config": build_inference_config(self.cfg, self.engine),
            }
            inference_provenance = make_stage_record(
                "inference", INFERENCE_PIPELINE_VERSION, provenance_inputs, mask
            )
            cache_meta = {
                "model": self.state.current_model_name or "",
                "model_path": self.state.current_model_path or "",
                "strategy": "sahi_640_40p",
                "model_sha256": model_sha256,
                "inference_provenance": inference_provenance,
                "field_boundary": boundary_meta,
                "field_area_m2": float(self.state.field_area_m2 or 0.0),
            }
            if not self._is_active():
                return
            from cache import save_mask, save_project_state
            cache_tif_path = self._cache_tif_path
            if not cache_tif_path or save_mask(
                cache_tif_path,
                mask,
                x0,
                y0,
                meta=cache_meta,
                commit_callback=self._is_active,
            ) is False:
                return
            if not self._is_active() or save_mask(
                cache_tif_path,
                mask,
                x0,
                y0,
                suffix="_best",
                meta=cache_meta,
                commit_callback=self._is_active,
            ) is False:
                return

            non_zero = np.count_nonzero(mask)
            self._update(
                mask_raw=mask,
                inference_original_mask=mask,
                mask_offset_x=x0,
                mask_offset_y=y0,
                _mask_overlay_dirty=True,
                inference_done=True,
                cache_model_matches_active_engine=True,
                inference_provenance=inference_provenance,
                model_sha256=model_sha256,
                inference_runtime={
                    "stage": "inference",
                    "seconds": float(time.perf_counter() - started_at),
                    "source_crop_shape": [int(mask.shape[0]), int(mask.shape[1])],
                    "display_mask_shape": [int(mask.shape[0]), int(mask.shape[1])],
                },
                inference_progress=1.0,
                status_message=f"AI 识别完成，原始掩膜已保存（非零像素: {non_zero}）。请执行「掩膜处理」。",
            )
            if not self._is_active():
                return
            save_project_state(
                cache_tif_path,
                self.state,
                stage="inference",
                commit_callback=self._is_active,
            )
            succeeded = True
            self.log.info(f"原始掩膜推理完成: mask shape={mask.shape}, non_zero={non_zero}, offset=({x0},{y0})")

        except Exception as e:
            self.log.error(f"推理异常: {traceback.format_exc()}")
            self._update(status_message=f"推理失败: {e}")
        finally:
            self._cleanup(
                inference_running=False,
                inference_progress=1.0,
                inference_done=succeeded,
            )
            if succeeded and self._is_active():
                from workflow import WorkflowUpdater
                WorkflowUpdater.advance(self.state, "INFERENCE_DONE")
