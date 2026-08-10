"""
掩膜处理器 — 碎片聚合 / 方向检测 / 黏连切割 / 种植带分类 / 中心线提取
"""
import math
import time
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
from config import AppLogger, MASK_PROCESSING_STRENGTH
from row_geometry import regularize_crop_mask
from row_orientation import estimate_mask_axis_pca, estimate_row_orientation


# ═══════════════════════════════════════════════════════════════
#  1. 方向检测
# ═══════════════════════════════════════════════════════════════

def detect_main_direction(mask: np.ndarray) -> float:
    """用 PCA 检测掩膜主方向角度（弧度，0~π）。

    返回值为田块多边形的长轴方向角度。
    注意: PCA 检测的是田块整体形状的主轴方向，适合近似矩形田块。
    对于不规则田块或种植行与田块长轴不平行的情况，
    建议后续使用 FFT 精化（见 detect_main_direction_refined）。

    文献参考: 此方法等价于最小面积外接矩形（MABR）的长轴方向，
    在田块级别的方向估计中是标准做法。
    """
    return estimate_mask_axis_pca(mask)


def detect_main_direction_refined(mask: np.ndarray,
                                   source_image: np.ndarray = None) -> float:
    """粗-精两阶段方向检测。

    Stage 1: PCA 粗估计田块主方向（field-level orientation）
    Stage 2: 若有源图像，使用 FFT 频谱分析精化种植行方向
             (参考: _estimate_row_angle in legacy_route_coverage.py)

    Args:
        mask: 二值掩膜
        source_image: 可选的原始 RGB/BGR 图像（用于 FFT 纹理分析）

    Returns:
        精化后的主方向角度（弧度，0~π）
    """
    coarse_angle = detect_main_direction(mask)

    if source_image is None:
        return coarse_angle

    try:
        refined_angle = estimate_row_orientation(mask, source_image)
        # 检查精化角度与粗估计的一致性（容差 15°）
        diff = abs(refined_angle - coarse_angle) % math.pi
        diff = min(diff, math.pi - diff)
        if diff < math.radians(15):
            # FFT 精化方向与 PCA 粗估计一致，采用精化结果
            return refined_angle
        else:
            # 差异较大时记录日志但仍用精化结果（FFT 更精确）
            log = AppLogger()
            log.info(f"方向精化: PCA={math.degrees(coarse_angle):.1f}°, "
                     f"FFT={math.degrees(refined_angle):.1f}°, "
                     f"差异={math.degrees(diff):.1f}° → 采用 FFT")
            return refined_angle
    except Exception as e:
        log = AppLogger()
        log.debug(f"FFT 精化不可用，使用 PCA 粗估计: {e}")
        return coarse_angle


def _perpendicular_angle(angle: float) -> float:
    """返回垂直于主方向的角度。"""
    return (angle + math.pi / 2) % math.pi


def _apply_strength_profile(config: Optional[Dict]) -> Dict:
    """Return the single supported mask-processing profile.

    The former light/strong profiles are intentionally no longer selectable.
    This normalization also protects direct callers and old cached settings
    from changing the processing parameters behind the UI.
    """
    cfg = dict(config or {})
    cfg["strength"] = MASK_PROCESSING_STRENGTH
    return cfg


# ═══════════════════════════════════════════════════════════════
#  2. 田头去除 — 去掉垂直于主方向的区域
# ═══════════════════════════════════════════════════════════════

def remove_headlands(mask: np.ndarray, main_angle: float,
                     angle_thresh_deg: float = 60.0,
                     min_area_ratio: float = 0.005,
                     return_headland: bool = False):
    """去除田头掉头区域（与主方向垂直的连通分量）。

    对每个连通分量，用最小外接矩形计算其主方向，
    如果与 main_angle 垂直（角度差 > angle_thresh_deg），
    且面积占比较小，则判定为田头区域并清除。

    Args:
        mask: 二值掩膜 (H, W) uint8, 0/255
        main_angle: 主方向角度（弧度）
        angle_thresh_deg: 角度差阈值（度），超过此值判定为垂直
        min_area_ratio: 最小面积占比阈值，面积占比小于此值的垂直区域才清除
    Returns:
        清除田头后的掩膜
    """
    if np.count_nonzero(mask) == 0:
        empty = mask.copy()
        return (empty, np.zeros_like(empty)) if return_headland else empty

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    total_area = np.count_nonzero(mask)
    result = mask.copy()
    headland = np.zeros_like(mask)

    angle_thresh_rad = math.radians(angle_thresh_deg)

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area / total_area < min_area_ratio:
            continue  # 太小的可能是噪声，跳过

        # 用该连通分量的掩膜计算方向
        comp_mask = (labels == i).astype(np.uint8) * 255
        comp_angle = detect_main_direction(comp_mask)

        # 计算与主方向的角度差（考虑 π 周期性）
        diff = abs(comp_angle - main_angle) % math.pi
        diff = min(diff, math.pi - diff)

        if diff > angle_thresh_rad:
            # 该分量方向与主方向垂直 → 田头区域
            result[labels == i] = 0
            headland[labels == i] = mask[labels == i]

    # Connected end-row/headland pixels can be merged with row bands into one
    # component. Preserve them as a separate layer by looking for dense
    # crosswise foreground columns/rows at the ends of the row direction.
    foreground = mask > 0
    main_angle = float(main_angle) % math.pi
    if abs(math.cos(main_angle)) >= abs(math.sin(main_angle)):
        col_counts = np.count_nonzero(foreground, axis=0)
        positive = col_counts[col_counts > 0]
        if positive.size:
            threshold = max(float(np.median(positive)) * 1.8, float(np.percentile(positive, 85)))
            cols = col_counts >= threshold
            headland[:, cols] = np.where(foreground[:, cols], mask[:, cols], headland[:, cols])
            result[:, cols] = np.where(foreground[:, cols], 0, result[:, cols])
    else:
        row_counts = np.count_nonzero(foreground, axis=1)
        positive = row_counts[row_counts > 0]
        if positive.size:
            threshold = max(float(np.median(positive)) * 1.8, float(np.percentile(positive, 85)))
            rows = row_counts >= threshold
            headland[rows, :] = np.where(foreground[rows, :], mask[rows, :], headland[rows, :])
            result[rows, :] = np.where(foreground[rows, :], 0, result[rows, :])

    return (result, headland) if return_headland else result


# ═══════════════════════════════════════════════════════════════
#  3. 碎片聚合 + 黏连切割
# ═══════════════════════════════════════════════════════════════

def aggregate_fragments(mask: np.ndarray, main_angle: float,
                        close_kernel_length: int = 15) -> np.ndarray:
    """用定向闭运算聚合碎片。

    沿主方向构造长条形核，桥接同方向的小间隙。
    """
    if np.count_nonzero(mask) == 0:
        return mask.copy()

    # 构造旋转的长条形核
    angle_deg = math.degrees(main_angle)
    k_len = max(5, close_kernel_length)
    k_w = max(3, k_len // 4)

    # 创建水平核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, k_w))
    # 旋转到主方向
    center = (k_len // 2, k_w // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel_rotated = cv2.warpAffine(kernel, M, (k_len, k_w),
                                     flags=cv2.INTER_NEAREST,
                                     borderValue=0)
    # 二值化旋转后的核
    _, kernel_rotated = cv2.threshold(kernel_rotated, 127, 255, cv2.THRESH_BINARY)

    # 闭运算聚合碎片
    result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_rotated, iterations=1)
    return result


def cut_adhesions(mask: np.ndarray, main_angle: float,
                  thin_kernel_length: int = 12,
                  use_distance_transform: bool = True,
                  distance_split_ratio: float = 0.3) -> np.ndarray:
    """切割种植带之间的黏连。

    分两步：
    1. 定向开运算：垂直于主方向腐蚀，断开细小桥接
    2. 距离变换：分析宽度瓶颈，切割较宽黏连

    Args:
        mask: 二值掩膜
        main_angle: 主方向角度
        thin_kernel_length: 定向开运算核长度
        use_distance_transform: 是否使用距离变换处理宽黏连
        distance_split_ratio: 距离变换切割阈值（相对于最大距离的比例）
    Returns:
        切割后的掩膜
    """
    if np.count_nonzero(mask) == 0:
        return mask.copy()

    perp_angle = _perpendicular_angle(main_angle)
    angle_deg = math.degrees(perp_angle)

    # Step 1: 定向开运算 — 垂直于主方向
    k_len = max(5, thin_kernel_length)
    k_w = max(3, k_len // 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, k_w))
    center = (k_len // 2, k_w // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel_rotated = cv2.warpAffine(kernel, M, (k_len, k_w),
                                     flags=cv2.INTER_NEAREST, borderValue=0)
    _, kernel_rotated = cv2.threshold(kernel_rotated, 127, 255, cv2.THRESH_BINARY)

    # 开运算：先腐蚀断开细桥接，再膨胀恢复带宽
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_rotated, iterations=1)

    if not use_distance_transform:
        return opened

    # Step 2: 距离变换 — 处理较宽黏连
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    dist_max = dist.max()
    if dist_max < 1:
        return opened

    # 找到"瓶颈"区域：距离值 < 最大距离的 ratio
    threshold_val = dist_max * distance_split_ratio
    bottles = (dist > 0) & (dist < threshold_val)

    # 在瓶颈区域切割
    result = opened.copy()
    result[bottles] = 0

    # 轻度闭运算恢复被误切的边缘
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return result


# ═══════════════════════════════════════════════════════════════
#  4. 种植带分类 — 4行 vs 2行（自适应）
# ═══════════════════════════════════════════════════════════════

def _measure_band_widths(mask: np.ndarray, main_angle: float,
                         geo=None, state=None) -> List[Dict]:
    """测量每条种植带的宽度。

    对每个连通分量，沿垂直于主方向的投影计算宽度。
    使用 bounding box 裁剪以节省内存。
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    perp_angle = _perpendicular_angle(main_angle)
    perp_dx = math.cos(perp_angle)
    perp_dy = math.sin(perp_angle)

    # 计算米/像素比例
    mpp = 1.0
    if geo and state and geo.is_ready():
        ox = int(getattr(state, "mask_offset_x", 0))
        oy = int(getattr(state, "mask_offset_y", 0))
        h, w = mask.shape[:2]
        mpp = geo.meters_per_pixel(ox + w / 2, oy + h / 2)

    bands = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 50:  # 过小的连通分量跳过
            continue

        # 使用 bounding box 裁剪，避免创建全尺寸掩膜
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        # 裁剪 labels 区域
        crop_labels = labels[by:by+bh, bx:bx+bw]
        crop_mask = (crop_labels == i)

        ys, xs = np.where(crop_mask)
        if len(xs) < 10:
            continue

        # 投影到垂直方向（使用全局坐标）
        global_xs = xs.astype(np.float64) + bx
        global_ys = ys.astype(np.float64) + by
        projections = global_xs * perp_dx + global_ys * perp_dy
        proj_min, proj_max = projections.min(), projections.max()
        width_px = proj_max - proj_min
        width_m = width_px * mpp

        cx, cy = centroids[i]

        bands.append({
            "id": i,
            "centroid": (cx, cy),
            "width_px": width_px,
            "width_m": width_m,
            "area": area,
            "bbox": (bx, by, bw, bh),
        })

    return bands


def classify_bands(bands: List[Dict],
                   user_threshold_m: float = 0.55,
                   auto_calibrate: bool = True,
                   cluster_tolerance: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
    """将种植带分类为 4行（宽）和 2行（窄）。

    Args:
        bands: 种植带列表，每项含 width_m 字段
        user_threshold_m: 用户设定的宽度阈值（米），高于此值为4行
        auto_calibrate: 是否自动校准阈值
        cluster_tolerance: 自动聚类时的容差（米）

    Returns:
        (wide_bands, narrow_bands) — 4行带列表 和 2行带列表
    """
    if not bands:
        return [], []

    # 优先使用剖面宽度（法向投影统计，更稳健；与导出诊断口径一致），
    # 缺失时回退到 bbox 宽度 width_m。
    widths = []
    for band in bands:
        value = band.get("profile_width_m")
        if value is None:
            value = band.get("width_m")
        try:
            widths.append(float(value) if value is not None else 0.0)
        except (TypeError, ValueError):
            widths.append(0.0)
    widths = np.asarray(widths, dtype=np.float64)

    if auto_calibrate and len(bands) >= 4:
        # 用双峰聚类（不依赖 sklearn）。先剔除离群值（边缘残留/田埂带，
        # 如剖面宽 12m 的带），否则最大间隙分界会被离群值拉偏。
        sorted_w = np.sort(widths)
        median_w = float(np.median(sorted_w))
        core = sorted_w[sorted_w <= max(median_w * 3.0, median_w + 2.0)]
        if len(core) >= 4:
            gaps = np.diff(core)
            split_idx = int(np.argmax(gaps)) + 1
            narrow_center = float(core[:split_idx].mean())
            wide_center = float(core[split_idx:].mean())
            calibrated_threshold = (narrow_center + wide_center) / 2.0

            if abs(calibrated_threshold - user_threshold_m) < cluster_tolerance + 0.1:
                threshold = calibrated_threshold
            else:
                threshold = user_threshold_m
        else:
            threshold = user_threshold_m
    else:
        threshold = user_threshold_m

    wide = [band for band, width in zip(bands, widths) if width >= threshold]
    narrow = [band for band, width in zip(bands, widths) if width < threshold]

    return wide, narrow


# ═══════════════════════════════════════════════════════════════
#  5. 中心线提取
# ═══════════════════════════════════════════════════════════════

def extract_centerlines(bands: List[Dict], main_angle: float,
                        full_mask: np.ndarray = None,
                        geo=None, state=None,
                        progress_callback=None) -> List[Dict]:
    """对每条种植带提取中心线。

    使用形态学骨架化 + 分支剪枝 + 自适应采样 + 多项式拟合。

    处理要点:
    - 自适应采样: 根据实际地理距离决定采样密度
    - 中心线平滑: 采样后做多项式拟合减少骨架锯齿
    - 连通分量预计算: connectedComponents 只在循环外执行一次

    Args:
        bands: 种植带列表（已筛选为4行带）
        main_angle: 主方向角度
        full_mask: 完整掩膜（用于提取连通分量）
        geo: GeoUtils 实例（用于计算 mpp）
        state: AppState 实例
        progress_callback: 可选回调 callback(band_idx, total_bands)

    Returns:
        带中心线的种植带列表，每项新增 centerline 字段
    """
    def _report(idx, total):
        if progress_callback:
            try:
                progress_callback(idx, total)
            except Exception:
                pass

    # 计算 mpp（米/像素）
    mpp = 1.0
    if geo and state and geo.is_ready():
        ox = int(getattr(state, 'mask_offset_x', 0))
        oy = int(getattr(state, 'mask_offset_y', 0))
        h, w = full_mask.shape[:2] if full_mask is not None else (1000, 1000)
        mpp = geo.meters_per_pixel(ox + w / 2, oy + h / 2)

    # 预计算连通分量（只执行一次，避免在循环中重复计算）
    total = len(bands)
    pre_labels = None
    if full_mask is not None:
        _n_labels, pre_labels, _pre_stats, _pre_centroids = cv2.connectedComponentsWithStats(full_mask, 8)
        # 如果 bands 的 id 来自之前的 connectedComponents 调用，
        # 需要确认 id 对齐。这里 pre_labels 的 id 从 0(背景) 开始递增，
        # 应与 _measure_band_widths 中 labels 的 id 一致。
        _report(0, total)  # 通知调用方：连通分量计算完成
    results = []
    for bi, b in enumerate(bands):
        _report(bi, total)

        # 从 full_mask 中提取该带的掩膜（使用 bbox 裁剪）
        if full_mask is not None and "bbox" in b and pre_labels is not None:
            bx, by, bw, bh = b["bbox"]
            # 先裁剪再比较，避免全图 bool 数组分配（性能关键）
            comp_mask = (pre_labels[by:by+bh, bx:bx+bw] == b["id"]).astype(np.uint8) * 255
        else:
            continue

        # 骨架化
        skeleton = _thin_skeleton(comp_mask)

        # 分支剪枝: 去除短分支（长度 < 主骨架 15%）
        skeleton = _prune_skeleton_branches(skeleton, prune_ratio=0.15)

        # 从骨架提取有序点列（使用全局坐标 + 自适应采样）
        centerline = _trace_skeleton(skeleton, main_angle,
                                     offset_x=bx, offset_y=by,
                                     mpp=mpp, point_interval_m=0.5)

        # 多项式拟合平滑中心线
        if len(centerline) >= 4:
            centerline = fit_centerline_curve(centerline)

        if len(centerline) >= 2:
            b["centerline"] = centerline
            results.append(b)

    _report(total, total)
    return results


def _prune_skeleton_branches(skeleton: np.ndarray,
                              prune_ratio: float = 0.15) -> np.ndarray:
    """骨架分支剪枝: 移除长度 < 主骨架 prune_ratio 的短分支。

    文献参考: Diao et al. (2016) 作物行骨架提取中的伪分支去除算法。

    Args:
        skeleton: 二值骨架图
        prune_ratio: 剪枝比例阈值（相对于主骨架长度）

    Returns:
        剪枝后的骨架图
    """
    if np.count_nonzero(skeleton) < 10:
        return skeleton

    # 找到端点（8邻域中只有1个邻居的像素）
    ys, xs = np.where(skeleton > 0)
    if len(xs) < 3:
        return skeleton

    # 使用连通分量分析分支结构
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, 8)
    if n_labels <= 1:
        return skeleton

    # 对每个连通分量，如果面积太小（相对于最大分量），移除
    if n_labels > 2:
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n_labels)]
        max_area = max(areas) if areas else 0
        prune_thresh = max_area * prune_ratio

        result = skeleton.copy()
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < prune_thresh:
                result[labels == i] = 0
        return result

    return skeleton


def _thin_skeleton(mask: np.ndarray) -> np.ndarray:
    """形态学骨架化，返回单像素宽骨架。"""
    binary = (mask > 0).astype(np.uint8) * 255
    skeleton = cv2.ximgproc.thinning(binary) if hasattr(cv2, 'ximgproc') else \
               _skeletonize_fallback(binary)
    return skeleton


def _skeletonize_fallback(binary: np.ndarray) -> np.ndarray:
    """无 cv2.ximgproc 时的骨架化回退实现（形态学细化）。"""
    skel = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary.copy()

    while True:
        eroded = cv2.erode(img, element)
        dilated = cv2.dilate(eroded, element)
        diff = cv2.subtract(img, dilated)
        skel = cv2.bitwise_or(skel, diff)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skel


def _trace_skeleton(skeleton: np.ndarray, main_angle: float,
                    offset_x: int = 0, offset_y: int = 0,
                    mpp: float = 1.0,
                    point_interval_m: float = 0.5) -> List[Tuple[float, float]]:
    """沿主方向追踪骨架点，返回有序点列。

    处理要点:
    - 自适应采样: 根据实际地理距离（米）而非固定数量采样
    - 分支剪枝: 移除短分支（长度 < 主中心线 15%）

    Args:
        skeleton: 二值骨架图
        main_angle: 主方向角度（弧度）
        offset_x, offset_y: 全局坐标偏移
        mpp: meters per pixel（每像素对应的米数）
        point_interval_m: 采样间距（米），默认 0.5m

    文献参考: Diao et al. (2016) 作物行骨架提取算法
    """
    ys, xs = np.where(skeleton > 0)
    if len(xs) < 2:
        return []

    coords = np.column_stack([xs.astype(np.float64) + offset_x,
                              ys.astype(np.float64) + offset_y])

    # 沿主方向投影排序
    dx = math.cos(main_angle)
    dy = math.sin(main_angle)
    projections = coords[:, 0] * dx + coords[:, 1] * dy
    order = np.argsort(projections)
    sorted_coords = coords[order]

    # 计算总长度（像素）
    total_length_px = 0.0
    for i in range(1, len(sorted_coords)):
        d = math.hypot(sorted_coords[i][0] - sorted_coords[i-1][0],
                       sorted_coords[i][1] - sorted_coords[i-1][1])
        total_length_px += d

    # 自适应采样: 每 point_interval_m 米一个点
    if mpp > 0:
        interval_px = point_interval_m / mpp
    else:
        interval_px = total_length_px / 20  # 回退到 ~20 个点

    n_samples = max(5, int(total_length_px / interval_px))
    step = max(1, len(sorted_coords) // n_samples)
    sampled = sorted_coords[::step]

    # 如果采样点太少，全部取
    if len(sampled) < 2:
        sampled = sorted_coords

    return [(float(p[0]), float(p[1])) for p in sampled]


def fit_centerline_curve(points: List[Tuple[float, float]],
                         n_segments: int = 0) -> List[Tuple[float, float]]:
    """对中心线点做多项式拟合，返回平滑曲线。

    如果中心线较长且弯曲，分段拟合后拼接。
    """
    if len(points) < 3:
        return points

    coords = np.array(points, dtype=np.float64)
    xs, ys = coords[:, 0], coords[:, 1]

    # 用弧长参数化
    dists = np.zeros(len(xs))
    for i in range(1, len(xs)):
        dists[i] = dists[i-1] + math.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1])

    # 多项式拟合（2阶或3阶）
    deg = min(3, len(points) - 1)
    try:
        poly_x = np.polyfit(dists, xs, deg)
        poly_y = np.polyfit(dists, ys, deg)
        # 在弧长上等距采样
        n_out = max(10, len(points))
        t_smooth = np.linspace(0, dists[-1], n_out)
        xs_smooth = np.polyval(poly_x, t_smooth)
        ys_smooth = np.polyval(poly_y, t_smooth)
        return [(float(x), float(y)) for x, y in zip(xs_smooth, ys_smooth)]
    except Exception:
        return points


# ═══════════════════════════════════════════════════════════════
#  6. 完整处理管道
# ═══════════════════════════════════════════════════════════════

def process_mask(raw_mask: np.ndarray,
                 geo=None, state=None,
                 config: Optional[Dict] = None,
                 progress_callback=None) -> Dict:
    """完整的掩膜处理管道。

    Args:
        raw_mask: 模型推理得到的原始二值掩膜 (H, W) uint8
        geo: GeoUtils 实例（用于米制参数）
        state: AppState 实例
        config: 可选配置覆盖
        progress_callback: 可选回调 callback(progress: float, stage: str)

    Returns:
        {
            "processed_mask": 处理后的掩膜,
            "main_angle": 主方向角度,
            "wide_bands": 4行带列表（含centerline）,
            "narrow_bands": 2行带列表,
            "all_bands": 所有带列表,
        }
    """
    log = AppLogger()
    config = _apply_strength_profile(config)
    cancel_callback = config.get("_cancel_callback")

    def abort_if_cancelled():
        if callable(cancel_callback) and cancel_callback():
            raise RuntimeError("task cancelled")

    start_time = time.perf_counter()
    last_stage_time = start_time
    stage_timings = []

    def timed_progress(pct, stage):
        nonlocal last_stage_time
        now = time.perf_counter()
        stage_timings.append({
            "stage": str(stage),
            "elapsed_s": float(now - last_stage_time),
            "at_s": float(now - start_time),
        })
        last_stage_time = now
        if progress_callback:
            try:
                progress_callback(pct, stage)
            except Exception:
                pass

    try:
        abort_if_cancelled()
        result = regularize_crop_mask(
            raw_mask,
            geo=geo,
            state=state,
            config=config,
            progress_callback=timed_progress,
        )
        diagnostics = result.setdefault("diagnostics", {})
        diagnostics["strength"] = config.get("strength", "standard")
        diagnostics["stage_timings"] = stage_timings
        diagnostics["total_time_s"] = float(time.perf_counter() - start_time)
        log.info(
            "metric mask regularization complete: "
            f"bands={len(result.get('all_bands', []))}, "
            f"wide={len(result.get('wide_bands', []))}, "
            f"narrow={len(result.get('narrow_bands', []))}, "
            f"strength={config.get('strength', 'standard')}, "
            f"angle={math.degrees(result.get('main_angle', 0.0)):.1f}, "
            f"IoU={diagnostics.get('iou_to_raw', 0.0):.3f}, "
            f"time={diagnostics['total_time_s']:.2f}s"
        )
        return result
    except Exception as e:
        if callable(cancel_callback) and cancel_callback():
            raise
        if not bool(config.get("allow_compatibility_fallback", False)):
            raise RuntimeError(
                f"metric mask regularization failed: {e}"
            ) from e
        log.warning(
            "metric mask regularization failed; explicit compatibility "
            f"fallback enabled: {e}"
        )

    # Compatibility implementation retained below for fallback. The metric,
    # row-aligned pipeline above is the preferred processing path.
    def _report(pct, stage):
        abort_if_cancelled()
        timed_progress(pct, stage)

    log = AppLogger()
    cfg = config or {}

    # 读取配置参数
    angle_thresh = cfg.get("headland_angle_thresh_deg", 60.0)
    min_area_ratio = cfg.get("headland_min_area_ratio", 0.005)
    close_kernel = cfg.get("aggregate_close_kernel", 15)
    thin_kernel = cfg.get("adhesion_thin_kernel", 12)
    use_dist = cfg.get("adhesion_use_distance_transform", True)
    dist_ratio = cfg.get("adhesion_distance_split_ratio", 0.3)
    width_threshold = cfg.get("band_width_threshold_m", 0.55)
    auto_calibrate = cfg.get("band_auto_calibrate", True)

    log.info("=== 掩膜处理管道开始 ===")

    # 1. 方向检测
    _report(0.05, "方向检测")
    main_angle = detect_main_direction(raw_mask)
    log.info(f"主方向: {math.degrees(main_angle):.1f}°")

    # 2. 田头分层：主体掩膜用于中心线提取，田头掩膜保留给调头/低权重碾压约束
    _report(0.15, "田头分层")
    mask, headland_mask = remove_headlands(raw_mask, main_angle, angle_thresh, min_area_ratio, return_headland=True)
    removed_px = np.count_nonzero(headland_mask)
    log.info(f"田头分层: 保留田头 {removed_px} 像素 ({removed_px/max(1,np.count_nonzero(raw_mask))*100:.1f}%)")

    # 中间态更新：让 UI 实时显示处理后的掩膜

    # 3. 碎片聚合
    _report(0.35, "碎片聚合")
    mask = aggregate_fragments(mask, main_angle, close_kernel)
    log.info(f"碎片聚合完成, 非零像素: {np.count_nonzero(mask)}")

    # 中间态更新

    # 4. 黏连切割
    _report(0.55, "黏连切割")
    mask = cut_adhesions(mask, main_angle, thin_kernel, use_dist, dist_ratio)
    log.info(f"黏连切割完成, 非零像素: {np.count_nonzero(mask)}")

    # 中间态更新

    # 5. 种植带分类
    _report(0.75, "种植带分类")
    bands = _measure_band_widths(mask, main_angle, geo, state)
    wide, narrow = classify_bands(bands, width_threshold, auto_calibrate)
    log.info(f"种植带分类: 4行={len(wide)}条, 2行={len(narrow)}条")

    # 6. 中心线提取（只对4行带）— 传入 geo/state 用于自适应采样
    _report(0.88, "中心线提取")
    def cl_progress(band_idx, total_bands):
        # 将逐条进度映射到 88%~100% 区间
        pct = 0.88 + (band_idx / max(1, total_bands)) * 0.12
        _report(min(pct, 1.0), f"中心线提取 ({band_idx}/{total_bands})")
    wide_with_cl = extract_centerlines(wide, main_angle, full_mask=mask,
                                        geo=geo, state=state,
                                        progress_callback=cl_progress)
    log.info(f"中心线提取: {len(wide_with_cl)}条有效中心线")

    _report(1.0, "完成")

    diagnostics = {
        "strength": config.get("strength", "standard"),
        "stage_timings": stage_timings,
        "total_time_s": float(time.perf_counter() - start_time),
        "fallback_pipeline": True,
    }
    uncertain_residual_mask = np.zeros_like(mask)
    return {
        "processed_mask": mask,
        "headland_mask": headland_mask,
        "uncertain_residual_mask": uncertain_residual_mask,
        "neutral_support_mask": uncertain_residual_mask,
        "planning_support_mask": cv2.bitwise_or(mask, headland_mask),
        "main_angle": main_angle,
        "wide_bands": wide_with_cl,
        "narrow_bands": narrow,
        "all_bands": bands,
        "diagnostics": diagnostics,
    }


# ═══════════════════════════════════════════════════════════════
#  7. 可视化辅助
# ═══════════════════════════════════════════════════════════════

def draw_bands_debug(mask: np.ndarray, wide_bands: List[Dict],
                     narrow_bands: List[Dict], main_angle: float) -> np.ndarray:
    """绘制种植带分类的调试图。"""
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for b in wide_bands:
        # 4行带：绿色矩形框
        if "bbox" in b:
            bx, by, bw, bh = b["bbox"]
            cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (0, 200, 0), 2)
        cx, cy = b["centroid"]
        cv2.putText(vis, f"4-row {b['width_m']:.2f}m", (int(cx)-30, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        # 画中心线
        cl = b.get("centerline", [])
        if len(cl) >= 2:
            pts = np.array(cl, dtype=np.int32)
            cv2.polylines(vis, [pts], False, (0, 0, 255), 2)

    for b in narrow_bands:
        # 2行带：蓝色矩形框
        if "bbox" in b:
            bx, by, bw, bh = b["bbox"]
            cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (200, 0, 0), 1)
        cx, cy = b["centroid"]
        cv2.putText(vis, f"2-row {b['width_m']:.2f}m", (int(cx)-30, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)

    # 画主方向箭头
    h, w = mask.shape[:2]
    cx, cy = w // 2, h // 2
    arrow_len = min(h, w) // 4
    dx = int(arrow_len * math.cos(main_angle))
    dy = int(arrow_len * math.sin(main_angle))
    cv2.arrowedLine(vis, (cx, cy), (cx + dx, cy + dy), (0, 255, 255), 2)

    return vis
