"""流程步骤管理 + 动画引擎"""
from typing import Optional, Tuple, List
import math
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw

from config import Config, AppLogger
from state import AppState, AutoPathSegment
from geo import GeoUtils
from path_pose import build_polyline_metric, pose_at_distance
from utils import get_chinese_font


class WorkflowUpdater:
    STEP_MAP = {
        "BOUNDARY_SET": 0, "INFERENCE_DONE": 1,
        "MASK_PROCESSED": 2, "PATH_PLANNED": 3,
        "ANIMATION_DONE": 4, "EXPORTED": 5,
    }

    @staticmethod
    def advance(state: AppState, step_key: str):
        target = WorkflowUpdater.STEP_MAP.get(step_key, -1)
        if target >= 0 and target > state.workflow_step:
            state.safe_update(workflow_step=target)
        if step_key == "ANIMATION_DONE":
            state.safe_update(simulation_done=True)
        elif step_key == "EXPORTED":
            state.safe_update(export_done=True)
        if target >= 0 and getattr(state, "tif_path", ""):
            try:
                from cache import request_project_state_save
                request_project_state_save(
                    state.tif_path,
                    state,
                    stage=step_key.lower(),
                )
            except Exception:
                pass

    @staticmethod
    def draw_workflow_bar(full_img: np.ndarray, state: AppState, win_w: int, ui_h: int):
        """绘制底部流程进度条（全部 PIL 抗锯齿渲染）"""
        bar_h, y_start = 25, ui_h - 25  # 25px 高进度条
        steps, n = state.workflow_steps, len(state.workflow_steps)
        sw = win_w // n

        # 背景
        cv2.rectangle(full_img, (0, y_start), (win_w, ui_h), (45, 45, 48), -1)
        cv2.rectangle(full_img, (0, y_start), (win_w, y_start + 1), (60, 60, 63), -1)

        # 进度线段（底部 3px）
        for i in range(n):
            x = i * sw
            if i < state.workflow_step:
                color = (50, 160, 70)       # 已完成 - 绿色
            elif i == state.workflow_step:
                color = (0, 110, 210)        # 当前 - 蓝色
            else:
                color = (70, 70, 73)         # 未完成 - 灰色
            seg_x1, seg_x2 = x + 6, x + sw - 6
            cv2.rectangle(full_img, (seg_x1, y_start + 22), (seg_x2, y_start + 25), color, -1)

        # PIL 通道：抗锯齿文字渲染
        pil = Image.fromarray(cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        ft_num = get_chinese_font(11)
        ft_lbl = get_chinese_font(13)
        number_chars = "①②③④⑤⑥"

        for i, name in enumerate(steps):
            x = i * sw
            dot_cx = x + sw // 2
            prefix = number_chars[i] if i < len(number_chars) else f"{i+1}."
            label = f"{prefix}{name}"
            fg = (210, 230, 210) if i <= state.workflow_step else (130, 130, 135)
            tb = draw.textbbox((0, 0), label, font=ft_lbl)
            lbl_w = tb[2] - tb[0]
            text_y = y_start + (22 - (tb[3] - tb[1])) // 2
            draw.text((dot_cx - lbl_w // 2, text_y), label, font=ft_lbl, fill=fg)

        full_img[:, :] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


class AnimationEngine:
    """动画引擎 — 基于真实物理尺度的平滑农机运动模拟（DECISION-012）

    核心改进:
    1. setup() 重建路径：转弯处圆弧插值，所有路径点等间距分布
    2. advance() 时间驱动：根据真实时间间隔 + 速度(m/s) 计算前进像素
    3. 亚帧插值：消除离散点跳跃，实现丝滑移动
    """

    # 农机速度参数 (m/s)
    WORK_SPEED_MPS = 1.2       # 田间作业速度 ~4.3 km/h
    TURN_SPEED_MPS = 0.6       # 转弯减速 ~2.2 km/h
    ENTRY_SPEED_MPS = 0.8      # 进场/离场速度
    # 路径点间距：每隔多少米插一个点（决定路径平滑度）
    PATH_RES_M = 0.3           # 每0.3米一个点

    def max_speed_multiplier(self) -> float:
        return max(
            5.0,
            float(self.cfg.section("animation").get(
                "max_speed_multiplier", 60.0
            )),
        )

    def __init__(self, state: AppState, geo: GeoUtils):
        self.state = state
        self.geo = geo
        self.cfg = Config()
        self.log = AppLogger()
        self._last_time: float = 0.0  # 上次 advance 的时间戳
        self._point_spacing_m: float = self.PATH_RES_M  # 路径点间距（米）
        self._seg_dists: List[float] = []
        self._cum_dists: List[float] = [0.0]
        self._pose_segments: List[dict] = []

    def _pixel_distance_m(self, p1, p2) -> float:
        """Return a finite distance in metres, falling back to pixel Euclidean distance."""
        try:
            distance = float(self.geo.pixel_distance_m(p1, p2))
        except Exception:
            distance = float("nan")
        if not math.isfinite(distance) or distance < 0:
            distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        return distance

    def setup(self, path_segments: List[AutoPathSegment]):
        """直接使用原始路径点，不做任何平滑或重采样。

        动画坐标与导出坐标保持一致。
        路径点之间的实际地理距离存入 _seg_dists 供 advance() 使用。
        """
        pts, types, statuses = [], [], []
        for seg in path_segments:
            for p in seg.points:
                point = (p.pixel_x, p.pixel_y)
                if pts and math.hypot(point[0] - pts[-1][0], point[1] - pts[-1][1]) < 1e-6:
                    # Adjacent path segments share an endpoint; keep it only once.
                    types[-1] = seg.segment_type
                    statuses[-1] = seg.status
                    continue
                pts.append(point)
                types.append(seg.segment_type)
                statuses.append(seg.status)

        if len(pts) < 2:
            self._fallback_setup(path_segments)
            return

        s = self.state
        s.anim_all_pts = pts
        s.anim_all_types = types
        s.anim_all_status = statuses
        s.anim_total_frames = len(pts)
        s.anim_frame = 0
        s.anim_frac = 0.0
        s.anim_active = True
        s.anim_paused = False
        s.anim_speed = 1.0

        # 计算每对相邻点之间的实际地理距离（米）
        self._seg_dists = []
        for i in range(len(pts) - 1):
            d = self._pixel_distance_m(pts[i], pts[i + 1])
            self._seg_dists.append(d)
        self._cum_dists = [0.0]
        for distance in self._seg_dists:
            self._cum_dists.append(self._cum_dists[-1] + max(0.0, distance))
        self._pose_segments = []
        pose_offset_m = 0.0
        for segment in path_segments:
            segment_points = [
                (float(point.pixel_x), float(point.pixel_y))
                for point in segment.points
            ]
            try:
                values, lengths, cumulative = build_polyline_metric(
                    segment_points, self._pixel_distance_m
                )
            except ValueError:
                continue
            segment_length_m = float(cumulative[-1])
            self._pose_segments.append({
                "points": values,
                "lengths": lengths,
                "cumulative": cumulative,
                "start_m": pose_offset_m,
                "end_m": pose_offset_m + segment_length_m,
                "segment_type": str(segment.segment_type),
            })
            pose_offset_m += segment_length_m

        # 计算平均间距供诊断使用
        total_m = sum(self._seg_dists)
        self._point_spacing_m = total_m / len(self._seg_dists) if self._seg_dists else self.PATH_RES_M

        # 作业速度（米/秒），如有地理数据则使用真实速度
        work_speed_mps = self._base_speed_mps("work")
        s.anim_speed_mps = work_speed_mps
        s.anim_heading_rad = 0.0
        s.anim_curvature_1pm = 0.0
        s.anim_track_left_mps = work_speed_mps
        s.anim_track_right_mps = work_speed_mps
        s.anim_turn_radius_m = float("inf")

        self._last_time = time.time()
        self.log.info(
            f"动画初始化: {len(pts)} 点 (原始路径，无平滑), "
            f"总长 {total_m:.1f}m, 平均间距 {self._point_spacing_m:.3f}m")

    def _fallback_setup(self, path_segments):
        """降级方案：直接使用原始点"""
        pts, st, types = [], [], []
        for seg in path_segments:
            p = [(p.pixel_x, p.pixel_y) for p in seg.points]
            pts.extend(p)
            st.extend([seg.status] * len(p))
            types.extend([seg.segment_type] * len(p))
        s = self.state
        s.anim_all_pts, s.anim_all_status, s.anim_all_types = pts, st, types
        s.anim_total_frames, s.anim_frame = len(pts), 0
        s.anim_frac = 0.0
        s.anim_active, s.anim_paused, s.anim_speed = True, False, 1.0
        work_speed_mps = self._base_speed_mps("work")
        s.anim_speed_mps = work_speed_mps
        # 计算段间距离（与 setup() 保持一致）
        self._seg_dists = []
        for i in range(len(pts) - 1):
            d = self._pixel_distance_m(pts[i], pts[i + 1])
            self._seg_dists.append(d)
        self._cum_dists = [0.0]
        for distance in self._seg_dists:
            self._cum_dists.append(self._cum_dists[-1] + max(0.0, distance))
        self._point_spacing_m = self.PATH_RES_M
        s.anim_heading_rad = 0.0
        s.anim_curvature_1pm = 0.0
        s.anim_track_left_mps = work_speed_mps
        s.anim_track_right_mps = work_speed_mps
        s.anim_turn_radius_m = float("inf")

    # 以下函数已弃用；setup 使用原始路径点，不做平滑/重采样。
    # 如需恢复请参见 backups/workflow.py.bak 中的 _smooth_join_segments / _resample_path / _avg_point_distance

    def advance(self) -> bool:
        """时间驱动的帧推进 — 基于真实地理距离 + 亚帧插值

        使用 _seg_dists 中每对点的实际地理距离，
        不做等距假设，确保动画轨迹与导出坐标完全一致。
        """
        s = self.state
        if not s.anim_active or s.anim_paused:
            return True

        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        # 限制 dt 防止异常跳帧
        dt = min(dt, 0.1)

        if dt <= 0:
            s.need_redraw = True
            return True

        # 当前点的类型决定速度
        cur_idx = min(s.anim_frame, len(s.anim_all_pts) - 1)
        cur_type = s.anim_all_types[cur_idx] if cur_idx < len(s.anim_all_types) else "work"
        speed_mps = self._base_speed_mps(cur_type)

        # 应用用户速度倍率
        effective_speed = speed_mps * s.anim_speed

        # 本帧前进的真实距离（米）
        advance_m = effective_speed * dt

        # 首帧诊断日志
        if not hasattr(self, '_diag_logged'):
            total_path_m = sum(self._seg_dists) if self._seg_dists else 0
            est_time = total_path_m / effective_speed if effective_speed > 0 else 0
            self.log.info(
                f"[动画诊断] speed={effective_speed:.2f} m/s, "
                f"advance_m={advance_m:.4f}m, "
                f"total_pts={len(s.anim_all_pts)}, total_path={total_path_m:.1f}m, "
                f"预计用时={est_time:.1f}s")
            self._diag_logged = True

        # 累积剩余前进距离（米），逐点消耗
        remaining_m = advance_m + s.anim_frac  # 上一帧的亚帧余量(米)

        steps = 0
        while remaining_m > 0 and (s.anim_frame + steps) < len(self._seg_dists):
            seg_d = self._seg_dists[s.anim_frame + steps]
            if remaining_m >= seg_d:
                remaining_m -= seg_d
                steps += 1
            else:
                break

        # 存储亚帧余量（米）
        s.anim_frac = remaining_m
        s.anim_frame += steps

        if s.anim_frame >= s.anim_total_frames:
            s.anim_frame = s.anim_total_frames - 1
            s.anim_active = False
            s.anim_paused = False
            s.ctrl_panel_visible = False
            s.anim_frac = 0.0
            self._gen_report()
            # 预演示完成，推进工作流
            WorkflowUpdater.advance(s, "ANIMATION_DONE")
            return False

        s.need_redraw = True
        return True

    def toggle_pause(self):
        self.state.anim_paused = not self.state.anim_paused
        self.state.need_redraw = True
        if not self.state.anim_paused:
            self._last_time = time.time()  # 恢复时重置时间

    def progress_ratio(self) -> float:
        """Return playback progress by path arc length, not point count."""
        s = self.state
        if not s.anim_all_pts or not self._cum_dists:
            return 0.0
        total_m = float(self._cum_dists[-1])
        if total_m <= 1e-9:
            return float(s.anim_frame >= max(0, len(s.anim_all_pts) - 1))
        idx = max(0, min(int(s.anim_frame), len(self._cum_dists) - 1))
        current_m = self._cum_dists[idx] + max(0.0, float(s.anim_frac))
        return max(0.0, min(1.0, current_m / total_m))

    def seek_ratio(self, ratio: float):
        """Move the visual playback head to an arc-length ratio of the route."""
        s = self.state
        if not s.anim_all_pts:
            return
        ratio = max(0.0, min(1.0, float(ratio)))
        total_m = float(self._cum_dists[-1]) if self._cum_dists else 0.0
        if ratio >= 1.0:
            s.anim_frame = len(s.anim_all_pts) - 1
            s.anim_frac = 0.0
        elif total_m <= 1e-9 or not self._seg_dists:
            s.anim_frame = min(
                len(s.anim_all_pts) - 1,
                int(round(ratio * max(0, len(s.anim_all_pts) - 1))),
            )
            s.anim_frac = 0.0
        else:
            target_m = ratio * total_m
            idx = int(np.searchsorted(self._cum_dists, target_m, side="right") - 1)
            idx = max(0, min(idx, len(self._seg_dists) - 1))
            s.anim_frame = idx
            s.anim_frac = max(0.0, target_m - self._cum_dists[idx])
        self._last_time = time.time()
        s.need_redraw = True

    def speed_up(self):
        maximum = self.max_speed_multiplier()
        levels = [
            level for level in (0.25, 0.5, 1, 2, 5, 10, 20, 30, 40, 60, 80, 100)
            if level <= maximum
        ]
        if not levels or levels[-1] < maximum:
            levels.append(maximum)
        current = self.state.anim_speed
        for lvl in levels:
            if current < lvl - 0.01:
                self.state.anim_speed = float(lvl)
                self.state.status_message = f"速度: {lvl}x"
                self.state.need_redraw = True
                return
        self.state.anim_speed = float(levels[-1])
        self.state.status_message = f"速度: {levels[-1]}x (最大)"
        self.state.need_redraw = True

    def speed_down(self):
        maximum = self.max_speed_multiplier()
        levels = sorted({
            0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 60.0,
            maximum,
        }, reverse=True)
        levels = [level for level in levels if level <= maximum]
        current = self.state.anim_speed
        for lvl in levels:
            if current > lvl + 0.01:
                self.state.anim_speed = float(lvl)
                self.state.status_message = f"速度: {lvl}x"
                self.state.need_redraw = True
                return
        self.state.anim_speed = 0.25
        self.state.status_message = "速度: 0.25x (最慢)"
        self.state.need_redraw = True

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _sample_path_at(self, distance_m: float) -> Tuple[float, float]:
        """Interpolate a pixel position at an absolute path arc length."""
        pts = self.state.anim_all_pts
        if not pts:
            return 0.0, 0.0
        if len(pts) == 1 or not self._seg_dists:
            return float(pts[0][0]), float(pts[0][1])

        distance_m = max(0.0, min(float(distance_m), self._cum_dists[-1]))
        idx = int(np.searchsorted(self._cum_dists, distance_m, side="right") - 1)
        idx = max(0, min(idx, len(self._seg_dists) - 1))
        segment_m = self._seg_dists[idx]
        ratio = 0.0 if segment_m <= 1e-9 else (
            distance_m - self._cum_dists[idx]
        ) / segment_m
        x = pts[idx][0] + ratio * (pts[idx + 1][0] - pts[idx][0])
        y = pts[idx][1] + ratio * (pts[idx + 1][1] - pts[idx][1])
        return float(x), float(y)

    def _current_speed_mps(self, idx: int) -> float:
        s = self.state
        current_type = (
            s.anim_all_types[idx]
            if idx < len(s.anim_all_types)
            else "work"
        )
        base_speed = self._base_speed_mps(current_type)
        return base_speed * s.anim_speed

    def _base_speed_mps(self, segment_type: str) -> float:
        params = dict(getattr(self.state, "harvester_params", {}) or {})
        if str(segment_type).startswith("turn"):
            key, fallback = "turn_speed_mps", self.TURN_SPEED_MPS
        elif str(segment_type) in ("entry", "exit"):
            key, fallback = "entry_speed_mps", self.ENTRY_SPEED_MPS
        else:
            key, fallback = "work_speed_mps", self.WORK_SPEED_MPS
        value = float(params.get(key, fallback))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a finite positive value")
        return value

    def _pose_segment_at(self, absolute_m: float):
        if not self._pose_segments:
            return None, 0.0
        distance = max(0.0, float(absolute_m))
        for index, segment in enumerate(self._pose_segments):
            if distance < segment["end_m"] - 1e-9 or index == len(self._pose_segments) - 1:
                return segment, max(
                    0.0, min(distance - segment["start_m"], segment["cumulative"][-1])
                )
        segment = self._pose_segments[-1]
        return segment, float(segment["cumulative"][-1])

    def get_interpolated_position(self):
        """Return a nonholonomic tracked-vehicle pose on the planned path.

        Position, heading and curvature all come from path arc length. The
        machine body therefore remains tangent to the trajectory, without the
        temporal heading lag that previously looked like lateral drift.
        """
        s = self.state
        pts = s.anim_all_pts
        if not pts:
            return None, 0.0

        idx = min(s.anim_frame, len(pts) - 1)
        absolute_m = (
            self._cum_dists[min(idx, len(self._cum_dists) - 1)]
            + max(0.0, s.anim_frac)
        )
        total_m = self._cum_dists[-1] if self._cum_dists else 0.0
        absolute_m = max(0.0, min(absolute_m, total_m))
        pose_segment, local_m = self._pose_segment_at(absolute_m)
        if pose_segment is None:
            return self._sample_path_at(absolute_m), s.anim_heading_rad
        heading_window_m = max(
            0.01,
            float(self.cfg.section("path_planning").get(
                "pose_heading_window_m", 0.30
            )),
        )
        reversing = pose_segment["segment_type"] == "turn_reverse"
        x, y, heading = pose_at_distance(
            pose_segment["points"],
            pose_segment["lengths"],
            pose_segment["cumulative"],
            local_m,
            heading_window_m,
            reversing=reversing,
        )

        half_window = max(0.08, heading_window_m * 0.5)
        sample_before = max(0.0, local_m - half_window)
        sample_after = min(float(pose_segment["cumulative"][-1]), local_m + half_window)
        _, _, heading_before = pose_at_distance(
            pose_segment["points"], pose_segment["lengths"],
            pose_segment["cumulative"], sample_before, heading_window_m,
            reversing=reversing,
        )
        _, _, heading_after = pose_at_distance(
            pose_segment["points"], pose_segment["lengths"],
            pose_segment["cumulative"], sample_after, heading_window_m,
            reversing=reversing,
        )
        curvature_span = max(1e-6, sample_after - sample_before)
        curvature = self._wrap_angle(
            heading_after - heading_before
        ) / curvature_span

        # Differential-track / ICR kinematics:
        # v_l = v(1 - B*k/2), v_r = v(1 + B*k/2).
        track_gauge_m = float(
            s.harvester_params.get("track_gauge_m", self.cfg.TRACK_GAUGE_M)
        )
        icr_factor = max(
            0.5,
            float(self.cfg.section("animation").get(
                "track_icr_factor", 1.0
            )),
        )
        effective_gauge_m = track_gauge_m * icr_factor
        center_speed_mps = self._base_speed_mps(pose_segment["segment_type"]) * s.anim_speed
        if reversing:
            center_speed_mps *= -1.0
        left_speed_mps = center_speed_mps * (
            1.0 - 0.5 * effective_gauge_m * curvature
        )
        right_speed_mps = center_speed_mps * (
            1.0 + 0.5 * effective_gauge_m * curvature
        )

        s.anim_heading_rad = heading
        s.anim_curvature_1pm = curvature
        s.anim_track_left_mps = left_speed_mps
        s.anim_track_right_mps = right_speed_mps
        s.anim_turn_radius_m = (
            abs(1.0 / curvature)
            if abs(curvature) > 1e-6
            else float("inf")
        )
        return (x, y), heading

    def get_remaining_seconds(self) -> float:
        """计算剩余作业时间（秒）"""
        s = self.state
        if not s.anim_active or not s.anim_all_pts:
            return 0.0
        cur_idx = min(s.anim_frame, len(s.anim_all_pts) - 1)
        # 使用实际段间距离计算剩余距离
        remaining_m = sum(self._seg_dists[cur_idx:]) if cur_idx < len(self._seg_dists) else 0
        remaining_m -= s.anim_frac  # 减去当前段已走过的亚帧余量
        # 当前速度
        cur_type = s.anim_all_types[cur_idx] if cur_idx < len(s.anim_all_types) else "work"
        speed_mps = self._base_speed_mps(cur_type)
        effective_speed = speed_mps * s.anim_speed
        return remaining_m / effective_speed if effective_speed > 0.01 else 0.0

    def set_speed_ratio(self, ratio: float):
        """Set the visual playback multiplier."""
        self.state.anim_speed = max(
            0.25,
            min(self.max_speed_multiplier(), ratio),
        )
        self._last_time = time.time()
        self.state.need_redraw = True

    def _gen_report(self):
        s = self.state
        total_len = work_len = 0.0
        # 优先使用自动路径，否则使用手动路径点
        if s.auto_path:
            for seg in s.auto_path:
                seg_len = 0.0
                for i in range(len(seg.points) - 1):
                    d = self.geo.pixel_distance_m(
                        (seg.points[i].pixel_x, seg.points[i].pixel_y),
                        (seg.points[i + 1].pixel_x, seg.points[i + 1].pixel_y))
                    seg_len += d
                total_len += seg_len
                if seg.segment_type == "work":
                    work_len += seg_len
        elif s.path_points and len(s.path_points) >= 2:
            for i in range(len(s.path_points) - 1):
                d = self.geo.pixel_distance_m(s.path_points[i], s.path_points[i + 1])
                total_len += d
                # 尊重 path_status: 1=作业, 0=转向
                seg_st = s.path_status[i] if i < len(s.path_status) else 1
                if seg_st == 1:
                    work_len += d
        cut_w = s.harvester_params.get("cutter_width_m", self.cfg.CUTTER_WIDTH_M)
        track_w = s.harvester_params.get("track_width_m", self.cfg.TRACK_WIDTH_M)
        track_gauge = s.harvester_params.get("track_gauge_m", self.cfg.TRACK_GAUGE_M)
        harvest_area = work_len * cut_w
        field_area = s.field_area_m2 if s.field_area_m2 > 0 else 0.0
        work_efficiency_pct = (work_len / total_len * 100) if total_len > 0 else 0
        s.anim_report = (
            f"作业总路径: {total_len:.1f} m\n"
            f"作业段:     {work_len:.1f} m\n"
            f"转弯段:     {total_len - work_len:.1f} m\n"
            f"路径作业效率: {s.last_field_efficiency or work_efficiency_pct:.1f}%\n"
            f"---\n"
            f"割台理论扫掠面积: {harvest_area:.1f} m²\n"
            f"田块总面积:   {field_area:.1f} m² ({field_area/666.67:.2f} 亩)\n"
            f"规划目标几何覆盖: {s.last_planned_target_geometric_coverage_pct:.1f}%\n"
            f"作物掩膜几何覆盖: {s.last_crop_mask_geometric_coverage_pct:.1f}%\n"
            f"原始识别掩膜几何覆盖: {s.last_detected_mask_geometric_coverage_pct:.1f}%\n"
            f"预计碾压率: {s.last_rolling_rate:.1f}%\n"
            f"---\n"
            f"收割机: 割台{cut_w:.1f}m 履带{track_w:.2f}m 中心距{track_gauge:.2f}m"
        )
        s.status_message = "预演示完成，已生成作业报告"
        s.need_redraw = True
        self.log.info(f"\n{s.anim_report}")
