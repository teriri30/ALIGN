import numpy as np


def test_row_angle_estimation_does_not_depend_on_numpy_covariance(monkeypatch):
    from row_geometry import estimate_row_angle

    mask = np.zeros((400, 600), dtype=np.uint8)
    for offset in range(-80, 81, 20):
        points = np.asarray([
            (x, int(round(0.35 * x + 120 + offset)))
            for x in range(40, 560)
        ], dtype=np.int32)
        valid = (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
        mask[points[valid, 1], points[valid, 0]] = 255

    monkeypatch.setattr(
        np,
        "cov",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("estimate_row_angle must not call np.cov")
        ),
    )

    angle = estimate_row_angle(mask)

    assert abs(angle - np.arctan(0.35)) < np.deg2rad(3.0)


def test_row_angle_estimation_uses_component_axes_when_global_layout_is_skewed():
    """Separated planting regions must not let their centroids define row angle."""
    import cv2
    from row_geometry import estimate_row_angle

    mask = np.zeros((700, 900), dtype=np.uint8)
    for y_value in range(390, 611, 20):
        cv2.line(mask, (40, y_value), (360, y_value), 255, 6)
    for y_value in range(90, 311, 20):
        cv2.line(mask, (540, y_value), (860, y_value), 255, 6)

    angle = estimate_row_angle(mask)
    error = min(abs(angle), abs(np.pi - angle))

    assert error < np.deg2rad(3.0)


def test_profile_gate_edges_isolate_band_next_to_broad_uniform_block():
    """宽等距块旁的四行带不能被宽块尾部拉偏中心线门控。"""
    from row_geometry import _profile_run_gates

    runs = [(1556, 2435), (2450, 2521), (2535, 2575)]
    gates = _profile_run_gates(runs, height=3600)

    # The middle run is the target four-row band.  Its gate must start after
    # the broad block and end before the next narrow run.
    assert gates[1] == (2442, 2528)
    assert gates[1][0] > runs[0][1]
    assert gates[1][1] < runs[2][0]


def test_regularize_output_preserves_raw_parallel_residual_in_processed_mask(monkeypatch):
    import row_geometry

    raw = np.zeros((100, 160), dtype=np.uint8)
    raw[30:36, 5:150] = 255      # row-parallel residual that must remain visible
    raw[5:85, 152:158] = 255     # crosswise headland residual
    rebuilt = np.zeros_like(raw)
    rebuilt[50:56, 5:150] = 255  # clean regularized body row

    def fake_residual_layers(raw_mask, rebuilt_mask, angle, angle_thresh_deg=60.0, min_area_ratio=0.001):
        headland = np.zeros_like(raw_mask)
        headland[5:85, 152:158] = raw_mask[5:85, 152:158]
        body_residual = np.zeros_like(raw_mask)
        body_residual[30:36, 5:150] = raw_mask[30:36, 5:150]
        return headland, body_residual

    monkeypatch.setattr(row_geometry, "residual_mask_layers", fake_residual_layers)

    visible_body = np.bitwise_or(rebuilt, fake_residual_layers(raw, rebuilt, 0.0)[1])
    planning_support = np.bitwise_or(visible_body, fake_residual_layers(raw, rebuilt, 0.0)[0])

    assert visible_body[32, 30] == 255
    assert visible_body[40, 155] == 0
    assert planning_support[40, 155] == 255


def test_detached_parallel_fragment_outside_body_extent_does_not_enter_body_residual():
    from row_geometry import residual_mask_layers

    raw = np.zeros((120, 220), dtype=np.uint8)
    rebuilt = np.zeros_like(raw)
    rebuilt[40:46, 20:150] = 255
    raw[40:46, 20:150] = 255
    raw[85:91, 166:216] = 255

    headland, body_residual, uncertain = residual_mask_layers(
        raw,
        rebuilt,
        main_angle=0.0,
        meters_per_px=0.1,
        min_area_ratio=0.001,
    )

    assert body_residual[87, 190] == 0
    assert headland[87, 190] == 255
    assert uncertain[87, 190] == 0


def test_detached_parallel_fragment_inside_body_extent_becomes_uncertain():
    """DECISION-009: projection overlap alone cannot promote a detached fragment."""
    from row_geometry import residual_mask_layers

    raw = np.zeros((120, 240), dtype=np.uint8)
    rebuilt = np.zeros_like(raw)
    rebuilt[35:43, 20:190] = 255
    raw[35:43, 20:190] = 255
    raw[80:88, 145:215] = 255

    headland, body_residual, uncertain = residual_mask_layers(
        raw,
        rebuilt,
        main_angle=0.0,
        meters_per_px=0.1,
        min_area_ratio=0.001,
    )

    assert np.count_nonzero(body_residual[80:88, 145:215]) == 0
    assert np.count_nonzero(headland[80:88, 145:215]) == 0
    assert np.count_nonzero(uncertain[80:88, 145:215]) == 560


def test_adjacent_parallel_residual_remains_attached_to_body():
    """DECISION-009: nearby raw edge evidence remains part of the work body."""
    from row_geometry import residual_mask_layers

    raw = np.zeros((100, 180), dtype=np.uint8)
    rebuilt = np.zeros_like(raw)
    rebuilt[40:48, 20:160] = 255
    raw[40:51, 20:160] = 255

    headland, body_residual, uncertain = residual_mask_layers(
        raw,
        rebuilt,
        main_angle=0.0,
        meters_per_px=0.05,
        min_area_ratio=0.001,
    )

    assert np.count_nonzero(body_residual[48:51, 20:160]) == 420
    assert np.count_nonzero(headland) == 0
    assert np.count_nonzero(uncertain) == 0


def test_attached_oblique_block_residual_does_not_become_work_body():
    """Attached residuals must still look like row strips before joining body."""
    import cv2
    from row_geometry import residual_mask_layers

    raw = np.zeros((160, 220), dtype=np.uint8)
    rebuilt = np.zeros_like(raw)
    rebuilt[58:66, 20:180] = 255
    raw[58:66, 20:180] = 255
    cv2.fillPoly(
        raw,
        [np.asarray([[22, 66], [185, 112], [185, 145], [22, 96]], dtype=np.int32)],
        255,
    )

    headland, body_residual, uncertain = residual_mask_layers(
        raw,
        rebuilt,
        main_angle=0.0,
        meters_per_px=0.1,
        min_area_ratio=0.001,
    )

    residual_area = int(np.count_nonzero(raw & cv2.bitwise_not(rebuilt)))
    assert np.count_nonzero(body_residual) == 0
    assert np.count_nonzero(headland) == 0
    assert np.count_nonzero(uncertain) == residual_area


def test_terminal_band_gap_uses_stricter_threshold_than_internal_gap():
    """DECISION-009: a detached terminal fragment cannot use the internal gap limit."""
    from row_geometry import close_band_support_gaps

    seed = np.zeros(100, dtype=bool)
    seed[10:70] = True
    seed[76:98] = True

    closed = close_band_support_gaps(
        seed,
        internal_max_gap=10,
        terminal_max_gap=3,
        terminal_guard=5,
    )

    assert not np.any(closed[70:76])

    seed[76:98] = False
    seed[76:86] = True
    closed_internal = close_band_support_gaps(
        seed,
        internal_max_gap=10,
        terminal_max_gap=3,
        terminal_guard=5,
    )
    assert np.all(closed_internal[70:76])


def test_internal_support_bridge_joins_real_row_parts_but_not_short_terminal_tail():
    from row_geometry import bridge_internal_support_gaps

    internal = np.zeros(120, dtype=bool)
    internal[10:55] = True
    internal[68:108] = True
    bridged = bridge_internal_support_gaps(
        internal,
        max_gap=20,
        terminal_guard=5,
        min_adjacent_run=20,
    )
    assert np.all(bridged[55:68])

    terminal = np.zeros(120, dtype=bool)
    terminal[10:80] = True
    terminal[100:120] = True
    guarded = bridge_internal_support_gaps(
        terminal,
        max_gap=20,
        terminal_guard=5,
        min_adjacent_run=15,
    )
    assert not np.any(guarded[80:88])


def test_generated_body_keeps_internal_gap_fill_but_removes_isolated_fragment():
    from row_geometry import constrain_generated_body_to_raw_support

    raw = np.zeros((80, 180), dtype=np.uint8)
    raw[35:41, 20:80] = 255
    raw[35:41, 98:150] = 255
    generated = raw.copy()
    generated[35:41, 80:98] = 255
    generated[60:66, 160:168] = 255

    constrained = constrain_generated_body_to_raw_support(
        generated,
        raw,
        main_angle=0.0,
        meters_per_px=0.1,
        config={"row_gap_close_m": 2.2, "row_gap_close_width_m": 0.2},
    )

    assert constrained[37, 88] == 255
    assert constrained[62, 164] == 0


def test_default_row_gap_repair_bridges_visible_internal_breaks():
    from row_geometry import repair_row_gaps

    raw = np.zeros((80, 180), dtype=np.uint8)
    raw[35:41, 20:80] = 255
    raw[35:41, 98:150] = 255

    repaired = repair_row_gaps(raw, main_angle=0.0, meters_per_px=0.1, config={})

    assert repaired[37, 88] == 255


def test_row_angle_rank_lists_sorted_candidates_and_wrapper_keeps_contract():
    """排名接口按得分降序返回全候选；estimate_row_angle 保持既有契约。"""
    from row_geometry import _rank_row_angles, estimate_row_angle

    mask = np.zeros((420, 640), dtype=np.uint8)
    for y_value in range(30, 391, 30):
        mask[y_value:y_value + 12, 20:620] = 255

    ranked = _rank_row_angles(mask)

    assert len(ranked) >= 2
    scores = [score for _angle, score in ranked]
    assert scores == sorted(scores, reverse=True)
    assert estimate_row_angle(mask) == ranked[0][0]


def test_comb_run_quality_counts_bands_but_not_broad_uniform_blocks():
    """组合条带数只统计 0.25~4 m 的横向条带，6 m 宽等距块必须被排除。"""
    from row_geometry import _comb_run_quality

    mask = np.zeros((1600, 600), dtype=np.uint8)
    y_value = 0
    for _ in range(10):
        mask[y_value:y_value + 50, :] = 255
        y_value += 100
    mask[y_value:y_value + 600, :] = 255

    assert _comb_run_quality(mask, 0.0, 0.01, {}) == 10


def test_row_angle_tiebreak_keeps_clear_winner_without_quality_scan(monkeypatch):
    """清晰胜出时不得触发任何横剖质量扫描。"""
    import math

    import row_geometry

    scan_calls = {"count": 0}

    def fake_quality(work, angle_deg, work_mpp, cfg):
        scan_calls["count"] += 1
        return 0

    monkeypatch.setattr(
        row_geometry,
        "_rank_row_angles",
        lambda mask: [(math.radians(20.0), 0.10), (math.radians(50.0), 0.05)],
    )
    monkeypatch.setattr(row_geometry, "_comb_run_quality", fake_quality)

    angle, diagnostics = row_geometry._resolve_row_angle(
        np.zeros((10, 10), dtype=np.uint8), 0.01, {}
    )

    assert abs(math.degrees(angle) - 20.0) < 1e-6
    assert diagnostics["tie_triggered"] is False
    assert diagnostics["reason"] == "clear_winner"
    assert scan_calls["count"] == 0


def test_row_angle_tiebreak_switches_when_challenger_family_has_clear_band_evidence(monkeypatch):
    """近平分时，挑战家族组合条带数达到 max(2, 25%) 优势才允许切换。"""
    import math

    import row_geometry

    monkeypatch.setattr(
        row_geometry,
        "_rank_row_angles",
        lambda mask: [
            (math.radians(112.77), 0.0685),
            (math.radians(112.27), 0.0679),
            (math.radians(108.77), 0.0635),
        ],
    )
    monkeypatch.setattr(
        row_geometry,
        "_comb_run_quality",
        lambda work, angle_deg, work_mpp, cfg: (
            10 if 107.5 <= angle_deg <= 108.55 else 4
        ),
    )

    angle, diagnostics = row_geometry._resolve_row_angle(
        np.zeros((10, 10), dtype=np.uint8), 0.01, {}
    )

    assert diagnostics["tie_triggered"] is True
    assert diagnostics["switched"] is True
    assert 107.5 <= math.degrees(angle) <= 108.55
    assert diagnostics["scans"]
    assert diagnostics["family_best"]


def test_row_angle_tiebreak_keeps_winner_on_marginal_quality_advantage(monkeypatch):
    """挑战家族未达优势门槛时不得切换行向。"""
    import math

    import row_geometry

    monkeypatch.setattr(
        row_geometry,
        "_rank_row_angles",
        lambda mask: [
            (math.radians(112.77), 0.0685),
            (math.radians(108.77), 0.0679),
        ],
    )
    monkeypatch.setattr(
        row_geometry,
        "_comb_run_quality",
        lambda work, angle_deg, work_mpp, cfg: (
            8
            if abs(angle_deg - 112.77) <= 1.5 or abs(angle_deg - 108.77) <= 1.5
            else 4
        ),
    )

    angle, diagnostics = row_geometry._resolve_row_angle(
        np.zeros((10, 10), dtype=np.uint8), 0.01, {}
    )

    assert diagnostics["tie_triggered"] is True
    assert diagnostics["switched"] is False
    assert abs(math.degrees(angle) - 112.77) < 1e-9


def test_degenerate_rebuild_gate_flags_collapsed_rebuild_only():
    """候选≥8 且重建<40% 必须给出显式失败理由；正常与小型圈选不得误报。"""
    from row_geometry import _degenerate_rebuild_reason

    reason = _degenerate_rebuild_reason(12, 2, {})
    assert reason is not None
    assert "candidates=12" in reason
    assert "rebuilt=2" in reason
    assert _degenerate_rebuild_reason(12, 11, {}) is None
    assert _degenerate_rebuild_reason(5, 1, {}) is None
    assert _degenerate_rebuild_reason(
        20, 5, {"degenerate_rebuild_ratio_min": 0.3}
    ) is not None
    assert _degenerate_rebuild_reason(
        6, 1, {"degenerate_candidates_min": 5}
    ) is not None
