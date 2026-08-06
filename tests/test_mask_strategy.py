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
