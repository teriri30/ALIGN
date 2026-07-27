import numpy as np


def test_body_interval_extends_to_row_supported_end_beyond_high_quality_core():
    from row_geometry import estimate_body_interval

    foreground = np.zeros((80, 220), dtype=bool)
    # Repeated row lattice: the central part is strong, while the last 35 px is
    # weaker but still row-supported and should remain harvestable.
    for y in (12, 24, 36, 48, 60):
        foreground[y:y + 4, 20:150] = True
        foreground[y:y + 4, 150:185:2] = True
    runs = [(12, 16), (24, 28), (36, 40), (48, 52), (60, 64)]

    start, end, diagnostics = estimate_body_interval(
        foreground,
        runs,
        work_mpp=0.1,
        config={
            "body_quality_min": 0.20,
            "body_quality_max": 0.35,
            "body_continuity_min": 0.75,
            "body_contrast_min": 0.10,
            "body_gap_close_m": 0.5,
            "body_end_extension_m": 0.0,
            "body_min_length_m": 4.0,
            "body_support_extension_min_continuity": 0.45,
            "body_support_extension_max_gap_m": 0.8,
        },
    )

    assert start <= 22
    assert end >= 180
    assert diagnostics["support_extended_end_px"] >= 180
