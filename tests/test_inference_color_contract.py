from types import SimpleNamespace

import numpy as np


def test_model_engine_adapts_rgb_to_ultralytics_bgr_numpy_contract():
    """DECISION-005: adapt RGB once before Ultralytics reverses BGR to RGB."""
    from inference import ModelEngine

    class FakeUltralyticsModel:
        def __init__(self):
            self.captured = None

        def predict(self, image, **_kwargs):
            self.captured = image
            return ["ok"]

    engine = ModelEngine()
    engine._loaded = True
    engine._model = FakeUltralyticsModel()
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 0] = [11, 22, 33]

    assert engine.predict(rgb) == ["ok"]
    assert engine._model.captured[0, 0].tolist() == [33, 22, 11]

    assert engine.predict([rgb]) == ["ok"]
    assert engine._model.captured[0][0, 0].tolist() == [33, 22, 11]


class _FakeEngine:
    def __init__(self):
        self.captured = []
        self.confidences = []

    def is_loaded(self):
        return True

    def predict(self, images, conf=None, iou=None):
        batch = images if isinstance(images, list) else [images]
        self.captured.extend(batch)
        self.confidences.append(conf)
        return [SimpleNamespace(masks=None) for _ in batch]


def test_tiled_inference_keeps_rgb_channel_order():
    """DECISION-005: tiled inference stays RGB until the ModelEngine boundary."""
    from inference import TiledInference

    engine = _FakeEngine()
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    image[0, 0] = [1, 2, 3]

    result = TiledInference(engine).run(image, capture_sz=640, overlap=0.0)

    assert result is not None
    assert engine.captured
    assert engine.captured[0][0, 0].tolist() == [1, 2, 3]


def test_source_tif_inference_keeps_rgb_channel_order(monkeypatch):
    """DECISION-004/005: source tiles stay RGB until the ModelEngine boundary."""
    import inference
    from state import AppState

    engine = _FakeEngine()
    runner = inference.InferenceRunner(AppState(), engine, geo=None)
    runner._update = lambda **_kwargs: True

    source_tile = np.zeros((640, 640, 3), dtype=np.uint8)
    source_tile[0, 0] = [11, 22, 33]

    monkeypatch.setattr(
        "raster_preprocessing.read_rgb_raster",
        lambda *args, **kwargs: (source_tile.copy(), {"fake": True}),
    )

    result, preprocessing = runner._run_source_tif_tiles(
        object(),
        x0_src=0,
        y0_src=0,
        crop_w=640,
        crop_h=640,
        out_w=640,
        out_h=640,
    )

    assert result.shape == (640, 640)
    assert preprocessing["valid_tile_count"] == 1
    assert engine.captured
    assert engine.captured[0][0, 0].tolist() == [11, 22, 33]
    assert engine.confidences == [0.25]


def test_source_tile_blend_weight_prefers_tile_centres_and_keeps_edge_support():
    """DECISION-004/009: overlap fusion must downweight, not hard-paste, tile edges."""
    from inference import _gaussian_tile_weight

    weight = _gaussian_tile_weight(64, 64)

    assert weight.shape == (64, 64)
    assert weight[32, 32] > weight[0, 0]
    assert weight[0, 0] >= 0.05
    assert np.isclose(weight[0, 0], weight[-1, -1], atol=1e-6)
    assert np.isclose(weight[0, -1], weight[-1, 0], atol=1e-6)


def test_fused_probability_uses_explicit_threshold():
    """DECISION-009: fused masks use one auditable threshold across tiles."""
    from inference import _probability_to_mask

    probability = np.asarray([[0.49, 0.50, 0.51]], dtype=np.float32)

    assert _probability_to_mask(probability).tolist() == [[0, 255, 255]]


def test_context_separator_filter_keeps_long_parallel_components():
    """DECISION-009: ROI filtering preserves the separator acceptance rule."""
    from inference import _filter_context_separators

    reference = np.zeros((128, 128), dtype=np.uint8)
    reference[64:68, 8:120] = 255
    candidate = np.zeros((128, 128), dtype=np.uint8)
    candidate[70, 8:120] = 255
    candidate[10:20, 40] = 255

    filtered = _filter_context_separators(
        candidate,
        reference,
        meters_per_pixel=0.02,
        minimum_length_m=1.0,
        angle_tolerance_deg=10.0,
    )

    assert filtered[70, 20] == 255
    assert filtered[15, 40] == 0


def test_contact_boundary_keeps_only_evidence_between_two_instances():
    """DECISION-009: do not flatten touching instance evidence into a union."""
    from inference import _contact_boundary_from_instances

    masks = np.zeros((2, 64, 64), dtype=np.float32)
    masks[0, 6:58, 4:36] = 1.0
    masks[1, 6:58, 32:60] = 1.0

    boundary = _contact_boundary_from_instances(
        masks,
        np.asarray([0.95, 0.85], dtype=np.float32),
        minimum_confidence=0.25,
        border_guard_px=4,
    )

    assert np.count_nonzero(boundary[:, 30:38]) > 0
    assert np.count_nonzero(boundary[:, :4]) == 0
    assert np.count_nonzero(boundary[:, -4:]) == 0


def test_context_fusion_recovers_nearby_support_and_preserves_long_separator():
    """DECISION-004/009: context fills local misses but keeps instance borders."""
    from inference import _fuse_context_evidence

    primary = np.zeros((140, 100), dtype=np.uint8)
    primary[10:130, 10:45] = 255
    primary[10:130, 55:90] = 255
    coverage = np.full((140, 100), 2, dtype=np.uint16)
    support = np.zeros((140, 100), dtype=np.uint16)
    support[10:130, 10:90] = 2
    boundaries = np.zeros((140, 100), dtype=np.uint16)
    boundaries[10:130, 50] = 2

    fused, diagnostics = _fuse_context_evidence(
        primary,
        support,
        boundaries,
        coverage,
        meters_per_pixel=0.10,
        recovery_distance_m=0.75,
        separator_min_length_m=2.0,
    )

    assert diagnostics["context_recovered_pixels"] > 0
    assert diagnostics["context_separator_pixels"] > 0
    assert np.count_nonzero(fused[20:120, 49:52]) == 0
    assert np.count_nonzero(fused[20:120, 46:49]) > 0


def test_context_fusion_default_requires_repeated_support():
    """DECISION-004/009: weak single-context support is not mask recovery."""
    from inference import _fuse_context_evidence

    primary = np.zeros((60, 60), dtype=np.uint8)
    coverage = np.full((60, 60), 2, dtype=np.uint16)
    support = np.zeros((60, 60), dtype=np.uint16)
    support[20:40, 10:50] = 1
    boundaries = np.zeros((60, 60), dtype=np.uint16)

    fused, diagnostics = _fuse_context_evidence(
        primary,
        support,
        boundaries,
        coverage,
        meters_per_pixel=0.10,
        recovery_distance_m=0.75,
    )

    assert diagnostics["context_recovered_pixels"] == 0
    assert np.count_nonzero(fused) == 0
