import inspect
import json
from pathlib import Path


EXPECTED_MODEL_DEFAULTS = {
    "capture_size": 640,
    "overlap": 0.40,
    "conf": 0.25,
    "iou": 0.45,
}


def test_tiled_inference_defaults_match_run_documentation():
    from inference import TiledInference

    signature = inspect.signature(TiledInference.run)
    assert signature.parameters["capture_sz"].default == EXPECTED_MODEL_DEFAULTS["capture_size"]
    assert signature.parameters["overlap"].default == EXPECTED_MODEL_DEFAULTS["overlap"]
    assert signature.parameters["conf"].default == EXPECTED_MODEL_DEFAULTS["conf"]
    assert signature.parameters["iou"].default == EXPECTED_MODEL_DEFAULTS["iou"]

    doc = inspect.getdoc(TiledInference.run)
    assert "overlap: 重叠率（0~1），默认 0.40，用于降低切片边缘漏检" in doc
    assert "conf: 模型置信度阈值，默认 0.25" in doc
    assert "iou: NMS 阈值，默认 0.45" in doc

    stale_fragments = (
        "默认 75%",
        "默认 0.1，优先召回",
        "默认 0.3",
        "让每个像素被多块覆盖后再融合",
    )
    for fragment in stale_fragments:
        assert fragment not in doc


def test_config_defaults_match_model_defaults_when_config_file_is_missing():
    from config import Config

    cfg = Config()
    previous_raw = dict(getattr(cfg, "_raw", {}))
    previous_loaded = getattr(cfg, "_loaded", False)
    try:
        cfg._raw = {}
        cfg._loaded = False
        assert cfg.TILE_CAPTURE_SIZE == EXPECTED_MODEL_DEFAULTS["capture_size"]
        assert cfg.TILE_OVERLAP == EXPECTED_MODEL_DEFAULTS["overlap"]
        assert cfg.MODEL_CONF == EXPECTED_MODEL_DEFAULTS["conf"]
        assert cfg.MODEL_IOU == EXPECTED_MODEL_DEFAULTS["iou"]
    finally:
        cfg._raw = previous_raw
        cfg._loaded = previous_loaded


def test_config_json_model_defaults_match_code_defaults():
    config_path = Path(__file__).resolve().parents[1] / "config.example.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = config["model"]

    assert model_config["tile_capture_size"] == EXPECTED_MODEL_DEFAULTS["capture_size"]
    assert model_config["tile_overlap"] == EXPECTED_MODEL_DEFAULTS["overlap"]
    assert model_config["conf_threshold"] == EXPECTED_MODEL_DEFAULTS["conf"]
    assert model_config["iou_threshold"] == EXPECTED_MODEL_DEFAULTS["iou"]
