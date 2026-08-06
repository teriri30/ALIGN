"""Trusted local assets used by the repeatable demonstration workflow."""
from __future__ import annotations

import os
import json
from typing import Dict, List


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TEST_IMAGE = os.path.join(_REPO_ROOT, "data", "tif", "result.tif")
_DEFAULT_MODEL_DIR = os.path.join(_REPO_ROOT, "data", "models")

_PACKAGED_MODEL_SPECS = (
    {
        "name": "YOLO11m-seg（内置对比）",
        "filename": "yolo11m_seg_best.pt",
        "sha256": "93afb1ba0492266e22b9f1e16811cc31a28f52ac48c53bd46e0ddc740ea520ad",
    },
    {
        "name": "YOLO11s-seg（内置对比）",
        "filename": "yolo11s_seg_best.pt",
        "sha256": "3268e61f4f0e0975806124f3aa7202eb881529ee0cd6f6e5b42e18bd5598372c",
    },
    {
        "name": "YOLOv8s-seg（内置对比）",
        "filename": "yolov8s_seg_best.pt",
        "sha256": "103c2eadd2e6240819319b81f046549faaffa0f949368ec1d339e565241617b2",
    },
    {
        "name": "YOLO11s-seg（分蘖期12m）",
        "filename": "yolo11s_seg_tiller12m_best.pt",
        "sha256": "5f31cc3950fe803e0768b2763f0ee9dd2a3a01395aa67a362ee734d41f4d3969",
    },
)


def _first_existing(paths) -> str:
    for path in paths:
        candidate = str(path or "").strip()
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return ""


def _manifest_models(model_dir: str) -> List[Dict[str, str]]:
    registry_path = os.path.join(model_dir, "model_registry.json")
    try:
        with open(registry_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return []
    models = []
    packaged_labels = {
        str(spec["filename"]): str(spec["name"])
        for spec in _PACKAGED_MODEL_SPECS
    }
    for item in payload.get("models", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("name", "") or "").strip()
        sha256 = str(item.get("sha256", "") or "").strip()
        if not filename or not sha256:
            continue
        models.append({
            "name": str(
                item.get("display_name", "") or packaged_labels.get(filename, filename)
            ),
            "path": os.path.abspath(os.path.join(model_dir, filename)),
            "sha256": sha256,
            "trusted_builtin": True,
        })
    return models


def _packaged_models(model_dir: str) -> List[Dict[str, str]]:
    manifest_models = _manifest_models(model_dir)
    if manifest_models:
        return manifest_models
    models: List[Dict[str, str]] = []
    for spec in _PACKAGED_MODEL_SPECS:
        path = os.path.join(model_dir, spec["filename"])
        models.append(
            {
                "name": spec["name"],
                "path": os.path.abspath(path),
                "sha256": spec["sha256"],
                "trusted_builtin": True,
            }
        )
    return models


def discover_builtin_assets() -> Dict[str, object]:
    """Discover portable built-in demo assets with explicit availability status."""
    env_image = str(os.environ.get("REUI_TEST_IMAGE", "") or "").strip()
    env_model_dir = str(os.environ.get("REUI_TEST_MODEL_DIR", "") or "").strip()
    image = _first_existing(
        [
            env_image,
            _DEFAULT_TEST_IMAGE,
        ]
    )
    model_dir = ""
    for candidate in (env_model_dir, _DEFAULT_MODEL_DIR):
        if candidate and os.path.isdir(candidate):
            model_dir = os.path.abspath(candidate)
            break
    if model_dir:
        models = _packaged_models(model_dir)
    else:
        models = []
    existing_models = [model for model in models if os.path.isfile(model["path"])]
    if image and models and len(existing_models) == len(models):
        status = "available"
    elif image or existing_models:
        status = "partial"
    else:
        status = "missing"
    return {
        "image": image,
        "models": models,
        "existing_models": existing_models,
        "status": status,
        "model_dir": model_dir,
    }


# Back-compat constant used by older callers/tests. Prefer builtin_test_image().
BUILTIN_TEST_IMAGE = discover_builtin_assets()["image"] or _DEFAULT_TEST_IMAGE


def builtin_test_image(*, existing_only: bool = True) -> str:
    assets = discover_builtin_assets()
    path = str(assets.get("image") or "")
    if existing_only and not path:
        return ""
    if existing_only and not os.path.isfile(path):
        return ""
    return path or _DEFAULT_TEST_IMAGE


def builtin_test_models(*, existing_only: bool = True) -> List[Dict[str, str]]:
    assets = discover_builtin_assets()
    models = [dict(model) for model in assets.get("models") or []]
    if existing_only:
        models = [model for model in models if os.path.isfile(model["path"])]
    return models


def builtin_test_model_entry(model_path: str) -> Dict[str, str]:
    target = os.path.normcase(os.path.abspath(str(model_path or "")))
    for model in builtin_test_models(existing_only=False):
        if os.path.normcase(os.path.abspath(model["path"])) == target:
            return model
    return {}


def builtin_assets_status() -> str:
    return str(discover_builtin_assets().get("status") or "missing")
