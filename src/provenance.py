"""Deterministic provenance helpers for scientific processing artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from version import APP_VERSION
PROVENANCE_SCHEMA_VERSION = 1


class ProvenanceError(ValueError):
    """Raised when a cached or exported artifact cannot be proven current."""


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: os.PathLike | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    target = os.path.abspath(os.fspath(path))
    digest = hashlib.sha256()
    with open(target, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(value.shape)))
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def artifact_sha256(artifact: Any) -> str:
    if isinstance(artifact, np.ndarray):
        return array_sha256(artifact)
    return canonical_sha256(artifact)


def make_stage_record(
    stage: str,
    algorithm_version: str,
    inputs: Mapping[str, Any],
    artifact: Any,
) -> dict:
    input_payload = jsonable(dict(inputs or {}))
    record = {
        "schema": PROVENANCE_SCHEMA_VERSION,
        "stage": str(stage),
        "algorithm_version": str(algorithm_version),
        "inputs": input_payload,
        "input_fingerprint": canonical_sha256(input_payload),
        "artifact_sha256": artifact_sha256(artifact),
    }
    record["fingerprint"] = canonical_sha256(record)
    return record


def verify_stage_record(
    record: Mapping[str, Any],
    stage: str,
    current_inputs: Mapping[str, Any],
    artifact: Any,
) -> bool:
    """DECISION-002: reject a stage whose inputs or artifact no longer match."""
    if not isinstance(record, Mapping):
        raise ProvenanceError(f"{stage} provenance record is missing")
    if int(record.get("schema", 0)) != PROVENANCE_SCHEMA_VERSION:
        raise ProvenanceError(f"{stage} provenance schema is unsupported")
    if str(record.get("stage", "")) != str(stage):
        raise ProvenanceError(f"{stage} provenance stage does not match")
    expected_inputs = canonical_sha256(dict(current_inputs or {}))
    if str(record.get("input_fingerprint", "")) != expected_inputs:
        raise ProvenanceError(f"{stage} input fingerprint does not match")
    expected_artifact = artifact_sha256(artifact)
    if str(record.get("artifact_sha256", "")) != expected_artifact:
        raise ProvenanceError(f"{stage} artifact hash does not match")
    return True


def verify_record_integrity(record: Mapping[str, Any]) -> bool:
    if not isinstance(record, Mapping) or not record.get("fingerprint"):
        raise ProvenanceError("stage provenance record is incomplete")
    payload = dict(record)
    fingerprint = str(payload.pop("fingerprint"))
    if canonical_sha256(payload) != fingerprint:
        raise ProvenanceError("stage provenance record fingerprint is invalid")
    return True


def git_revision(root: os.PathLike | str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.fspath(root),
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_is_dirty(root: os.PathLike | str) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=os.fspath(root),
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def code_identity(root: os.PathLike | str, relative_paths: list[str]) -> dict:
    root_path = Path(root).resolve()
    files = {}
    for relative in relative_paths:
        path = root_path / relative
        if not path.is_file():
            raise FileNotFoundError(f"scientific source file is missing: {relative}")
        files[str(relative).replace("\\", "/")] = file_sha256(path)
    return {"files": files, "fingerprint": canonical_sha256(files)}
