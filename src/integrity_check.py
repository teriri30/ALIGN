"""Verify an exported route manifest without starting the GUI."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from provenance import file_sha256, verify_record_integrity


def _resolve_recorded_path(recorded: str, manifest_path: Path) -> Path:
    path = Path(recorded)
    if path.is_file():
        return path
    sibling = manifest_path.parent / path.name
    return sibling


def verify_manifest(manifest_file: os.PathLike | str, verify_sources: bool = False) -> list[str]:
    """DECISION-001: independently verify a formal route and evidence chain."""
    manifest_path = Path(manifest_file).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []

    output = payload.get("output", {})
    output_path = _resolve_recorded_path(str(output.get("path", "")), manifest_path)
    if not output_path.is_file():
        errors.append(f"output file is missing: {output_path}")
    elif file_sha256(output_path) != str(output.get("sha256", "")):
        errors.append("output SHA-256 does not match manifest")

    stages = payload.get("analysis", {}).get("stage_provenance", {})
    for stage in ("inference", "mask", "path"):
        record = stages.get(stage) or {}
        if not record:
            errors.append(f"{stage} provenance record is missing")
            continue
        try:
            verify_record_integrity(record)
        except ValueError as exc:
            errors.append(f"{stage} provenance is invalid: {exc}")

    if verify_sources:
        for label, item in (
            ("source image", payload.get("source_image", {})),
            ("model", payload.get("model", {})),
        ):
            recorded_path = str(item.get("path", ""))
            expected = str(item.get("sha256", ""))
            path = _resolve_recorded_path(recorded_path, manifest_path)
            if not path.is_file():
                errors.append(f"{label} file is missing: {path}")
            elif not expected or file_sha256(path) != expected:
                errors.append(f"{label} SHA-256 does not match manifest")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Path to <route>.manifest.json")
    parser.add_argument(
        "--verify-sources",
        action="store_true",
        help="also hash the recorded source GeoTIFF and model",
    )
    args = parser.parse_args(argv)
    errors = verify_manifest(args.manifest, verify_sources=args.verify_sources)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: manifest and recorded artifacts are internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
