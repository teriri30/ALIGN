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


def _machine_file_errors(
    path: Path,
    expected_count: int | None,
    protocol: dict,
) -> list[str]:
    """Validate the on-disk terminal projection, not only its hash."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return [f"machine output cannot be read: {path} ({exc})"]

    errors = []
    rows = [line for line in lines if line.startswith("$PATH,")]
    if expected_count is not None and len(rows) != int(expected_count):
        errors.append(
            "machine output point count does not match manifest: "
            f"{path} (expected {int(expected_count)}, got {len(rows)})"
        )

    final_record = str(protocol.get("final_record") or "")
    if final_record and (not lines or lines[-1] != final_record):
        errors.append(f"machine output final record does not match manifest: {path}")

    status_codes = protocol.get("status_codes") or {}
    expected_status = str(status_codes.get("file_record_status", "2"))
    for index, row in enumerate(rows):
        fields = row[:-1].split(",") if row.endswith("*") else []
        if len(fields) != 9 or fields[0] != "$PATH":
            errors.append(f"machine output record format is invalid: {path}")
            break
        if fields[2] != str(index):
            errors.append(
                f"machine output index is not contiguous: {path} (expected {index})"
            )
            break
        if expected_count is not None and fields[3] != str(int(expected_count)):
            errors.append(
                "machine output declared total does not match manifest: "
                f"{path} (record {index})"
            )
            break
        if fields[-1] != expected_status:
            errors.append(
                f"machine output status code does not match manifest: {path}"
            )
            break
    return errors


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

    path_protocol = output.get("path_protocol") or {}
    if str(output.get("format", "")).upper() == "$PATH_MACHINE_COMPAT":
        output_files = list(path_protocol.get("output_files") or [])
        if not output_files:
            output_files = [str(output.get("path", ""))]
        points_per_file = list(path_protocol.get("points_per_file") or [])
        integrity_entries = list(path_protocol.get("output_file_integrity") or [])
        expected_total = output.get("point_count")
        actual_total = 0
        for index, recorded in enumerate(output_files):
            entry = integrity_entries[index] if index < len(integrity_entries) else {}
            if not isinstance(entry, dict):
                entry = {}
            recorded_path = str(entry.get("path") or recorded)
            path = _resolve_recorded_path(recorded_path, manifest_path)
            expected_hash = str(entry.get("sha256") or "")
            if not path.is_file():
                errors.append(f"output file is missing: {path}")
                continue
            if expected_hash and file_sha256(path) != expected_hash:
                if index == 0:
                    message = "output SHA-256 does not match manifest"
                else:
                    message = f"output file SHA-256 does not match manifest: {path}"
                if message not in errors:
                    errors.append(message)
            expected_count = entry.get("point_count")
            if expected_count is None and index < len(points_per_file):
                expected_count = points_per_file[index]
            if expected_count is not None:
                try:
                    expected_count = int(expected_count)
                except (TypeError, ValueError):
                    errors.append(f"manifest point count is invalid: {path}")
                    expected_count = None
            try:
                rows = [
                    line
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.startswith("$PATH,")
                ]
            except (OSError, UnicodeError) as exc:
                errors.append(f"machine output cannot be read: {path} ({exc})")
                continue
            actual_total += len(rows)
            errors.extend(_machine_file_errors(path, expected_count, path_protocol))
        if expected_total is not None:
            try:
                if actual_total != int(expected_total):
                    errors.append(
                        "machine output total point count does not match manifest: "
                        f"(expected {int(expected_total)}, got {actual_total})"
                    )
            except (TypeError, ValueError):
                errors.append("manifest total point count is invalid")

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
