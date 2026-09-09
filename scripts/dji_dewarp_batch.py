"""Batch dewarp DJI Matrice 4E JPGs using XMP DewarpData/K6 parameters.

The output keeps the source directory layout and writes a JSON manifest.  It
does not copy the source XMP block into the re-encoded JPEGs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


XMP_SCAN_BYTES = 262_144
TAG_NAMES = {
    "product_name": "ProductName",
    "drone_model": "DroneModel",
    "dewarp_flag": "DewarpFlag",
    "camera_serial": "CameraSerialNumber",
    "drone_serial": "DroneSerialNumber",
    "dewarp_data": "DewarpData",
    "dewarp_data_k6": "DewarpDataK6",
}
TAG_PATTERNS = {
    key: re.compile(
        rb"drone-dji:" + tag.encode("ascii") + rb'="([^"]*)"'
    )
    for key, tag in TAG_NAMES.items()
}


def read_xmp_head(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(XMP_SCAN_BYTES)


def read_metadata(head: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, pattern in TAG_PATTERNS.items():
        match = pattern.search(head)
        result[key] = (
            match.group(1).decode("utf-8", "replace") if match else ""
        )
    return result


def parse_calibration(raw: str, count: int) -> dict | None:
    if not raw or ";" not in raw:
        return None
    date, payload = raw.split(";", 1)
    try:
        values = [float(item.strip()) for item in payload.split(",")]
    except ValueError:
        return None
    if len(values) != count or not all(math.isfinite(value) for value in values):
        return None
    return {"date": date, "values": values, "raw": raw}


def jpeg_size(path: Path) -> tuple[int, int] | None:
    """Read JPEG dimensions without decoding the complete image."""
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            return None
        while True:
            byte = handle.read(1)
            if not byte:
                return None
            if byte != b"\xff":
                continue
            while byte == b"\xff":
                byte = handle.read(1)
            if not byte:
                return None
            marker = byte[0]
            if marker in (0xD8, 0xD9):
                continue
            if marker == 0xDA:
                return None
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                return None
            length = int.from_bytes(length_bytes, "big")
            is_sof = (
                0xC0 <= marker <= 0xC3
                or 0xC5 <= marker <= 0xC7
                or 0xC9 <= marker <= 0xCB
                or 0xCD <= marker <= 0xCF
            )
            if is_sof:
                handle.read(1)  # sample precision
                height = int.from_bytes(handle.read(2), "big")
                width = int.from_bytes(handle.read(2), "big")
                return width, height
            handle.seek(length - 2, os.SEEK_CUR)


def read_image(path: Path) -> np.ndarray | None:
    """Unicode-safe JPEG read on Windows."""
    encoded = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def camera_signature(metadata: dict[str, str], size: tuple[int, int] | None) -> tuple:
    return (
        metadata.get("camera_serial", ""),
        metadata.get("drone_serial", ""),
        metadata.get("product_name", ""),
        size,
    )


def make_undistort_map(
    calibration: dict, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, list[list[float]], list[float]]:
    values = calibration["values"]
    fx, fy, cx_dji, cy_dji = values[:4]
    camera_matrix = np.array(
        [
            [fx, 0.0, width / 2.0 + cx_dji],
            [0.0, fy, height / 2.0 + cy_dji],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(values[4:], dtype=np.float64)
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    return map_x, map_y, camera_matrix.tolist(), distortion.tolist()


def clean_record(record: dict) -> dict:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch dewarp DJI Matrice 4E JPGs using XMP DewarpData."
    )
    parser.add_argument("--src", required=True, help="Source JPG root")
    parser.add_argument("--dst", required=True, help="Separate output root")
    parser.add_argument(
        "--reuse-reference",
        help="Original JPG with valid DewarpData/K6; enables same-camera fallback",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quality", type=int, default=98)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output JPGs"
    )
    args = parser.parse_args()
    if args.workers < 1 or not 1 <= args.quality <= 100:
        parser.error("--workers must be >= 1 and --quality must be 1..100")

    source_root = Path(args.src).resolve()
    output_root = Path(args.dst).resolve()
    if not source_root.is_dir():
        parser.error(f"Source directory does not exist: {source_root}")
    if source_root == output_root:
        parser.error("--dst must be separate from --src")
    output_root.mkdir(parents=True, exist_ok=True)

    source_files = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg"}
        and output_root not in path.parents
    )
    if args.max_files > 0:
        source_files = source_files[: args.max_files]

    reference_path: Path | None = None
    reference_calibration: dict | None = None
    reference_signature: tuple | None = None
    reference_method = ""
    if args.reuse_reference:
        reference_path = Path(args.reuse_reference).resolve()
        if not reference_path.is_file():
            parser.error(f"Reference JPG does not exist: {reference_path}")
        reference_metadata = read_metadata(read_xmp_head(reference_path))
        reference_calibration = parse_calibration(
            reference_metadata["dewarp_data_k6"], 12
        )
        reference_method = "opencv_rational_k6_reused_same_camera"
        if reference_calibration is None:
            reference_calibration = parse_calibration(
                reference_metadata["dewarp_data"], 9
            )
            reference_method = "opencv_standard_k5_reused_same_camera"
        reference_size = jpeg_size(reference_path)
        reference_signature = camera_signature(reference_metadata, reference_size)
        if reference_calibration is None or reference_size is None:
            parser.error("Reference JPG has no valid DewarpData/K6 or dimensions")
        if not all(reference_signature[:3]):
            parser.error("Reference JPG lacks camera/product identity fields")

    records: list[dict] = []
    for path in source_files:
        metadata = read_metadata(read_xmp_head(path))
        size = jpeg_size(path)
        relative = path.relative_to(source_root).as_posix()
        record = {
            "source": relative,
            "output": "",
            "status": "",
            "reason": "",
            "width": size[0] if size else None,
            "height": size[1] if size else None,
            "product_name": metadata["product_name"],
            "drone_model": metadata["drone_model"],
            "camera_serial": metadata["camera_serial"],
            "drone_serial": metadata["drone_serial"],
            "dewarp_flag": metadata["dewarp_flag"],
            "dewarp_data": metadata["dewarp_data"],
            "dewarp_data_k6": metadata["dewarp_data_k6"],
            "method": "",
            "parameter_source": "",
            "error": "",
            "_path": path,
            "_calibration": None,
        }
        if size is None:
            record["status"] = "skipped_invalid_jpeg_dimensions"
            record["reason"] = "JPEG dimensions could not be read"
        elif metadata["dewarp_flag"] == "1":
            record["status"] = "skipped_already_dewarped_flag_1"
            record["reason"] = "DewarpFlag=1"
        else:
            calibration = parse_calibration(metadata["dewarp_data_k6"], 12)
            method = "opencv_rational_k6"
            if calibration is None:
                calibration = parse_calibration(metadata["dewarp_data"], 9)
                method = "opencv_standard_k5"
            if calibration is not None:
                record["_calibration"] = calibration
                record["method"] = method
                record["parameter_source"] = "per_image_xmp"
            elif (
                reference_calibration is not None
                and reference_signature is not None
                and camera_signature(metadata, size) == reference_signature
            ):
                record["_calibration"] = reference_calibration
                record["method"] = reference_method
                record["parameter_source"] = str(reference_path)
                record["reason"] = (
                    "Per-image DewarpData missing; reused calibration only after "
                    "camera, aircraft, product, and dimensions matched reference"
                )
            else:
                record["status"] = "skipped_missing_or_unsafe_dewarp_data"
                record["reason"] = (
                    "No valid per-image calibration and no matching same-camera "
                    "reference was supplied"
                )
        if record["_calibration"] is not None:
            record["output"] = relative
        records.append(record)

    map_cache: dict[tuple, dict] = {}
    for record in records:
        if record["status"] or record["_calibration"] is None:
            continue
        key = (
            record["method"],
            record["width"],
            record["height"],
            tuple(record["_calibration"]["values"]),
        )
        if key not in map_cache:
            map_x, map_y, camera_matrix, distortion = make_undistort_map(
                record["_calibration"], record["width"], record["height"]
            )
            map_cache[key] = {
                "map_x": map_x,
                "map_y": map_y,
                "camera_matrix": camera_matrix,
                "distortion": distortion,
            }
        record["_map_key"] = key

    eligible = [record for record in records if not record["status"]]
    started = time.perf_counter()

    def process(record: dict) -> dict:
        result = clean_record(record)
        output_path = output_root / Path(record["output"])
        try:
            if output_path.exists() and not args.overwrite:
                result["status"] = "skipped_existing_output"
                result["reason"] = "Output exists; use --overwrite to replace it"
                return result
            image = read_image(record["_path"])
            if image is None:
                result["status"] = "error_decode"
                result["reason"] = "cv2.imdecode returned None"
                return result
            height, width = image.shape[:2]
            if (width, height) != (record["width"], record["height"]):
                result["status"] = "error_dimension_changed"
                result["reason"] = (
                    f"header={(record['width'], record['height'])}, "
                    f"decoded={(width, height)}"
                )
                return result
            maps = map_cache[record["_map_key"]]
            corrected = cv2.remap(
                image,
                maps["map_x"],
                maps["map_y"],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            ok, encoded = cv2.imencode(
                ".jpg", corrected, [cv2.IMWRITE_JPEG_QUALITY, args.quality]
            )
            if not ok:
                result["status"] = "error_encode"
                result["reason"] = "cv2.imencode returned false"
                return result
            output_path.parent.mkdir(parents=True, exist_ok=True)
            encoded.tofile(str(output_path))
            if not output_path.exists() or output_path.stat().st_size == 0:
                result["status"] = "error_write"
                result["reason"] = "Output file is missing or empty"
                return result
            result["status"] = (
                "processed_reused_camera_calibration"
                if "reused_same_camera" in record["method"]
                else "processed"
            )
            result["output_bytes"] = output_path.stat().st_size
            result["reason"] = result["reason"] or (
                "cx/cy converted from image-center origin; output keeps same "
                "camera matrix and pixel dimensions"
            )
            return result
        except Exception as exc:  # per-file failure is recorded, not hidden
            result["status"] = "error_exception"
            result["error"] = repr(exc)
            return result

    finished: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, record) for record in eligible]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            finished[result["source"]] = result
            if index % 100 == 0 or index == len(futures):
                print(f"processed {index}/{len(futures)}", flush=True)

    final_records = []
    for record in records:
        if record["status"]:
            final_records.append(clean_record(record))
        else:
            final_records.append(finished[record["source"]])

    status_counts = Counter(record["status"] for record in final_records)
    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "algorithm": "OpenCV initUndistortRectifyMap + remap",
        "preferred_metadata": "DewarpDataK6",
        "fallback_metadata": "DewarpData",
        "cx_cy_policy": "cx_cv=W/2+cx_DJI; cy_cv=H/2+cy_DJI",
        "k6_order": ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"],
        "jpeg_quality": args.quality,
        "source_xmp_retained": False,
        "source_count": len(source_files),
        "parameter_group_count": len(map_cache),
        "status_counts": dict(status_counts),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (output_root / "dewarp_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "dewarp_manifest.json").write_text(
        json.dumps({"summary": summary, "records": final_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if not any(key.startswith("error") for key in status_counts) else 2


if __name__ == "__main__":
    raise SystemExit(main())
