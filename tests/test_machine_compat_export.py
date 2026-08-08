import math

import numpy as np
import pytest


def _point(lon, lat, segment_index, segment_type, x, y):
    return {
        "lon": float(lon),
        "lat": float(lat),
        "pixel_x": float(x),
        "pixel_y": float(y),
        "segment_index": int(segment_index),
        "point_index": 0,
        "segment_type": str(segment_type),
        "status": 1 if segment_type == "work" else 0,
    }


def _route_with_one_curved_turn():
    return [
        _point(120.0000, 30.0000, 0, "work", 0, 0),
        _point(120.0010, 30.0000, 0, "work", 10, 0),
        _point(120.0011, 30.0001, 1, "turn", 11, 1),
        _point(120.0015, 30.0005, 1, "turn", 12, 2),
        _point(120.0020, 30.0010, 1, "turn", 20, 3),
        _point(120.0030, 30.0010, 2, "work", 30, 3),
        _point(120.0040, 30.0010, 2, "work", 40, 3),
    ]


def test_machine_compatibility_collapses_a_curved_turn_to_one_chord():
    from path_planner import prepare_machine_compatibility_points

    points, diagnostics = prepare_machine_compatibility_points(
        _route_with_one_curved_turn()
    )

    assert [(round(p["pixel_x"]), round(p["pixel_y"])) for p in points] == [
        (0, 0),
        (10, 0),
        (30, 3),
        (40, 3),
    ]
    assert [p["status"] for p in points] == [1, 0, 1, 0]
    assert diagnostics["turn_geometry"] == "straight_chord"
    assert diagnostics["turn_run_count"] == 1
    assert diagnostics["removed_near_duplicate_count"] == 0


def test_machine_compatibility_compresses_each_dense_work_line_to_two_points():
    from path_planner import prepare_machine_compatibility_points

    route = [
        _point(120.0000, 30.0000, 0, "work", 0, 0),
        _point(120.0005, 30.0001, 0, "work", 5, 2),
        _point(120.0010, 30.0000, 0, "work", 10, 0),
        _point(120.0012, 30.0002, 1, "turn", 12, 2),
        _point(120.0020, 30.0010, 2, "work", 20, 10),
        _point(120.0025, 30.0007, 2, "work", 25, 7),
        _point(120.0030, 30.0010, 2, "work", 30, 10),
    ]

    points, diagnostics = prepare_machine_compatibility_points(route)

    assert [
        (round(point["pixel_x"]), round(point["pixel_y"]))
        for point in points
    ] == [(0, 0), (10, 0), (20, 10), (30, 10)]
    assert [point["status"] for point in points] == [1, 0, 1, 0]
    assert diagnostics["work_line_point_compression"] is True


def test_machine_compatibility_turn_chord_ignores_multiple_non_work_segments():
    from path_planner import prepare_machine_compatibility_points

    route = [
        _point(120.0000, 30.0000, 0, "work", 0, 0),
        _point(120.0010, 30.0000, 0, "work", 10, 0),
        _point(120.0012, 30.0002, 1, "turn_approach", 12, 2),
        _point(120.0015, 30.0005, 2, "turn", 15, 5),
        _point(120.0018, 30.0008, 3, "turn_aux", 18, 8),
        _point(120.0030, 30.0010, 4, "work", 30, 10),
        _point(120.0040, 30.0010, 4, "work", 40, 10),
    ]

    points, diagnostics = prepare_machine_compatibility_points(route)

    assert [(round(p["pixel_x"]), round(p["pixel_y"])) for p in points] == [
        (0, 0),
        (10, 0),
        (30, 10),
        (40, 10),
    ]
    assert [p["status"] for p in points] == [1, 0, 1, 0]
    assert diagnostics["source_segment_count"] == 5
    assert diagnostics["turn_run_count"] == 1


def test_machine_compatibility_removes_sub_two_centimetre_points(tmp_path):
    from path_planner import export_machine_compatibility_path

    route = [
        _point(120.0000, 30.0000, 0, "work", 0, 0),
        _point(120.0000, 30.00000003, 0, "work", 1, 0),
        _point(120.0010, 30.0000, 0, "work", 10, 0),
    ]
    output = tmp_path / "machine.path"

    points, diagnostics = export_machine_compatibility_path(route, str(output))

    # A logical work line is now projected directly to its two endpoints, so
    # the dense middle pose never reaches the protocol serializer.
    assert diagnostics["removed_near_duplicate_count"] == 0
    assert diagnostics["work_line_geometry"] == "straight_endpoint_chord"
    assert len(points) == 2
    assert all(points[index]["status"] in (0, 1) for index in range(len(points)))
    for first, second in zip(points, points[1:]):
        assert math.isfinite(float(first["lat"]))
        assert math.isfinite(float(second["lat"]))

    rows = [
        line
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.startswith("$PATH")
    ]
    assert len(rows) == 2
    for index, row in enumerate(rows):
        fields = row.rstrip("*").split(",")
        assert fields[1] == "0"
        assert fields[2] == str(index)
        assert fields[3] == "2"
        assert fields[-1] in {"0", "1"}


def test_machine_compatibility_treats_reverse_tag_as_one_straight_turn():
    from path_planner import prepare_machine_compatibility_points

    route = [
        _point(120.0000, 30.0000, 0, "work", 0, 0),
        _point(120.0010, 30.0000, 0, "work", 10, 0),
        _point(120.0020, 30.0010, 1, "turn_reverse", 20, 10),
        _point(120.0025, 30.0015, 1, "turn_reverse", 25, 15),
        _point(120.0030, 30.0010, 2, "work", 30, 10),
        _point(120.0040, 30.0010, 2, "work", 40, 10),
    ]

    points, diagnostics = prepare_machine_compatibility_points(route)

    assert [(round(p["pixel_x"]), round(p["pixel_y"])) for p in points] == [
        (0, 0),
        (10, 0),
        (30, 10),
        (40, 10),
    ]
    assert [p["status"] for p in points] == [1, 0, 1, 0]
    assert diagnostics["turn_geometry"] == "straight_chord"
    assert diagnostics["turn_run_count"] == 1


def test_machine_compatibility_requires_a_work_segment():
    from path_planner import prepare_machine_compatibility_points

    route = [
        _point(120.0000, 30.0000, 0, "turn", 0, 0),
        _point(120.0010, 30.0000, 0, "turn", 10, 0),
    ]

    with pytest.raises(ValueError, match="work"):
        prepare_machine_compatibility_points(route)


def test_machine_compatibility_rejects_a_chord_outside_a_concave_field(tmp_path):
    from path_planner import export_machine_compatibility_path

    route = [
        _point(120.0000, 30.0080, 0, "work", 1, 8),
        _point(120.0001, 30.0080, 0, "work", 2, 8),
        _point(120.0008, 30.0020, 1, "turn", 8, 2),
        _point(120.0009, 30.0020, 2, "work", 9, 2),
        _point(120.0010, 30.0020, 2, "work", 10, 2),
    ]
    # L-shaped field: both work-line ends are inside, but their direct chord
    # crosses the missing upper-right corner.
    field = np.asarray(
        [(0, 0), (10, 0), (10, 3), (3, 3), (3, 10), (0, 10)],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="越出田块边界"):
        export_machine_compatibility_path(
            route,
            str(tmp_path / "machine.path"),
            field_polygon=field,
        )


def test_machine_compatibility_splits_after_fifty_points_and_continues_numbering(
    tmp_path,
):
    from path_planner import export_machine_compatibility_path

    route = []
    for index in range(26):
        y = float(index * 2)
        route.extend([
            _point(120.0000 + index * 0.0001, 30.0000 + y * 0.00001,
                   index * 2, "work", 0, y),
            _point(120.0010 + index * 0.0001, 30.0000 + y * 0.00001,
                   index * 2, "work", 100, y),
        ])
        if index < 25:
            route.append(
                _point(120.0010 + index * 0.0001, 30.0001 + y * 0.00001,
                       index * 2 + 1, "turn", 100, y + 1)
            )

    (tmp_path / "001.txt").write_text("occupied\n", encoding="utf-8")
    points, diagnostics = export_machine_compatibility_path(
        route,
        str(tmp_path / "requested.path"),
    )

    assert len(points) == 52
    assert diagnostics["file_count"] == 2
    assert diagnostics["max_points_per_file"] == 50
    assert diagnostics["points_per_file"] == [50, 2]
    output_files = [tmp_path / "002.txt", tmp_path / "003.txt"]
    assert diagnostics["output_files"] == [str(path) for path in output_files]
    for path, expected_total in zip(output_files, (50, 2)):
        rows = [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("$PATH")
        ]
        assert len(rows) == expected_total
        assert rows[0].split(",")[2] == "0"
        assert rows[-1].rstrip("*").split(",")[3] == str(expected_total)
        assert rows[-1].rstrip("*").split(",")[2] == str(expected_total - 1)
        assert all(row.rstrip("*").split(",")[-1] in {"0", "1"} for row in rows)
        assert rows[-1].rstrip("*").split(",")[-1] == "0"


def test_machine_compatibility_directory_export_starts_at_001(tmp_path):
    from path_planner import export_machine_compatibility_path

    route = _route_with_one_curved_turn()
    output_dir = tmp_path / "machine_files"
    _points, diagnostics = export_machine_compatibility_path(
        route,
        str(output_dir),
    )

    assert diagnostics["file_count"] == 1
    assert diagnostics["output_files"] == [str(output_dir / "001.txt")]
    assert (output_dir / "001.txt").exists()
