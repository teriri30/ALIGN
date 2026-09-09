import os

from config import AppLogger


def test_cleanup_old_logs_matches_current_and_rotated_log_names(tmp_path):
    names = [
        "app_log_20260901_000000.log",
        "app_log_20260901_000001.log.1",
        "app_log_20260901_000002.log.2",
        "app_log_20260901_000003.log",
        "app_log_20260901_000004.log.1",
        "app_log_20260901_000005.log",
        "app_log_20260901_000006.log.3",
    ]
    for index, name in enumerate(names):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        os.utime(path, (1_000 + index, 1_000 + index))

    AppLogger._cleanup_old_logs(str(tmp_path), max_keep=5)

    remaining = sorted(path.name for path in tmp_path.glob("app_log_*.log*"))
    assert remaining == sorted(names[-5:])
