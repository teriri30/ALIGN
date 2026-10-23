"""缓存模型路径重绑：reUI 改名成 ALIGN 后，旧缓存里的绝对路径会失效。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _write(path, data=b"weights"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_rebinds_by_filename_and_hash(monkeypatch, tmp_path):
    from provenance import file_sha256
    from pyside6_app import workers

    model_dir = tmp_path / "models"
    old = _write(tmp_path / "old" / "yolo11m_seg_best.pt")
    moved = _write(model_dir / "yolo11m_seg_best.pt", b"weights")
    import runtime_paths

    monkeypatch.setattr(runtime_paths, "builtin_model_dir", lambda: str(model_dir))

    assert workers._rebind_model_path(str(old), file_sha256(str(old))) == str(moved)


def test_rejects_candidate_with_different_hash(monkeypatch, tmp_path):
    from provenance import file_sha256
    from pyside6_app import workers

    model_dir = tmp_path / "models"
    _write(model_dir / "yolo11m_seg_best.pt", b"other-weights")
    import runtime_paths

    monkeypatch.setattr(runtime_paths, "builtin_model_dir", lambda: str(model_dir))

    assert workers._rebind_model_path("E:/old/yolo11m_seg_best.pt", "0" * 64) == ""


def test_returns_empty_without_name_or_hash(monkeypatch, tmp_path):
    import runtime_paths
    from pyside6_app import workers

    monkeypatch.setattr(runtime_paths, "builtin_model_dir", lambda: str(tmp_path))
    assert workers._rebind_model_path("", "a" * 64) == ""
    assert workers._rebind_model_path("E:/old/x.pt", "") == ""
