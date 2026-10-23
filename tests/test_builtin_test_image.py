"""随包发布的内置测试影像：必须存在、够小、带地理参考。"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "assets" / "builtin" / "field_sample.tif"


def test_sample_exists_and_is_small_enough_to_ship():
    assert SAMPLE.is_file(), "内置测试影像样例会随安装包发布，不能缺"
    size_mb = SAMPLE.stat().st_size / 1e6
    assert size_mb < 20, f"样例 {size_mb:.1f} MB 太大，会拖大安装包"


def test_sample_keeps_georeference():
    rasterio = pytest.importorskip("rasterio")
    with rasterio.open(SAMPLE) as src:
        assert src.width == src.height == 2000
        assert src.crs is not None and src.crs.to_epsg() == 32650
        assert src.res[0] < 0.02  # 比 2 cm/px 更细，行带不会糊成一片
        assert src.count >= 3


def test_builtin_test_image_resolves_to_existing_file():
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from builtin_test_assets import builtin_test_image

    path = builtin_test_image()
    assert path and Path(path).is_file()


def test_packaging_ships_the_sample():
    spec = (ROOT / "packaging" / "align.spec").read_text(encoding="utf-8")
    assert "assets" in spec and "builtin" in spec, "spec 必须把样例打进 data/builtin"
