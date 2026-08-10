"""地理坐标工具"""
import math
import logging
from typing import Optional, Tuple

from pyproj import CRS, Geod


class GeoUtils:
    def __init__(self):
        self._geod = Geod(ellps="WGS84")
        self._transformer = None
        self._dataset = None
        self._affine = None
        self._source_crs: Optional[CRS] = None
        self._warned_no_geo = False  # 避免重复打印警告

    def clear(self):
        self._dataset = None
        self._affine = None
        self._transformer = None
        self._source_crs = None

    @staticmethod
    def _parse_crs(value):
        if value is None:
            return None
        try:
            return CRS.from_user_input(value)
        except Exception:
            return None

    def set_dataset(self, dataset, transformer, source_crs=None):
        self._dataset = dataset
        self._affine = None
        self._transformer = transformer
        crs = source_crs if source_crs is not None else getattr(dataset, "crs", None)
        self._source_crs = self._parse_crs(crs)

    def set_affine(self, affine, transformer, source_crs=None):
        self._dataset = None
        self._affine = affine
        self._transformer = transformer
        self._source_crs = self._parse_crs(source_crs)

    def is_ready(self) -> bool:
        return (self._dataset is not None or self._affine is not None) and self._transformer is not None

    def _xy(self, row: float, col: float) -> Tuple[float, float]:
        if self._dataset is not None:
            return self._dataset.xy(float(row), float(col))
        if self._affine is not None:
            return self._affine * (float(col) + 0.5, float(row) + 0.5)
        raise RuntimeError("georeference unavailable")

    def _to_lonlat(self, x: float, y: float) -> Tuple[float, float]:
        try:
            lon, lat = self._transformer.transform(x, y, errcheck=True)
        except TypeError:
            # Lightweight test doubles and older transformer implementations.
            lon, lat = self._transformer.transform(x, y)
        lon, lat = float(lon), float(lat)
        if not math.isfinite(lon) or not math.isfinite(lat):
            raise ValueError("coordinate transform returned NaN or infinity")
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise ValueError(f"coordinate transform is not WGS84 lon/lat: {lon}, {lat}")
        return lon, lat

    def pixel_to_lonlat(self, px: float, py: float) -> Tuple[float, float]:
        if not self.is_ready():
            raise RuntimeError("georeference unavailable")
        gx, gy = self._xy(float(py), float(px))
        return self._to_lonlat(gx, gy)

    def _projected_unit_factors(self):
        if self._source_crs is None or not self._source_crs.is_projected:
            return None
        axes = self._source_crs.axis_info
        if len(axes) < 2:
            return None
        fx = float(axes[0].unit_conversion_factor or 0.0)
        fy = float(axes[1].unit_conversion_factor or 0.0)
        if fx <= 0.0 or fy <= 0.0:
            return None
        return fx, fy

    def pixel_distance_m(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """计算两个像素坐标之间的地理距离（米）。
        GeoTIFF 多数使用投影坐标系（如 EPSG:32650，单位为米），应优先
        直接用 dataset.xy() 得到的投影坐标计算欧氏距离；只有投影距离不可用
        时才退回到经纬度大地线距离，避免把米坐标误当经纬度导致距离放大。
        """
        if not self.is_ready():
            if not self._warned_no_geo:
                logging.warning("[GeoUtils] 地理数据不可用，pixel_distance_m 回退到欧氏像素距离")
                self._warned_no_geo = True
            return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        gx1, gy1 = self._xy(float(p1[1]), float(p1[0]))
        gx2, gy2 = self._xy(float(p2[1]), float(p2[0]))
        if self._source_crs is None:
            # Backward compatibility for callers that provide a metre affine
            # and a lightweight transformer but no CRS metadata.
            dist = math.hypot(gx2 - gx1, gy2 - gy1)
        else:
            lon1, lat1 = self._to_lonlat(gx1, gy1)
            lon2, lat2 = self._to_lonlat(gx2, gy2)
            _, _, dist = self._geod.inv(lon1, lat1, lon2, lat2)
        if not math.isfinite(dist) or dist < 0.0:
            raise ValueError("invalid geographic distance")
        return float(dist)

    def pixels_per_meter(self, px: float, py: float) -> float:
        """计算指定位置的 pixels-per-meter（每米对应多少像素）。
        综合 x 和 y 方向的像素间距，取平均值作为局部比例尺。
        """
        if not self.is_ready():
            return 1.0
        dx = self.pixel_distance_m((px, py), (px + 1.0, py))
        dy = self.pixel_distance_m((px, py), (px, py + 1.0))
        avg_mpp = (dx + dy) / 2.0 if (dx > 0 and dy > 0) else max(dx, dy)
        if not math.isfinite(avg_mpp) or avg_mpp <= 0.0:
            raise ValueError("invalid raster ground sampling distance")
        return 1.0 / avg_mpp

    def meters_per_pixel(self, px: float, py: float) -> float:
        """计算指定位置的 meters-per-pixel（每像素对应多少米）。"""
        ppm = self.pixels_per_meter(px, py)
        return 1.0 / ppm if ppm > 0 else 1.0

    def validate_raster_extent(self, width: int, height: int):
        """Validate transform output at raster corners and return its WGS84 extent."""
        width, height = int(width), int(height)
        if width <= 0 or height <= 0:
            raise ValueError("invalid raster dimensions")
        samples = [
            (0.0, 0.0),
            (float(width - 1), 0.0),
            (float(width - 1), float(height - 1)),
            (0.0, float(height - 1)),
            ((width - 1) / 2.0, (height - 1) / 2.0),
        ]
        lonlat = [self.pixel_to_lonlat(x, y) for x, y in samples]
        lons = [point[0] for point in lonlat]
        lats = [point[1] for point in lonlat]
        if width > 1 and self.pixel_distance_m(samples[0], samples[1]) <= 0.0:
            raise ValueError("raster horizontal ground span is zero")
        if height > 1 and self.pixel_distance_m(samples[0], samples[3]) <= 0.0:
            raise ValueError("raster vertical ground span is zero")
        return {
            "min_lon": min(lons),
            "max_lon": max(lons),
            "min_lat": min(lats),
            "max_lat": max(lats),
            "source_crs": self._source_crs.to_string() if self._source_crs else "",
            "target_crs": "EPSG:4326",
        }

    def pixel_polygon_area_m2(self, points) -> float:
        """Area of a pixel-coordinate polygon in square metres.

        The app may display a downsampled preview.  When set_affine() is given
        the display transform, these display pixel coordinates still map to the
        correct projected metre coordinates.  Shoelace on projected coordinates
        is more reliable than using a single local pixels-per-metre value.
        """
        if not self.is_ready() or not points or len(points) < 3:
            return 0.0
        projected = []
        for px, py in points:
            gx, gy = self._xy(float(py), float(px))
            projected.append((float(gx), float(gy)))
        factors = self._projected_unit_factors()
        if self._source_crs is not None:
            lonlat = [self._to_lonlat(x, y) for x, y in projected]
            lons, lats = zip(*lonlat)
            area, _ = self._geod.polygon_area_perimeter(lons, lats)
            return abs(float(area))
        fx, fy = factors if factors is not None else (1.0, 1.0)
        area = 0.0
        for i, (x1, y1) in enumerate(projected):
            x2, y2 = projected[(i + 1) % len(projected)]
            area += x1 * y2 - x2 * y1
        return abs(area) * 0.5 * fx * fy
