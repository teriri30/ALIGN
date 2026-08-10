"""Runtime environment normalization that only depends on the standard library."""
from __future__ import annotations

import os
import sys
from runtime_paths import _bundle_candidates


def configure_geospatial_environment() -> dict:
    prefix = os.path.abspath(sys.prefix)
    bundle_roots = _bundle_candidates()
    candidates = {
        "GDAL_DATA": [
            os.path.join(prefix, "Library", "share", "gdal"),
            os.path.join(prefix, "share", "gdal"),
            *[str(root / "gdal_data") for root in bundle_roots],
            *[str(root / "data" / "gdal_data") for root in bundle_roots],
        ],
        "PROJ_DATA": [
            os.path.join(prefix, "Library", "share", "proj"),
            os.path.join(prefix, "share", "proj"),
            os.path.join(prefix, "Lib", "site-packages", "rasterio", "proj_data"),
            os.path.join(prefix, "Lib", "site-packages", "pyproj", "proj_dir", "share", "proj"),
            *[str(root / "proj_data") for root in bundle_roots],
            *[str(root / "rasterio" / "proj_data") for root in bundle_roots],
            *[str(root / "pyproj" / "proj_dir" / "share" / "proj") for root in bundle_roots],
        ],
    }
    configured = {}
    for variable, paths in candidates.items():
        existing = os.environ.get(variable, "")
        if existing and os.path.isdir(existing):
            configured[variable] = existing
            continue
        for path in paths:
            if os.path.isdir(path):
                os.environ[variable] = path
                configured[variable] = path
                break
    return configured
