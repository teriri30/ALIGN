# PyInstaller onedir build for the Windows portable release.
from pathlib import Path
import sys

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


ROOT = Path(SPECPATH).resolve().parent
SOURCE_ROOT = ROOT / "src"


datas = [
    (str(ROOT / "config.example.json"), "."),
    (str(ROOT / "data" / "models"), "data/models"),
    (str(ROOT / "packaging" / "README-Windows.txt"), "."),
]
binaries = []
hiddenimports = [
    "rasterio._base",
    "rasterio._io",
    "rasterio.crs",
    "rasterio.enums",
    "rasterio.serde",
    "rasterio.transform",
    "pyproj.datadir",
    "torch._C",
    "torchvision._C",
]
hiddenimports.extend(collect_submodules("rasterio"))


def add_directory_if_present(path: Path, destination: str) -> None:
    if path.is_dir():
        datas.append((str(path), destination))


def filter_host_tool_binaries(entries):
    """Keep Codex/workspace helper runtimes out of the portable bundle."""
    cleaned = []
    replacement_dirs = (
        Path(sys.prefix),
        Path(sys.prefix) / "DLLs",
        Path(sys.prefix) / "Library" / "bin",
    )
    for destination, source, kind in entries:
        source_path = Path(source)
        source_parts = {part.casefold() for part in source_path.parts}
        if "codex-runtimes" not in source_parts:
            cleaned.append((destination, source, kind))
            continue

        replacement = next(
            (
                directory / source_path.name
                for directory in replacement_dirs
                if (directory / source_path.name).is_file()
            ),
            None,
        )
        if replacement is not None:
            print(
                "Replacing host-tool DLL with build-environment DLL: "
                f"{source_path} -> {replacement}"
            )
            cleaned.append((destination, str(replacement), kind))
        else:
            print(f"Excluding host-tool DLL from bundle: {source_path}")
    return cleaned

for package in (
    "PySide6", "numpy", "cv2", "PIL", "scipy", "shapely", "affine",
    "pyproj", "rasterio", "torch", "torchvision", "ultralytics",
):
    try:
        datas.extend(collect_data_files(package))
        binaries.extend(collect_dynamic_libs(package))
    except Exception:
        # PyInstaller's built-in hooks cover packages that do not expose
        # optional data or native libraries in this environment.
        pass

# Rasterio/pyproj wheels and conda packages place GDAL/PROJ definitions in
# different locations.  Copy whichever locations exist so the frozen app can
# configure both libraries without relying on a user's Conda installation.
prefix = Path(sys.prefix)
add_directory_if_present(prefix / "Library" / "share" / "gdal", "gdal_data")
add_directory_if_present(prefix / "share" / "gdal", "gdal_data")
add_directory_if_present(prefix / "Library" / "share" / "proj", "proj_data")
add_directory_if_present(prefix / "share" / "proj", "proj_data")

try:
    import pyproj

    add_directory_if_present(
        Path(pyproj.__file__).resolve().parent / "proj_dir" / "share" / "proj",
        "proj_data",
    )
except Exception:
    pass

build_info = ROOT / "build_info.json"
if build_info.is_file():
    datas.append((str(build_info), "."))

analysis = Analysis(
    [str(SOURCE_ROOT / "frozen_main.py")],
    pathex=[str(SOURCE_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(ROOT / "packaging" / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "pytest_cov", "ruff", "tkinter"],
    noarchive=False,
)
analysis.binaries = filter_host_tool_binaries(analysis.binaries)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="align",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=(
        str(ROOT / "packaging" / "align.ico")
        if (ROOT / "packaging" / "align.ico").exists()
        else None
    ),
)
coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    analysis.zipfiles,
    strip=False,
    upx=False,
    name="align",
    contents_directory="_internal",
)
