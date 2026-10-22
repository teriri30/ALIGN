"""生成 packaging/align.ico（安装包与 EXE 共用图标）。

图形语义：深色底 + 三条绿色作业行带 + 一条白色对行路径，16 px 下也能认出是
“田里有行、机器沿行走”。改了配色或构图后重新运行：

    python packaging/make_icon.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 1024
BG = (14, 23, 38, 255)
ROW = (74, 222, 128, 255)
PATH = (246, 248, 251, 255)
ACCENT = (255, 226, 77, 255)


def build() -> Image.Image:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    radius = int(SIZE * 0.22)
    draw.rounded_rectangle((0, 0, SIZE - 1, SIZE - 1), radius=radius, fill=BG)

    # 作业行带：上窄下宽，避免看起来像“三条横杠”
    rows = [
        (0.30, 0.14, 0.92, 0.055),
        (0.52, 0.10, 0.86, 0.075),
        (0.75, 0.16, 0.80, 0.095),
    ]
    for center, left, right, width in rows:
        y = SIZE * center
        thickness = SIZE * width
        draw.rounded_rectangle(
            (SIZE * left, y - thickness / 2, SIZE * right, y + thickness / 2),
            radius=thickness / 2,
            fill=ROW,
        )

    # 对行路径：从右上绕到左下，末端一个黄色节点
    path_width = int(SIZE * 0.055)
    points = [
        (SIZE * 0.80, SIZE * 0.16),
        (SIZE * 0.52, SIZE * 0.16),
        (SIZE * 0.30, SIZE * 0.44),
        (SIZE * 0.30, SIZE * 0.86),
    ]
    draw.line(points, fill=PATH, width=path_width, joint="curve")
    draw.ellipse(
        (
            SIZE * 0.30 - path_width * 0.95,
            SIZE * 0.86 - path_width * 0.95,
            SIZE * 0.30 + path_width * 0.95,
            SIZE * 0.86 + path_width * 0.95,
        ),
        fill=ACCENT,
    )
    return image


def main() -> int:
    target = Path(__file__).with_name("align.ico")
    image = build()
    image.save(
        target,
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
