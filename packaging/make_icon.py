"""生成 packaging/align.ico（安装包、EXE、窗口图标共用）。

设计取向：字母标 A + 极简几何，不用插画。
- 深色圆角方底（与主界面 bg_darkest 同色系），自上而下极轻的明度过渡；
- 细笔画的几何 A，圆头收笔，留白充足，不显笨重；
- 横杠换成薄荷绿 accent，作为唯一彩色焦点，呼应"对行 / 中心线"。
16 px 下仍要能认出 A，所以笔画下限约 7%。

用法：
    python packaging/make_icon.py            # 生成 align.ico
    python packaging/make_icon.py --preview  # 另存多尺寸预览图到 outputs/icon-preview
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

SIZE = 1024
SS = 4  # 超采样倍数：先画 4 倍再缩，边缘更干净

BG_TOP = (14, 22, 34, 255)
BG_BOTTOM = (8, 13, 21, 255)
INK = (248, 250, 252, 255)
ACCENT = (52, 211, 153, 255)
HAIRLINE = (255, 255, 255, 26)


def _rounded_mask(size: int, radius_ratio: float) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=int(size * radius_ratio), fill=255
    )
    return mask


def _background(size: int, light: bool = False) -> Image.Image:
    """深色圆角底 + 自上而下的轻微渐变（避免大面积死黑，显得有质感）。

    ``light=True`` 用于 ≤32 px：小尺寸下深底在深色任务栏上会"消失"，
    把底色抬亮一档，方块轮廓才立得住。
    """
    bottom_color = (22, 31, 45, 255) if light else BG_BOTTOM
    top_color = (36, 48, 68, 255) if light else BG_TOP
    bottom = Image.new("RGBA", (size, size), bottom_color)
    top = Image.new("RGBA", (size, size), top_color)
    column = Image.new("L", (1, size))
    for y in range(size):
        column.putpixel((0, y), int(255 * (1 - y / max(1, size - 1)) ** 1.15))
    gradient = Image.composite(top, bottom, column.resize((size, size)))
    gradient.putalpha(_rounded_mask(size, 0.225))
    return gradient


def _stroke(draw: ImageDraw.ImageDraw, points, width: int, color) -> None:
    draw.line(points, fill=color, width=width, joint="curve")
    r = width / 2
    for x, y in (points[0], points[-1]):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def _letter_a_geometry(s: int, wide: bool = False):
    """A 的三段几何：顶点、左右脚、横杠两端。"""
    # 光学居中：整体比几何中心略高约 1.2%，避免看起来往下坠
    spread = 0.026 if wide else 0.0  # 小尺寸把两脚再撑开一点，内三角更通透
    apex = (s * 0.500, s * 0.264)
    foot_l = (s * (0.302 - spread), s * 0.764)
    foot_r = (s * (0.698 + spread), s * 0.764)
    t = 0.578  # 横杠位置：按字形视觉中线再抬 2%，避免看起来偏低
    bar_y = apex[1] + (foot_l[1] - apex[1]) * t
    inset = s * 0.010

    def x_on_leg(a, b):
        return a[0] + (b[0] - a[0]) * t

    return (
        apex,
        foot_l,
        foot_r,
        (x_on_leg(apex, foot_l) - inset, bar_y),
        (x_on_leg(apex, foot_r) + inset, bar_y),
    )


def build(variant: str = "crossbar", size: int = SIZE) -> Image.Image:
    canvas = size * SS if size == SIZE else size
    base = _background(canvas)
    draw = ImageDraw.Draw(base)
    s = canvas
    apex, foot_l, foot_r, bar_l, bar_r = _letter_a_geometry(s)
    ratio = 0.066 if size > 48 else 0.082  # 16–48 px 加粗，避免糊成一团
    w = max(2, int(s * ratio))

    if variant == "refined":
        return _build_refined(size)
    if variant.startswith("solid"):
        return _build_solid(variant, size)
    if variant == "crossbar":
        _stroke(draw, [foot_l, apex], w, INK)
        _stroke(draw, [apex, foot_r], w, INK)
        _stroke(draw, [bar_l, bar_r], int(w * 0.86), ACCENT)
    elif variant == "guide":
        _stroke(draw, [foot_l, apex], w, INK)
        _stroke(draw, [apex, foot_r], w, INK)
        # 细一号的导引线，两端略微探出双腿，暗示"对行基准线"
        ext = s * 0.045
        _stroke(
            draw,
            [(bar_l[0] - ext, bar_l[1]), (bar_r[0] + ext, bar_l[1])],
            max(2, int(w * 0.52)),
            ACCENT,
        )
    elif variant == "exit":
        _stroke(draw, [foot_l, apex], w, INK)
        _stroke(draw, [apex, foot_r], w, INK)
        # 横杠只向右探出一段，像一条驶出田头的作业路径
        ext = s * 0.062
        _stroke(
            draw,
            [(bar_l[0], bar_l[1]), (bar_r[0] + ext, bar_l[1])],
            max(2, int(w * 0.62)),
            ACCENT,
        )
    elif variant == "dual":
        _stroke(draw, [foot_l, apex], w, INK)
        _stroke(draw, [apex, foot_r], w, ACCENT)
        _stroke(draw, [bar_l, bar_r], int(w * 0.86), INK)
    elif variant == "path":
        _stroke(draw, [foot_l, apex], w, INK)
        _stroke(draw, [apex, foot_r], w, INK)
        count, gap = 3, 0.10
        span = bar_r[0] - bar_l[0]
        seg = span / (count + (count - 1) * gap)
        x = bar_l[0]
        for _ in range(count):
            _stroke(draw, [(x, bar_l[1]), (x + seg, bar_l[1])], int(w * 0.8), ACCENT)
            x += seg * (1 + gap)
    else:
        raise SystemExit(f"unknown variant: {variant}")

    # 极细内描边：浅色任务栏上也有边界
    draw.rounded_rectangle(
        (1, 1, s - 2, s - 2),
        radius=int(s * 0.225),
        outline=HAIRLINE,
        width=max(1, s // 512),
    )

    if canvas != size:
        return base.resize((size, size), Image.LANCZOS)
    return base


def _vertical_gradient(size: int, top, bottom) -> Image.Image:
    layer = Image.new("RGBA", (size, size), bottom)
    column = Image.new("L", (1, size))
    for y in range(size):
        column.putpixel((0, y), int(255 * (1 - y / max(1, size - 1)) ** 1.05))
    return Image.composite(Image.new("RGBA", (size, size), top), layer, column.resize((size, size)))


def _radial_highlight(size: int, intensity: int = 46) -> Image.Image:
    """左上偏中的柔光，让深色卡片像被打了一束顶光。"""
    layer = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(layer)
    cx, cy, r = size * 0.42, size * 0.24, size * 0.62
    steps = 26
    for i in range(steps, 0, -1):
        k = i / steps
        alpha = int(intensity * (1 - k) ** 1.8)
        box = (cx - r * k, cy - r * k * 0.85, cx + r * k, cy + r * k * 0.85)
        draw.ellipse(box, fill=alpha)
    glow = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    glow.putalpha(layer)
    return glow


def _shaped_gradient(size: int, mask: Image.Image, top, bottom) -> Image.Image:
    shape = _vertical_gradient(size, top, bottom)
    shape.putalpha(mask)
    return shape


def _build_refined(size: int) -> Image.Image:
    """精致版，按尺寸分级做减法（optical size）。

    - ≥128：玻璃顶光、极淡行带纹理、渐变白 A、绿色端点
    - 48–64：去掉纹理与顶光，保留端点与细描边
    - 32–40：去掉端点，只留白 A + 绿线
    - ≤24：纯白 A、加粗、单色绿线，不做任何半透明装饰
    """
    canvas = size * 4  # 所有尺寸都 4 倍超采样后缩小，小图边缘才不糊
    s = canvas
    with_texture = False  # 行带纹理在评审中显得像噪点，去掉
    with_arc = False      # 顶部弧线像一条杂线，去掉，只留径向柔光
    with_node = size >= 48
    with_hairline = size >= 32
    with_gradient = size >= 64

    tile = _background(canvas, light=size <= 32)
    if with_arc:
        tile = Image.alpha_composite(tile, _radial_highlight(canvas, 34))
    draw = ImageDraw.Draw(tile)

    if with_texture:
        band_mask = Image.new("L", (s, s), 0)
        bd = ImageDraw.Draw(band_mask)
        for i in range(4):
            y = s * (0.315 + i * 0.132)
            bd.rounded_rectangle(
                (s * 0.150, y - s * 0.010, s * 0.850, y + s * 0.010),
                radius=s * 0.010,
                fill=255,
            )
        bands = Image.new("RGBA", (s, s), (120, 160, 210, 255))
        bands.putalpha(band_mask.point(lambda v: int(v * 0.042)))
        tile = Image.alpha_composite(tile, bands)
        draw = ImageDraw.Draw(tile)

    apex, foot_l, foot_r, bar_l, bar_r = _letter_a_geometry(s, wide=size <= 32)
    # 笔画占画布的比例：尺寸越小、相对越粗，保证 16 px 也能读出 A
    if size >= 64:
        ratio = 0.066
    elif size >= 48:
        ratio = 0.072
    elif size >= 32:
        ratio = 0.085
    elif size >= 24:
        ratio = 0.095
    else:
        ratio = 0.108
    w = max(2, int(s * ratio))

    letter = Image.new("L", (s, s), 0)
    ld = ImageDraw.Draw(letter)
    for pts in ([foot_l, apex], [apex, foot_r]):
        ld.line(pts, fill=255, width=w, joint="curve")
        for x, y in pts:
            ld.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=255)
    # 负空间：沿横杠位置从 A 上挖一条槽（宽度 = 绿线 + 2×间隙），
    # 绿线落进槽里，交界处不再是"贴纸压在字上"，而是有前后层次。
    ext = s * (0.052 if size >= 48 else 0.040)
    line_w0 = max(2, int(w * (0.46 if size >= 48 else 0.64)))
    gap = max(2, int(s * 0.012)) if size >= 48 else 0
    cut = Image.new("L", (s, s), 0)
    cd = ImageDraw.Draw(cut)
    if gap:
        cd.rounded_rectangle(
            (bar_l[0] - ext * 0.7 - gap / 2, bar_l[1] - (line_w0 + gap) / 2,
             bar_r[0] + ext + gap / 2, bar_l[1] + (line_w0 + gap) / 2),
            radius=(line_w0 + gap) / 2,
            fill=255,
        )
    letter = ImageChops.subtract(letter, cut)

    if with_gradient:
        top_ink, bottom_ink = (255, 255, 255, 255), (222, 229, 238, 255)
    else:
        top_ink = bottom_ink = (250, 252, 255, 255)
    tile = Image.alpha_composite(tile, _shaped_gradient(s, letter, top_ink, bottom_ink))
    draw = ImageDraw.Draw(tile)

    if size >= 48:
        edge = Image.new("L", (s, s), 0)
        ed = ImageDraw.Draw(edge)
        for pts in ([foot_l, apex], [apex, foot_r]):
            ed.line(pts, fill=64, width=max(1, w // 14), joint="curve")
        edge_ink = Image.new("RGBA", (s, s), (10, 16, 26, 255))
        edge_ink.putalpha(edge)
        tile = Image.alpha_composite(tile, edge_ink)
        draw = ImageDraw.Draw(tile)

    line_w = line_w0 if size > 32 else max(2, int(w * 0.80))
    bar_mask = Image.new("L", (s, s), 0)
    bmd = ImageDraw.Draw(bar_mask)
    bmd.rounded_rectangle(
        (bar_l[0] - ext * 0.7, bar_l[1] - line_w / 2, bar_r[0] + ext, bar_l[1] + line_w / 2),
        radius=line_w / 2,
        fill=255,
    )
    if size > 32:
        bar_color = _shaped_gradient(s, bar_mask, (74, 222, 160, 255), (44, 190, 138, 255))
    else:
        bar_color = Image.new("RGBA", (s, s), ACCENT)
        bar_color.putalpha(bar_mask)
    tile = Image.alpha_composite(tile, bar_color)
    draw = ImageDraw.Draw(tile)
    if with_node:
        node_r = max(2, int(s * 0.020))
        nx = bar_r[0] + ext
        draw.ellipse((nx - node_r, bar_l[1] - node_r, nx + node_r, bar_l[1] + node_r), fill=(110, 231, 183, 255))

    if with_hairline:
        draw.rounded_rectangle(
            (1, 1, s - 2, s - 2),
            radius=int(s * 0.225),
            outline=(255, 255, 255, 44 if size > 48 else (74 if size > 24 else 96)),
            width=max(1, s // 384),
        )
    if with_arc:
        draw.arc(
            (int(s * 0.03), int(s * 0.012), int(s * 0.97), int(s * 0.5)),
            start=200, end=340, fill=(255, 255, 255, 52), width=max(1, s // 320),
        )

    return tile.resize((size, size), Image.LANCZOS)

# --------------------------------------------------------------------------
# 实心字母标：参照成熟桌面图标的规律——品牌色块 + 实心符号 + 高对比，
# 不靠 1px 细线（细线在 16 px 会糊掉）。
# --------------------------------------------------------------------------

def _solid_legs(s: int, thick: float):
    """A 的两条实心腿（厚度按画布比例），返回左右腿多边形与横杠端点。"""
    apex_y = s * 0.205
    base_y = s * 0.795
    foot_l = s * 0.245
    foot_r = s * 0.755
    half = s * thick / 2
    apex = (s * 0.5, apex_y)

    def leg_points(foot_x, sign):
        # sign=-1 左腿，+1 右腿：沿腿方向偏移半个厚度
        dx = foot_x - apex[0]
        dy = base_y - apex[1]
        length = (dx * dx + dy * dy) ** 0.5
        nx, ny = dy / length, -dx / length  # 法向
        pts = [
            (apex[0] + nx * half * sign, apex[1] + ny * half * sign),
            (foot_x + nx * half * sign, base_y + ny * half * sign),
            (foot_x - nx * half * sign, base_y - ny * half * sign),
            (apex[0] - nx * half * sign, apex[1] - ny * half * sign),
        ]
        return pts

    bar_y = s * 0.628
    def x_on_leg(foot_x):
        k = (bar_y - apex[1]) / (base_y - apex[1])
        return apex[0] + (foot_x - apex[0]) * k

    return leg_points(foot_l, -1), leg_points(foot_r, 1), bar_y, x_on_leg(foot_l), x_on_leg(foot_r)


def _build_solid(variant: str, size: int) -> Image.Image:
    """参照成熟桌面图标：色块底 + 一个实心符号 + 高对比，不用细线。

    符号用"粗折线 + 圆头圆角"画成实心 A（不是平行四边形拼接，避免顶点缺口），
    横杠位置抠出背景色的槽，再把薄荷绿横杠放进去。
    """
    canvas = size * 4
    s = canvas
    light = size <= 32
    if variant == "solid_green":
        top_color, bottom_color = (26, 148, 114, 255), (9, 92, 71, 255)
    elif variant == "solid_invert":
        top_color, bottom_color = (86, 232, 178, 255), (36, 186, 137, 255)
    else:
        top_color = (30, 43, 62, 255) if light else BG_TOP
        bottom_color = (18, 27, 41, 255) if light else BG_BOTTOM

    tile = _vertical_gradient(s, top_color, bottom_color)
    tile.putalpha(_rounded_mask(s, 0.225))
    if variant == "solid_dark" and not light:
        tile = Image.alpha_composite(tile, _radial_highlight(s, 30))
    draw = ImageDraw.Draw(tile)

    apex = (s * 0.500, s * 0.212)
    foot_l = (s * 0.262, s * 0.788)
    foot_r = (s * 0.738, s * 0.788)
    thick = 0.180 if size >= 48 else 0.200
    w = int(s * thick)
    bar_y = s * 0.602
    bar_h = int(s * (0.096 if size >= 48 else (0.120 if size > 24 else 0.155)))

    if variant == "solid_invert":
        ink_top = ink_bottom = (8, 22, 16, 255)
    else:
        ink_top, ink_bottom = (255, 255, 255, 255), (228, 236, 245, 255)

    # 1) 实心 A
    letter = Image.new("L", (s, s), 0)
    ld = ImageDraw.Draw(letter)
    for pts in ([foot_l, apex], [apex, foot_r]):
        ld.line(pts, fill=255, width=w, joint="curve")
        for x, y in pts:
            ld.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=255)

    # 2) 抠出横杠槽（左右各留一点腿，形成负空间）
    def x_at(y):
        k = (y - apex[1]) / (foot_l[1] - apex[1])
        return apex[0] + (foot_l[0] - apex[0]) * k, apex[0] + (foot_r[0] - apex[0]) * k

    left_leg_x, right_leg_x = x_at(bar_y)
    gap = s * (0.016 if size >= 48 else (0.012 if size > 24 else 0.0))
    cut = Image.new("L", (s, s), 0)
    cd = ImageDraw.Draw(cut)
    cd.rounded_rectangle(
        (left_leg_x + gap, bar_y - bar_h / 2, right_leg_x - gap, bar_y + bar_h / 2),
        radius=bar_h / 2,
        fill=255,
    )
    letter = ImageChops.subtract(letter, cut)
    tile = Image.alpha_composite(tile, _shaped_gradient(s, letter, ink_top, ink_bottom))
    draw = ImageDraw.Draw(tile)

    # 3) 薄荷绿横杠落进槽里
    if variant == "solid_invert":
        bar_fill = (10, 22, 18, 255)
    else:
        bar_fill = ACCENT if size > 24 else (74, 222, 160, 255)  # 16/20/24 提亮，小图不发灰
    draw.rounded_rectangle(
        (left_leg_x + gap, bar_y - bar_h / 2, right_leg_x - gap, bar_y + bar_h / 2),
        radius=bar_h / 2,
        fill=bar_fill,
    )

    # 4) 内描边：浅色任务栏上不丢边界
    outline_alpha = 58 if (variant == "solid_dark" and light) else (30 if variant == "solid_invert" else 44)
    draw.rounded_rectangle(
        (1, 1, s - 2, s - 2),
        radius=int(s * 0.225),
        outline=(255, 255, 255, outline_alpha),
        width=max(1, s // 400),
    )
    return tile.resize((size, size), Image.LANCZOS)

def _preview(folder: Path) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    for variant in ("solid_dark", "solid_green", "solid_invert", "refined"):
        big = build(variant, 256)
        sheet = Image.new("RGBA", (840, 320), (26, 30, 38, 255))
        sheet.alpha_composite(big, (16, 32))
        x = 300
        for px in (128, 64, 48, 32, 24, 16):
            sheet.alpha_composite(build(variant, px), (x, 40))
            x += px + 14
        sheet.save(folder / f"icon_{variant}.png")
        print(f"preview: {folder / f'icon_{variant}.png'}")
    return 0


def main(argv: list[str]) -> int:
    if "--preview" in argv:
        target = Path(__file__).resolve().parent.parent / "outputs" / "icon-preview"
        return _preview(target)
    target = Path(__file__).with_name("align.ico")
    image = build("solid_dark")
    image.save(
        target,
        format="ICO",
        sizes=[
            (16, 16), (20, 20), (24, 24), (32, 32),
            (40, 40), (48, 48), (64, 64), (128, 128), (256, 256),
        ],
    )
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
