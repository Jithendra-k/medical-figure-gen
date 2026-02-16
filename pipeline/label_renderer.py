"""
Stage 4: Label Renderer
Renders text labels on the generated image using Pillow.
Supports 6 annotation styles. Also generates SVG with editable labels.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from config import FONT_PATH, FONT_BOLD_PATH, GENERATED_DIR

# Type alias for annotation styles
LabelStyle = Literal[
    "plain_text", "boxed_text", "numbered",
    "color_coded", "minimal", "textbook",
]

# Color palette for color_coded style
COLORS = [
    "#E74C3C",  # red
    "#3498DB",  # blue
    "#2ECC71",  # green
    "#F39C12",  # orange
    "#9B59B6",  # purple
    "#1ABC9C",  # teal
    "#E67E22",  # deep orange
    "#2980B9",  # dark blue
    "#27AE60",  # dark green
    "#8E44AD",  # dark purple
    "#D35400",  # rust
    "#16A085",  # dark teal
    "#C0392B",  # dark red
    "#2C3E50",  # navy
    "#7F8C8D",  # gray
]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font, falling back to default if not found."""
    path = FONT_BOLD_PATH if bold else FONT_PATH
    try:
        return ImageFont.truetype(str(path), size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("arial.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def _text_size(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> tuple[int, int]:
    """Get text width and height reliably across Pillow versions."""
    bbox = font.getbbox(text)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def compute_label_layout(
    points: list[dict],
    image_width: int,
    image_height: int,
    label_side: str = "right",
    font_size: int = 14,
) -> list[dict]:
    """
    Given anatomical point positions from the vision model,
    compute optimal label text positions on the specified side.

    Sorts by y-coordinate and distributes labels evenly, avoiding overlap.
    Supports "both" for bilateral labels (multi-view diagrams).

    Returns list of dicts with: label, point_x, point_y, label_x, label_y, anchor
    """
    if not points:
        return []

    if label_side == "both":
        return _layout_bilateral(points, image_width, image_height, font_size)

    return _layout_one_side(points, image_width, image_height, label_side, font_size)


def _layout_bilateral(
    points: list[dict],
    image_width: int,
    image_height: int,
    font_size: int,
) -> list[dict]:
    """Split labels by x-position: left half → left side, right half → right side."""
    mid_x = image_width / 2
    left_points = [p for p in points if p["point_x"] < mid_x]
    right_points = [p for p in points if p["point_x"] >= mid_x]

    # If all labels on one side, fall back to a reasonable split
    if not left_points:
        left_points = points[: len(points) // 2]
        right_points = points[len(points) // 2 :]
    elif not right_points:
        right_points = points[len(points) // 2 :]
        left_points = points[: len(points) // 2]

    left_layout = _layout_one_side(
        left_points, image_width, image_height, "left", font_size,
    )
    right_layout = _layout_one_side(
        right_points, image_width, image_height, "right", font_size,
    )

    return left_layout + right_layout


def _layout_one_side(
    points: list[dict],
    image_width: int,
    image_height: int,
    side: str,
    font_size: int,
) -> list[dict]:
    """Layout labels on one side of the image."""
    if not points:
        return []

    margin = 20
    font = _load_font(font_size, bold=True)
    label_height = font_size + 12

    # Sort by y-coordinate
    sorted_points = sorted(points, key=lambda p: p["point_y"])

    # Determine label column position and text anchor
    if side == "left":
        label_x = margin
        anchor = "left"
    elif side in ("bottom", "top"):
        label_x = image_width - margin
        anchor = "right"
    else:  # default: right
        label_x = image_width - margin
        anchor = "right"

    # Compute ideal y positions (matching point y), then resolve overlaps
    min_y = margin + font_size
    max_y = image_height - margin

    ideal_ys = [max(min_y, min(p["point_y"], max_y)) for p in sorted_points]

    # Resolve overlaps by pushing labels down
    resolved_ys: list[float] = []
    for y in ideal_ys:
        if resolved_ys and y < resolved_ys[-1] + label_height:
            y = resolved_ys[-1] + label_height
        resolved_ys.append(y)

    # If labels overflow the bottom, redistribute evenly
    if resolved_ys and resolved_ys[-1] > max_y:
        n = len(resolved_ys)
        available = max_y - min_y
        spacing = max(label_height, available / max(n, 1))
        resolved_ys = [min_y + i * spacing for i in range(n)]

    positions = []
    for i, point in enumerate(sorted_points):
        tw, _ = _text_size(font, point["label"])

        if anchor == "right":
            lx = image_width - margin
        else:
            lx = margin

        positions.append({
            "label": point["label"],
            "point_x": point["point_x"],
            "point_y": point["point_y"],
            "label_x": lx,
            "label_y": resolved_ys[i],
            "anchor": anchor,
        })

    return positions


def render_labels_on_png(
    image_path: Path,
    label_positions: list[dict],
    style: LabelStyle = "boxed_text",
    session_id: str = "",
) -> Path:
    """
    Render labels on the PNG image using Pillow.

    Args:
        image_path: Path to the raster image
        label_positions: List of dicts with label, point_x, point_y, label_x, label_y, anchor
        style: Annotation style
        session_id: For output path

    Returns:
        Path to the annotated PNG
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    renderers = {
        "plain_text": _render_plain_text,
        "boxed_text": _render_boxed_text,
        "numbered": _render_numbered,
        "color_coded": _render_color_coded,
        "minimal": _render_minimal,
        "textbook": _render_textbook,
    }

    renderer = renderers.get(style, _render_boxed_text)
    renderer(draw, img, label_positions)

    # Save output
    out_dir = GENERATED_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "annotated.png"
    img.save(str(out_path), "PNG")

    print(f"[label_renderer] Annotated PNG saved: {out_path}")
    return out_path


# ─── Rendering Styles ────────────────────────────────────────────────


def _render_plain_text(draw: ImageDraw.ImageDraw, img: Image.Image, positions: list[dict]):
    """Simple text labels with thin leader lines and dots."""
    font = _load_font(15)

    for pos in positions:
        px, py = pos["point_x"], pos["point_y"]
        lx, ly = pos["label_x"], pos["label_y"]
        text = pos["label"]

        # Leader line
        draw.line([(px, py), (lx, ly)], fill="#555555", width=1)

        # Dot at anatomical point
        r = 3
        draw.ellipse([px - r, py - r, px + r, py + r], fill="#333333")

        # Text
        tw, _ = _text_size(font, text)
        if pos.get("anchor") == "right":
            draw.text((lx - tw, ly - 8), text, fill="#222222", font=font)
        else:
            draw.text((lx, ly - 8), text, fill="#222222", font=font)


def _render_boxed_text(draw: ImageDraw.ImageDraw, img: Image.Image, positions: list[dict]):
    """Text in rounded rectangles with leader lines — the default style."""
    font = _load_font(13, bold=True)

    for pos in positions:
        px, py = pos["point_x"], pos["point_y"]
        lx, ly = pos["label_x"], pos["label_y"]
        text = pos["label"]

        # Measure text
        tw, th = _text_size(font, text)
        padding = 5

        # Box position (aligned to anchor side)
        if pos.get("anchor") == "right":
            box_x1 = lx - tw - padding * 2
            box_x2 = lx
            text_x = box_x1 + padding
            line_end_x = box_x1
        else:
            box_x1 = lx
            box_x2 = lx + tw + padding * 2
            text_x = lx + padding
            line_end_x = box_x2

        box_y1 = ly - th // 2 - padding
        box_y2 = ly + th // 2 + padding

        # Leader line (from point to box edge)
        draw.line([(px, py), (line_end_x, ly)], fill="#555555", width=1)

        # Dot at anatomical point
        r = 4
        draw.ellipse(
            [px - r, py - r, px + r, py + r],
            fill="#E74C3C", outline="#333333", width=1,
        )

        # Box background (solid white for reliability)
        draw.rounded_rectangle(
            [box_x1, box_y1, box_x2, box_y2],
            radius=4,
            fill="white",
            outline="#888888",
            width=1,
        )

        # Text
        draw.text((text_x, box_y1 + padding), text, fill="#222222", font=font)


def _render_numbered(draw: ImageDraw.ImageDraw, img: Image.Image, positions: list[dict]):
    """Circled numbers on image with numbered legend below."""
    num_font = _load_font(12, bold=True)
    legend_font = _load_font(14)
    legend_bold = _load_font(14, bold=True)

    w, h = img.size

    # Draw numbered circles on the image at each point
    for i, pos in enumerate(positions):
        px, py = pos["point_x"], pos["point_y"]
        num = str(i + 1)

        # Circled number
        r = 13
        draw.ellipse(
            [px - r, py - r, px + r, py + r],
            fill="#1A56DB", outline="white", width=2,
        )

        # Center the number text
        nw, nh = _text_size(num_font, num)
        draw.text((px - nw // 2, py - nh // 2), num, fill="white", font=num_font)

    # Legend at bottom
    legend_top = max(int(h * 0.78), h - 30 - len(positions) * 24)
    legend_top = max(10, legend_top)

    # Legend background
    draw.rectangle([8, legend_top - 8, w - 8, h - 8], fill="white", outline="#CCCCCC")

    for i, pos in enumerate(positions):
        y = legend_top + i * 22
        num = str(i + 1)
        text = pos["label"]

        draw.text((20, y), f"{num}.", fill="#1A56DB", font=legend_bold)
        draw.text((48, y), text, fill="#222222", font=legend_font)


def _render_color_coded(draw: ImageDraw.ImageDraw, img: Image.Image, positions: list[dict]):
    """Colored dots on image with matching colored text labels."""
    font = _load_font(14, bold=True)

    for i, pos in enumerate(positions):
        px, py = pos["point_x"], pos["point_y"]
        lx, ly = pos["label_x"], pos["label_y"]
        text = pos["label"]
        color = COLORS[i % len(COLORS)]

        # Colored dot on image
        r = 6
        draw.ellipse(
            [px - r, py - r, px + r, py + r],
            fill=color, outline="white", width=2,
        )

        # Leader line in matching color
        draw.line([(px, py), (lx, ly)], fill=color, width=2)

        # Colored text
        tw, _ = _text_size(font, text)
        if pos.get("anchor") == "right":
            draw.text((lx - tw, ly - 8), text, fill=color, font=font)
        else:
            draw.text((lx, ly - 8), text, fill=color, font=font)


def _render_minimal(draw: ImageDraw.ImageDraw, img: Image.Image, positions: list[dict]):
    """Very thin lines, small text, minimal visual clutter."""
    font = _load_font(11)

    for pos in positions:
        px, py = pos["point_x"], pos["point_y"]
        lx, ly = pos["label_x"], pos["label_y"]
        text = pos["label"]

        # Thin leader line
        draw.line([(px, py), (lx, ly)], fill="#AAAAAA", width=1)

        # Small dot
        r = 2
        draw.ellipse([px - r, py - r, px + r, py + r], fill="#999999")

        # Small text
        tw, _ = _text_size(font, text)
        if pos.get("anchor") == "right":
            draw.text((lx - tw, ly - 6), text, fill="#666666", font=font)
        else:
            draw.text((lx, ly - 6), text, fill="#666666", font=font)


def _render_textbook(draw: ImageDraw.ImageDraw, img: Image.Image, positions: list[dict]):
    """Classic textbook look with bold labels and right-angle leader lines."""
    font = _load_font(14, bold=True)

    for pos in positions:
        px, py = pos["point_x"], pos["point_y"]
        lx, ly = pos["label_x"], pos["label_y"]
        text = pos["label"]
        tw, _ = _text_size(font, text)

        # Right-angle leader line
        if pos.get("anchor") == "right":
            bend_x = lx - tw - 15
        else:
            bend_x = lx + tw + 15

        # Horizontal from point to bend
        draw.line([(px, py), (bend_x, py)], fill="#333333", width=2)
        # Vertical from bend to label row
        draw.line([(bend_x, py), (bend_x, ly)], fill="#333333", width=2)
        # Horizontal to label
        draw.line([(bend_x, ly), (lx, ly)], fill="#333333", width=2)

        # Small square at anatomical point
        s = 3
        draw.rectangle([px - s, py - s, px + s, py + s], fill="#333333")

        # Bold text
        if pos.get("anchor") == "right":
            text_x = lx - tw
            draw.text((text_x, ly - 9), text, fill="#1a1a1a", font=font)
            # Underline
            draw.line([(text_x, ly + 8), (lx, ly + 8)], fill="#333333", width=1)
        else:
            draw.text((lx, ly - 9), text, fill="#1a1a1a", font=font)
            draw.line([(lx, ly + 8), (lx + tw, ly + 8)], fill="#333333", width=1)


# ─── SVG Export ──────────────────────────────────────────────────────


def render_labels_as_svg(
    image_path: Path,
    label_positions: list[dict],
    style: LabelStyle = "boxed_text",
    session_id: str = "",
) -> Path:
    """
    Generate an SVG with the raster image embedded + vector label overlays.
    Labels are real <text> elements — editable, searchable, scalable.
    """
    with Image.open(image_path) as img:
        w, h = img.size

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    # Build SVG
    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'xmlns:xlink="http://www.w3.org/1999/xlink" '
                 f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">')

    # Embedded raster image
    lines.append(f'  <image width="{w}" height="{h}" '
                 f'href="data:image/png;base64,{b64}" />')

    # Style-dependent rendering
    lines.append('  <g id="annotations" font-family="Arial, Helvetica, sans-serif">')

    for i, pos in enumerate(label_positions):
        px, py = pos["point_x"], pos["point_y"]
        lx, ly = pos["label_x"], pos["label_y"]
        text = _svg_escape(pos["label"])
        anchor_attr = "end" if pos.get("anchor") == "right" else "start"

        # Pick color based on style
        if style == "color_coded":
            color = COLORS[i % len(COLORS)]
        else:
            color = "#555555"

        dot_color = COLORS[i % len(COLORS)] if style == "color_coded" else "#E74C3C"
        stroke_w = 2 if style in ("color_coded", "textbook") else 1

        # Leader line
        lines.append(f'    <line x1="{px}" y1="{py}" x2="{lx}" y2="{ly}" '
                     f'stroke="{color}" stroke-width="{stroke_w}" />')

        # Dot at anatomical point
        r = 4
        lines.append(f'    <circle cx="{px}" cy="{py}" r="{r}" '
                     f'fill="{dot_color}" stroke="white" stroke-width="1" />')

        # Text label
        text_color = color if style == "color_coded" else "#222222"
        font_size = 14
        font_weight = "bold" if style in ("boxed_text", "textbook", "color_coded") else "normal"

        # Optional background rect for boxed_text style
        if style == "boxed_text":
            # Approximate text width (SVG can't do getbbox, so estimate)
            approx_tw = len(text) * 8
            padding = 5
            if anchor_attr == "end":
                rx = lx - approx_tw - padding
            else:
                rx = lx - padding
            ry = ly - font_size // 2 - padding
            rw = approx_tw + padding * 2
            rh = font_size + padding * 2
            lines.append(f'    <rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" '
                         f'rx="4" fill="white" fill-opacity="0.9" stroke="#888" stroke-width="1" />')

        lines.append(f'    <text x="{lx}" y="{ly + 5}" text-anchor="{anchor_attr}" '
                     f'font-size="{font_size}" font-weight="{font_weight}" '
                     f'fill="{text_color}">{text}</text>')

    lines.append('  </g>')
    lines.append('</svg>')

    svg_content = "\n".join(lines)

    out_dir = GENERATED_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "annotated.svg"
    out_path.write_text(svg_content, encoding="utf-8")

    print(f"[label_renderer] Annotated SVG saved: {out_path}")
    return out_path


def _svg_escape(text: str) -> str:
    """Escape special characters for SVG text content."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
