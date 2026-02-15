"""
Stage 3: Vision-Based Label Placer  (Numbered-Region Approach)

Instead of asking VLMs for exact pixel coordinates (regression — unreliable),
we convert the problem to CLASSIFICATION:

  Pass 1 (Coarse):  Overlay ~24-30 numbered rectangular zones on the image.
                     Ask VLM: "Which zone number contains each structure?"
  Pass 2 (Fine):    Crop the identified zone, divide into 3×3 sub-cells (A-I).
                     Ask VLM: "Which sub-cell is the exact centre of the structure?"

This two-pass approach is dramatically more accurate because VLMs are excellent
at classification ("is X in zone 12?") but poor at regression ("what pixel is X at?").
"""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import (
    GOOGLE_API_KEY, OPENAI_API_KEY,
    LLM_PROVIDER, GEMINI_LLM_MODEL, OPENAI_LLM_MODEL,
)
from pipeline.utils import retry_on_rate_limit

# ─────────────────────────── Prompts ───────────────────────────

COARSE_PROMPT = """You are a medical image analyst. The image shows: {description}

The image has been divided into {num_zones} numbered zones.
Each zone is a rectangle outlined in red with a large WHITE number on a RED circle at its centre.

For each anatomical structure listed below, tell me which ZONE NUMBER it is located in.
Look carefully at the image — find where each structure actually appears, then report the zone number that overlaps that area.

Structures to locate:
{labels}

RULES:
- Different structures that are in the SAME area can share the same zone number.
- If a structure spans multiple zones, pick the zone that contains its CENTER.
- LOOK at the actual image content, not just the zone numbers.

Return a JSON array (one entry per structure):
[{{"label": "exact label text", "zone": <zone_number_int>}}, ...]

Respond with ONLY the JSON array, no markdown."""


FINE_PROMPT = """You are a medical image analyst doing precise localisation.

This is a CROPPED region of a larger medical image. The crop shows part of: {description}

The crop has been divided into a 3×3 grid of sub-cells labelled:
  A | B | C      (top row)
  D | E | F      (middle row)
  G | H | I      (bottom row)

The grid lines and letters are drawn on the image.

For each structure listed below, tell me which sub-cell (A–I) contains the CENTER of that structure.
If the structure is not visible in this crop, respond with "E" (centre) as best estimate.

Structures:
{labels}

Return a JSON array:
[{{"label": "exact label text", "cell": "A"}}, ...]

Respond with ONLY the JSON array, no markdown."""


# ─────────────────────── Grid overlay builders ─────────────────────────

def _build_zone_grid(
    image_bytes: bytes,
    cols: int = 6,
    rows: int = 5,
) -> tuple[bytes, list[dict]]:
    """
    Overlay numbered rectangular zones on the image.

    Returns:
        (image_with_zones_bytes, zone_list)
        zone_list: [{id: int, x1, y1, x2, y2, cx, cy}, ...]
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    cell_w = w / cols
    cell_h = h / rows

    try:
        font = ImageFont.truetype("arialbd.ttf", max(20, int(min(cell_w, cell_h) * 0.35)))
    except (OSError, IOError):
        font = ImageFont.load_default()

    zones: list[dict] = []
    zone_id = 1

    for r in range(rows):
        for c in range(cols):
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            zones.append({
                "id": zone_id,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "cx": cx, "cy": cy,
            })

            # Draw zone border
            draw.rectangle([x1, y1, x2, y2], outline=(220, 40, 40, 200), width=2)

            # Draw zone number in a red circle at centre
            num_text = str(zone_id)
            bbox = font.getbbox(num_text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            circle_r = max(tw, th) // 2 + 6
            draw.ellipse(
                [cx - circle_r, cy - circle_r, cx + circle_r, cy + circle_r],
                fill=(220, 40, 40, 200),
            )
            draw.text(
                (cx - tw // 2, cy - th // 2 - 1),
                num_text,
                fill=(255, 255, 255, 255),
                font=font,
            )

            zone_id += 1

    result = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue(), zones


def _build_fine_grid(crop_bytes: bytes) -> bytes:
    """
    Overlay a 3×3 sub-cell grid (A-I) on a cropped region for fine localisation.
    """
    img = Image.open(io.BytesIO(crop_bytes)).convert("RGBA")
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    cell_w = w / 3
    cell_h = h / 3

    try:
        font = ImageFont.truetype("arialbd.ttf", max(16, int(min(cell_w, cell_h) * 0.3)))
    except (OSError, IOError):
        font = ImageFont.load_default()

    cell_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

    for idx, letter in enumerate(cell_labels):
        r = idx // 3
        c = idx % 3
        x1 = int(c * cell_w)
        y1 = int(r * cell_h)
        x2 = int((c + 1) * cell_w)
        y2 = int((r + 1) * cell_h)
        cx_cell = (x1 + x2) // 2
        cy_cell = (y1 + y2) // 2

        # Cell border
        draw.rectangle([x1, y1, x2, y2], outline=(40, 120, 220, 180), width=2)

        # Letter label in a blue circle
        bbox = font.getbbox(letter)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pr = max(tw, th) // 2 + 4
        draw.ellipse(
            [cx_cell - pr, cy_cell - pr, cx_cell + pr, cy_cell + pr],
            fill=(40, 120, 220, 180),
        )
        draw.text(
            (cx_cell - tw // 2, cy_cell - th // 2 - 1),
            letter,
            fill=(255, 255, 255, 255),
            font=font,
        )

    result = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────── Sub-cell → pixel mapping ─────────────────────────

# Each letter maps to (fractional_x, fractional_y) within the zone
_SUBCELL_OFFSETS: dict[str, tuple[float, float]] = {
    "A": (1/6, 1/6), "B": (3/6, 1/6), "C": (5/6, 1/6),
    "D": (1/6, 3/6), "E": (3/6, 3/6), "F": (5/6, 3/6),
    "G": (1/6, 5/6), "H": (3/6, 5/6), "I": (5/6, 5/6),
}


def _subcell_to_pixel(zone: dict, cell_letter: str) -> tuple[int, int]:
    """Convert a zone + sub-cell letter to full-image pixel coords."""
    fx, fy = _SUBCELL_OFFSETS.get(cell_letter.upper(), (0.5, 0.5))
    px = int(zone["x1"] + (zone["x2"] - zone["x1"]) * fx)
    py = int(zone["y1"] + (zone["y2"] - zone["y1"]) * fy)
    return px, py


# ─────────────────────── Main entry point ─────────────────────────

def locate_labels(
    image_path: Path,
    labels: list[str],
    description: str,
    provider: str | None = None,
) -> list[dict]:
    """
    Two-pass numbered-region approach for accurate label placement.

    Pass 1: Coarse — identify which numbered zone each structure is in.
    Pass 2: Fine  — within each zone, identify the 3×3 sub-cell.

    Returns list of dicts: [{label, point_x, point_y}, ...]
    """
    provider = provider or LLM_PROVIDER

    if not labels:
        return []

    with Image.open(image_path) as img:
        width, height = img.size

    with open(image_path, "rb") as f:
        original_bytes = f.read()

    # ── Pass 1: Coarse zone assignment ───────────────────────
    cols, rows = _pick_grid_size(width, height, len(labels))
    zone_img_bytes, zones = _build_zone_grid(original_bytes, cols=cols, rows=rows)

    coarse_prompt = COARSE_PROMPT.format(
        description=description,
        num_zones=len(zones),
        labels="\n".join(f"  {i+1}. {lbl}" for i, lbl in enumerate(labels)),
    )

    print(f"[vision_placer] Pass 1: {cols}x{rows} zone grid ({len(zones)} zones), "
          f"{len(labels)} labels -> {provider}")

    coarse_result = _call_vision(zone_img_bytes, coarse_prompt, provider)
    zone_map = _parse_coarse(coarse_result, labels, zones)

    for lbl, z in zone_map.items():
        print(f"[vision_placer]   {lbl} -> zone {z['id']}")

    # ── Pass 2: Fine sub-cell within each zone ───────────────
    print(f"[vision_placer] Pass 2: Fine localisation within zones...")

    # Group labels by zone to minimise VLM calls
    zone_to_labels: dict[int, list[str]] = {}
    for lbl, z in zone_map.items():
        zone_to_labels.setdefault(z["id"], []).append(lbl)

    positions: list[dict] = []

    for z_id, z_labels in zone_to_labels.items():
        zone = next(z for z in zones if z["id"] == z_id)

        # Crop the zone from the original image with small padding for context
        with Image.open(io.BytesIO(original_bytes)) as full_img:
            pad = 20
            crop_box = (
                max(0, zone["x1"] - pad),
                max(0, zone["y1"] - pad),
                min(width, zone["x2"] + pad),
                min(height, zone["y2"] + pad),
            )
            crop = full_img.crop(crop_box)
            crop_buf = io.BytesIO()
            crop.save(crop_buf, format="PNG")
            crop_bytes = crop_buf.getvalue()

        # Overlay 3×3 fine grid on the crop
        fine_img = _build_fine_grid(crop_bytes)

        fine_prompt = FINE_PROMPT.format(
            description=description,
            labels="\n".join(f"  - {lbl}" for lbl in z_labels),
        )

        fine_result = _call_vision(fine_img, fine_prompt, provider)
        cell_map = _parse_fine(fine_result, z_labels)

        for lbl in z_labels:
            cell_letter = cell_map.get(lbl, "E")
            px, py = _subcell_to_pixel(zone, cell_letter)
            px = max(10, min(px, width - 10))
            py = max(10, min(py, height - 10))

            positions.append({
                "label": lbl,
                "point_x": px,
                "point_y": py,
            })
            print(f"[vision_placer]   {lbl} -> zone {z_id} cell {cell_letter} -> ({px}, {py})")

    # Ensure all labels are present
    found = {p["label"] for p in positions}
    for i, lbl in enumerate(labels):
        if lbl not in found:
            print(f"[vision_placer] Warning: '{lbl}' missing, using fallback")
            positions.append({
                "label": lbl,
                "point_x": width // 2,
                "point_y": int(height * (0.1 + 0.8 * i / max(len(labels), 1))),
            })

    print(f"[vision_placer] Located {len(positions)} labels successfully")
    return positions


# ─────────────────────── Helpers ─────────────────────────

def _pick_grid_size(
    width: int, height: int, num_labels: int,
) -> tuple[int, int]:
    """Pick a cols x rows grid that gives roughly 2x as many zones as labels."""
    target = max(num_labels * 2, 12)
    aspect = width / height
    rows = max(3, int(math.sqrt(target / aspect)))
    cols = max(3, int(rows * aspect))
    while cols * rows < target:
        if cols <= rows:
            cols += 1
        else:
            rows += 1
    return cols, rows


def _parse_coarse(
    result: list[dict],
    labels: list[str],
    zones: list[dict],
) -> dict[str, dict]:
    """Parse coarse pass result -> {label: zone_dict}."""
    zone_by_id = {z["id"]: z for z in zones}
    max_zone = max(z["id"] for z in zones)

    mapping: dict[str, dict] = {}
    for item in result:
        lbl = item.get("label", "")
        z_id = item.get("zone")
        if isinstance(z_id, int) and 1 <= z_id <= max_zone:
            mapping[lbl] = zone_by_id[z_id]

    # Fill in missing labels — spread across zones
    missing = [lbl for lbl in labels if lbl not in mapping]
    if missing:
        used_zones = sorted(set(z["id"] for z in mapping.values())) if mapping else []
        available = [z for z in zones if z["id"] not in used_zones] or zones
        for i, lbl in enumerate(missing):
            z = available[i % len(available)]
            mapping[lbl] = z
            print(f"[vision_placer] Warning: '{lbl}' not in coarse response, assigned to zone {z['id']}")

    return mapping


def _parse_fine(result: list[dict], labels: list[str]) -> dict[str, str]:
    """Parse fine pass result -> {label: cell_letter}."""
    valid_cells = set("ABCDEFGHI")
    mapping: dict[str, str] = {}

    for item in result:
        lbl = item.get("label", "")
        cell = str(item.get("cell", "E")).upper().strip()
        if cell in valid_cells:
            mapping[lbl] = cell
        else:
            mapping[lbl] = "E"

    return mapping


def _call_vision(image_bytes: bytes, prompt: str, provider: str) -> list[dict]:
    """Call the appropriate vision model."""
    if provider == "gemini":
        return _locate_with_gemini(image_bytes, prompt)
    elif provider == "openai":
        return _locate_with_openai(image_bytes, prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


@retry_on_rate_limit(max_retries=3, initial_wait=10)
def _locate_with_gemini(image_bytes: bytes, prompt: str) -> list[dict]:
    """Use Gemini Vision."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)

    response = client.models.generate_content(
        model=GEMINI_LLM_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
        config=types.GenerateContentConfig(temperature=0.1),
    )

    return _parse_json_array(response.text or "")


@retry_on_rate_limit(max_retries=3, initial_wait=10)
def _locate_with_openai(image_bytes: bytes, prompt: str) -> list[dict]:
    """Use GPT-4o Vision."""
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    b64 = base64.b64encode(image_bytes).decode()

    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
        temperature=0.1,
        max_tokens=4000,
    )

    return _parse_json_array(response.choices[0].message.content or "")


def _parse_json_array(text: str) -> list[dict]:
    """Parse a JSON array from VLM response text."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        for v in result.values():
            if isinstance(v, list):
                return v
        return []
    except json.JSONDecodeError:
        print(f"[vision_placer] Failed to parse JSON: {text[:300]}")
        return []
