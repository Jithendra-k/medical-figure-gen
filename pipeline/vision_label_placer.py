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

IMPORTANT: Your ENTIRE response must be a valid JSON array and nothing else.  
Format: [{{"label": "exact label text", "zone": <zone_number_int>}}, ...]
Example: [{{"label": "Femur", "zone": 12}}, {{"label": "Tibia", "zone": 18}}]
Do NOT include any explanation, markdown, or commentary — output ONLY the JSON array."""


FINE_PROMPT = """You are a medical image analyst doing precise localisation.

This is a CROPPED region of a larger medical image. The crop shows part of: {description}

The crop has been divided into a 4×4 grid of 16 sub-cells:
  Row 1:  A1 | A2 | A3 | A4
  Row 2:  B1 | B2 | B3 | B4
  Row 3:  C1 | C2 | C3 | C4
  Row 4:  D1 | D2 | D3 | D4

The grid lines and labels are drawn on the image.

For each structure listed below, identify which sub-cell contains the EXACT CENTER of that structure.
Be as precise as possible — look at where the structure actually is, not the middle of the crop.
If the structure is not visible in this crop, respond with "B2" as best estimate.

Structures:
{labels}

IMPORTANT: Your ENTIRE response must be a valid JSON array and nothing else.
Format: [{{"label": "exact label text", "cell": "A1"}}, ...]
Example: [{{"label": "Femur", "cell": "C3"}}, {{"label": "Tibia", "cell": "D1"}}]
Do NOT include any explanation, markdown, or commentary — output ONLY the JSON array."""


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
    Overlay a 4×4 sub-cell grid on a cropped region for fine localisation.
    Labels: A1-A4, B1-B4, C1-C4, D1-D4.
    """
    img = Image.open(io.BytesIO(crop_bytes)).convert("RGBA")
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    grid_n = 4
    cell_w = w / grid_n
    cell_h = h / grid_n

    try:
        font = ImageFont.truetype("arialbd.ttf", max(14, int(min(cell_w, cell_h) * 0.25)))
    except (OSError, IOError):
        font = ImageFont.load_default()

    row_letters = ["A", "B", "C", "D"]

    for r in range(grid_n):
        for c in range(grid_n):
            label = f"{row_letters[r]}{c + 1}"
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)
            cx_cell = (x1 + x2) // 2
            cy_cell = (y1 + y2) // 2

            # Cell border
            draw.rectangle([x1, y1, x2, y2], outline=(40, 120, 220, 180), width=2)

            # Label in a blue circle
            bbox = font.getbbox(label)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            pr = max(tw, th) // 2 + 4
            draw.ellipse(
                [cx_cell - pr, cy_cell - pr, cx_cell + pr, cy_cell + pr],
                fill=(40, 120, 220, 180),
            )
            draw.text(
                (cx_cell - tw // 2, cy_cell - th // 2 - 1),
                label,
                fill=(255, 255, 255, 255),
                font=font,
            )

    result = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────── Sub-cell → pixel mapping ─────────────────────────

# Each 4×4 cell label maps to (fractional_x, fractional_y) within the zone
_SUBCELL_OFFSETS: dict[str, tuple[float, float]] = {}
for _r_idx, _row_letter in enumerate("ABCD"):
    for _c_idx in range(4):
        _key = f"{_row_letter}{_c_idx + 1}"
        _SUBCELL_OFFSETS[_key] = (
            (2 * _c_idx + 1) / 8,   # center of column
            (2 * _r_idx + 1) / 8,   # center of row
        )


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
            cell_letter = cell_map.get(lbl, "B2")
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
    """Parse fine pass result -> {label: cell_label}."""
    valid_cells = set(_SUBCELL_OFFSETS.keys())
    mapping: dict[str, str] = {}

    for item in result:
        lbl = item.get("label", "")
        cell = str(item.get("cell", "B2")).upper().strip()
        if cell in valid_cells:
            mapping[lbl] = cell
        else:
            mapping[lbl] = "B2"  # center-ish fallback

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
    """Use Gemini Vision with forced JSON output."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)

    response = client.models.generate_content(
        model=GEMINI_LLM_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",  # Force JSON output
        ),
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


import re as _re

def _parse_json_array(text: str) -> list[dict]:
    """Robustly extract a JSON array from VLM response text."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        for v in result.values():
            if isinstance(v, list):
                return v
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array embedded in prose
    # Match the outermost [...] that contains {"label":
    match = _re.search(r'\[\s*\{[^\]]*"label"[^\]]*\]', text, _re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Last resort: find all {"label": ..., "zone": ...} or {"label": ..., "cell": ...}
    items = _re.findall(
        r'\{\s*"label"\s*:\s*"([^"]+)"\s*,\s*"(zone|cell)"\s*:\s*["\s]*(\w+)["\s]*\}',
        text,
    )
    if items:
        extracted = []
        for label, key, value in items:
            val = int(value) if value.isdigit() else value
            extracted.append({"label": label, key: val})
        return extracted

    print(f"[vision_placer] Failed to parse JSON: {text[:300]}")
    return []
