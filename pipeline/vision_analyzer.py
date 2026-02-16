"""
Stage 3a: Vision Analyzer
Analyzes the generated image to determine which anatomical structures
are ACTUALLY visible, then produces confirmed labels.

This runs AFTER image generation and BEFORE label placement, ensuring
we only try to label structures that actually exist in the image.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image

from config import (
    GOOGLE_API_KEY, OPENAI_API_KEY,
    LLM_PROVIDER, GEMINI_LLM_MODEL, OPENAI_LLM_MODEL,
)
from pipeline.utils import retry_on_rate_limit


ANALYZE_PROMPT = """You are a medical/scientific image analyst.

Look at this image carefully. It is a scientific medical illustration of: {description}

Your task: Identify ALL distinct anatomical structures, organs, bones, muscles,
or scientific components that are CLEARLY VISIBLE and DISTINGUISHABLE in this image.

{hint_section}

RULES:
- Only list structures you can ACTUALLY SEE as distinct visual elements in the image.
- Use standard medical/scientific terminology.
- Order them logically: top-to-bottom, left-to-right as they appear in the image.
- Each label must refer to a visually distinct region — something a reader could point to.
- Do NOT invent structures that aren't visible.
- Return between 5 and 20 labels.

IMPORTANT: Your ENTIRE response must be a valid JSON object and nothing else.
Format: {{"labels": ["Structure 1", "Structure 2", ...], "label_side": "right"}}

Set "label_side" to:
- "right" if the subject is centered or left-of-center (default)
- "left" if the subject is right-of-center
- "both" if the image has multiple panels or the subject fills the full width

Do NOT include any explanation, markdown, or commentary — output ONLY the JSON object."""


def analyze_image(
    image_path: Path,
    description: str,
    suggested_labels: list[str] | None = None,
    provider: str | None = None,
) -> dict:
    """
    Analyze the generated image to determine which structures are actually visible.

    Args:
        image_path: Path to the generated raster image.
        description: Brief description from the planner.
        suggested_labels: Optional hint labels from the planner (used as suggestions, not forced).
        provider: "gemini" or "openai".

    Returns:
        dict with:
            "labels": list[str]  — confirmed visible structures
            "label_side": str    — recommended label placement side
    """
    provider = provider or LLM_PROVIDER

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Build hint section — gives the VLM context but doesn't force labels
    if suggested_labels:
        hint_section = (
            "The illustration was intended to show these structures (use as hints, "
            "but ONLY include ones you can actually see):\n"
            + "\n".join(f"  - {lbl}" for lbl in suggested_labels)
        )
    else:
        hint_section = ""

    prompt = ANALYZE_PROMPT.format(
        description=description,
        hint_section=hint_section,
    )

    print(f"[vision_analyzer] Analyzing image for visible structures -> {provider}")

    if provider == "gemini":
        result = _analyze_with_gemini(image_bytes, prompt)
    elif provider == "openai":
        result = _analyze_with_openai(image_bytes, prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    labels = result.get("labels", [])
    label_side = result.get("label_side", "right")

    # Sanity: if analysis returned nothing useful, fall back to suggested labels
    if not labels and suggested_labels:
        print("[vision_analyzer] Warning: analysis returned no labels, using planner suggestions")
        labels = suggested_labels

    # Cap at 20 labels
    labels = labels[:20]

    print(f"[vision_analyzer] Found {len(labels)} visible structures, label_side={label_side}")
    for lbl in labels:
        print(f"[vision_analyzer]   - {lbl}")

    return {"labels": labels, "label_side": label_side}


@retry_on_rate_limit(max_retries=3, initial_wait=10)
def _analyze_with_gemini(image_bytes: bytes, prompt: str) -> dict:
    """Use Gemini Vision to analyze the image."""
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
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    return _parse_analysis(response.text or "")


@retry_on_rate_limit(max_retries=3, initial_wait=10)
def _analyze_with_openai(image_bytes: bytes, prompt: str) -> dict:
    """Use GPT-4o Vision to analyze the image."""
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
        temperature=0.2,
        max_tokens=2000,
    )

    return _parse_analysis(response.choices[0].message.content or "")


def _parse_analysis(text: str) -> dict:
    """Parse the VLM analysis response into {labels, label_side}."""
    import re

    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict) and "labels" in result:
            return result
        # Maybe it returned just an array
        if isinstance(result, list):
            return {"labels": result, "label_side": "right"}
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in prose
    match = re.search(r'\{[^}]*"labels"\s*:\s*\[[^\]]*\][^}]*\}', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if "labels" in result:
                return result
        except json.JSONDecodeError:
            pass

    # Try to find JSON array embedded
    match = re.search(r'\[\s*"[^"]+(?:"\s*,\s*"[^"]+)*"\s*\]', text)
    if match:
        try:
            labels = json.loads(match.group(0))
            return {"labels": labels, "label_side": "right"}
        except json.JSONDecodeError:
            pass

    print(f"[vision_analyzer] Failed to parse analysis: {text[:300]}")
    return {"labels": [], "label_side": "right"}
