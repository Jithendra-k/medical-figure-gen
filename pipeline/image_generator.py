"""
Stage 2: Image Generator
Takes structured prompt → generates raster PNG image via API.
Supports Gemini Imagen and OpenAI DALL-E 3.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from PIL import Image

from config import (
    GOOGLE_API_KEY, OPENAI_API_KEY,
    IMAGE_GEN_PROVIDER, GEMINI_IMAGE_MODEL, OPENAI_IMAGE_MODEL,
    GENERATED_DIR,
)
from pipeline.utils import retry_on_rate_limit


# Hard suffix appended to EVERY image generation prompt as a safety net.
# Even if the planner forgets or the LLM ignores instructions, this ensures
# the image gen model gets explicit anti-text, anti-collage rules.
_PROMPT_SUFFIX = (
    " STRICT RULES: This must be ONE single illustration — no collage, no grid, "
    "no multiple panels, no multiple views, no montage. "
    "Absolutely NO text, NO letters, NO numbers, NO labels, NO words anywhere in the image. "
    "NO leader lines, NO arrows, NO annotation lines, NO callout lines. "
    "Pure visual artwork only."
)


def generate_image(spec: dict, session_id: str, provider: str | None = None) -> Path:
    """
    Generate a raster image from the structured spec.

    Args:
        spec: dict from prompt_structurer with 'drawing_prompt' key
        session_id: unique ID for this generation session
        provider: 'gemini' or 'openai' (overrides config default)

    Returns:
        Path to the saved PNG image
    """
    provider = provider or IMAGE_GEN_PROVIDER
    prompt = spec["drawing_prompt"] + _PROMPT_SUFFIX

    if provider == "gemini":
        img = _generate_with_gemini(prompt)
    elif provider == "openai":
        img = _generate_with_openai(prompt)
    else:
        raise ValueError(f"Unknown image provider: {provider}")

    # Save to disk
    out_dir = GENERATED_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raster.png"
    img.save(str(out_path), "PNG")

    return out_path


@retry_on_rate_limit(max_retries=3, initial_wait=15)
def _generate_with_gemini(prompt: str) -> Image.Image:
    """Generate image using Gemini's image generation capability."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["image", "text"],  # type: ignore[call-arg]
        ),
    )

    # Extract image from response parts
    candidates = response.candidates
    if not candidates or not candidates[0].content or not candidates[0].content.parts:
        raise RuntimeError("Gemini returned empty response")
    for part in candidates[0].content.parts:
        if part.inline_data is not None and part.inline_data.data is not None:
            img_bytes: bytes = part.inline_data.data
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    raise RuntimeError("Gemini did not return an image. Response: " + str(response.text))


def _generate_with_openai(prompt: str) -> Image.Image:
    """Generate image using OpenAI DALL-E 3."""
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    response = client.images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        response_format="b64_json",
        n=1,
    )

    if not response.data:
        raise RuntimeError("OpenAI returned empty response")
    img_b64 = response.data[0].b64_json or ""
    img_bytes = base64.b64decode(img_b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
