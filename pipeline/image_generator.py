"""
Stage 2: Image Generator
Takes structured prompt → generates raster PNG image via API.
Supports Gemini Imagen and OpenAI DALL-E 3.
"""

from __future__ import annotations

import base64
import io
import os
import re
from pathlib import Path

from PIL import Image

from config import (
    GOOGLE_API_KEY, OPENAI_API_KEY,
    IMAGE_GEN_PROVIDER, GEMINI_IMAGE_MODEL, OPENAI_IMAGE_MODEL,
    GENERATED_DIR,
)
from pipeline.utils import retry_on_rate_limit


# Short suffixes — image gen models work best with brief positive prompts.
_PROMPT_SUFFIX = ". Single illustration, pure visual artwork, no text."
_PROMPT_SUFFIX_MULTI = ". Pure visual artwork, no text."

# Regex to strip any negative/prohibition phrases the planner may have included.
_NEGATION_RE = re.compile(
    r"(?i)"
    r"(\b(do not|don'?t|never|must not|should not|cannot|no|without|absolutely no|strictly no)\b"
    r"[^.;]*[.;]?\s*)",
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
    is_multi_view = spec.get("diagram_type") == "multi_view"

    # Clean the drawing prompt: strip any negative/prohibition phrases and append short suffix
    raw_prompt = spec["drawing_prompt"]
    cleaned = _NEGATION_RE.sub("", raw_prompt).strip()
    # Collapse any leftover double spaces or double periods
    cleaned = re.sub(r"  +", " ", cleaned)
    cleaned = re.sub(r"\.\..+", ".", cleaned)

    suffix = _PROMPT_SUFFIX_MULTI if is_multi_view else _PROMPT_SUFFIX
    prompt = cleaned + suffix
    print(f"[image_gen] {'multi-view' if is_multi_view else 'single'} "
          f"Prompt ({len(prompt)} chars): {prompt[:120]}...")

    if provider == "gemini":
        img = _generate_with_gemini(prompt)
    elif provider == "openai":
        size = "1792x1024" if is_multi_view else "1024x1024"
        img = _generate_with_openai(prompt, size=size)
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


def _generate_with_openai(prompt: str, size: str = "1024x1024") -> Image.Image:
    """Generate image using OpenAI DALL-E 3."""
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    response = client.images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=prompt,
        size=size,
        quality="standard",
        response_format="b64_json",
        n=1,
    )

    if not response.data:
        raise RuntimeError("OpenAI returned empty response")
    img_b64 = response.data[0].b64_json or ""
    img_bytes = base64.b64decode(img_b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
