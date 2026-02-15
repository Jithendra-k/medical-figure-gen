"""
Stage 1: Diagram Planner
Takes raw user message → returns a structured diagram plan.
Includes planner + auditor feedback loop.
"""

from __future__ import annotations

import json

from google import genai
from google.genai import types
import openai

from config import (
    GOOGLE_API_KEY, OPENAI_API_KEY,
    LLM_PROVIDER, GEMINI_LLM_MODEL, OPENAI_LLM_MODEL,
    AUDITOR_MAX_ITERATIONS,
)
from pipeline.utils import retry_on_rate_limit


PLANNER_SYSTEM_PROMPT = """You are a scientific illustration planner for medical/biomedical textbook figures.

Given a user's description, produce a JSON diagram plan with these fields:

1. "drawing_prompt": A detailed prompt for an image generation model.
   - Start with "A single clean scientific medical illustration of..."
   - Describe the subject matter in detail (anatomy, orientation, structures visible)
   - Specify: clean white background, clear outlines, minimal flat coloring, clinical style
   - Include spatial hint: "Position the main illustration in the left 65% of the canvas, leaving the right 35% as blank white space for future annotations"
   
   CRITICAL — SINGLE IMAGE RULE (include in the drawing_prompt):
     * "ONE single illustration only. Do NOT create a collage, mosaic, montage, grid, or multiple panels."
     * "Do NOT show the subject from multiple angles or views. Show exactly ONE view."
     * "No inset images, no zoomed views, no multiple copies of the subject."
   
   CRITICAL — NO TEXT / NO ANNOTATIONS (include ALL in the drawing_prompt):
     * "The image must contain absolutely NO text of any kind."
     * "No labels, no letters, no words, no numbers, no annotations, no captions, no watermarks."
     * "No leader lines, no arrows, no annotation lines, no callout lines, no pointers."
     * "Do not write any words or characters on or near the illustration."
     * "The entire image must be purely visual artwork with zero text elements and zero annotation elements."
   
   - Describe ONLY visual elements. Never mention text/labels/annotations/lines in the drawing_prompt.
   - Be specific about which structures should be clearly visible and distinguishable.
   - Keep under 250 words.

2. "labels": An array of short text labels that should annotate the diagram.
   - Each label is an anatomical/scientific term (e.g., "Phalanges", "Cell Body")
   - Order them logically (top-to-bottom or proximal-to-distal as they'd appear in the image)
   - Include 3-15 labels depending on complexity
   - NOTE: These labels will be rendered AFTER image generation. They are NOT part of the image.

3. "label_side": Where labels should be rendered in the annotation overlay: "right" (default for anatomy), "left", "top", "bottom", or "around" (for flowcharts/cycles)
   - NOTE: This controls the annotation renderer, NOT the image content. It has nothing to do with text in the image.

4. "style": One of: "line_diagram", "colored_diagram", "cross_section", "flowchart", "comparison"

5. "description": A 1-sentence summary of what the figure shows (used by the vision model to locate structures).

6. "diagram_type": "anatomy" (single unified object with labeled parts) or "relational" (multiple discrete objects with connections)

Respond ONLY with valid JSON. No markdown, no code fences.

Example input: "draw a human hand anatomy"
Example output:
{
  "drawing_prompt": "A single clean scientific medical illustration of a human right hand viewed from the dorsal (back) side, showing the complete skeletal anatomy with all 27 bones clearly visible and distinguishable. Show the five distal phalanges at the fingertips, five middle phalanges, five proximal phalanges, five metacarpal bones in the palm, and eight carpal bones at the wrist (scaphoid, lunate, triquetrum, pisiform, trapezium, trapezoid, capitate, hamate). Each bone should be drawn with clear distinct outlines and light bone-yellow tones. Clean white background, clinical medical textbook style. Position the main illustration in the left 65% of the canvas, leaving the right 35% as blank white space. ONE single illustration only. Do NOT create a collage, mosaic, montage, grid, or multiple panels. Do NOT show the subject from multiple angles or views. The image must contain absolutely NO text of any kind. No labels, no letters, no words, no numbers, no annotations, no captions, no watermarks. No leader lines, no arrows, no annotation lines, no callout lines, no pointers. The entire image must be purely visual artwork with zero text elements and zero annotation elements.",
  "labels": ["Distal Phalanges", "Middle Phalanges", "Proximal Phalanges", "Metacarpals", "Trapezium", "Trapezoid", "Capitate", "Hamate", "Scaphoid", "Lunate", "Triquetrum", "Pisiform"],
  "label_side": "right",
  "style": "colored_diagram",
  "description": "Dorsal view of a single human right hand showing all 27 skeletal bones.",
  "diagram_type": "anatomy"
}"""


AUDITOR_SYSTEM_PROMPT = """You are a diagram plan auditor. Review the following diagram plan and check for:

1. Does the drawing_prompt clearly describe the visual content for an image generation model?
2. Are the labels comprehensive? Are any obvious anatomical/scientific parts missing?
3. Does the drawing_prompt contain strong anti-text instructions? (It should explicitly say NO text, no labels, no letters, no words, no numbers.)
4. Is the spatial hint present? (telling the model to leave empty space on one side for annotations)
5. Are labels ordered logically (as they'd appear in the image)?
6. Is the description specific enough for a vision model to identify the structures?

IMPORTANT NOTES:
- "label_side" is a RENDERING parameter that controls where the annotation software places labels AFTER image generation. It has NOTHING to do with text in the image. Do NOT suggest removing it or say it contradicts no-text instructions.
- "labels" are rendered by our annotation software AFTER the image is generated. They are NOT part of the image. Do NOT suggest removing them.
- Focus ONLY on the quality of the "drawing_prompt" and the completeness of the "labels" list.

If the plan is good, respond with:
{"status": "approved", "feedback": ""}

If the plan needs fixes, respond with:
{"status": "needs_fixes", "feedback": "specific description of what to fix"}

Respond ONLY with valid JSON. No markdown, no code fences."""


def create_plan(user_message: str, provider: str | None = None) -> dict:
    """
    Create a diagram plan from user message, with optional auditor loop.

    Returns dict with keys: drawing_prompt, labels, label_side, style, description, diagram_type
    """
    provider = provider or LLM_PROVIDER

    # Stage 1: Generate initial plan
    plan = _generate_plan(user_message, provider)
    print(f"[planner] Initial plan: {len(plan.get('labels', []))} labels, "
          f"type={plan.get('diagram_type')}, side={plan.get('label_side')}")

    # Stage 1b: Auditor loop
    if AUDITOR_MAX_ITERATIONS > 0:
        plan = _audit_plan(plan, user_message, provider)

    return plan


def _generate_plan(user_message: str, provider: str) -> dict:
    """Generate an initial diagram plan using the LLM."""
    if provider == "gemini":
        return _plan_with_gemini(user_message)
    elif provider == "openai":
        return _plan_with_openai(user_message)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


@retry_on_rate_limit(max_retries=3, initial_wait=10)
def _plan_with_gemini(user_message: str) -> dict:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_LLM_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=PLANNER_SYSTEM_PROMPT,
            temperature=0.3,
        ),
    )
    return _parse_plan(response.text or "")


def _plan_with_openai(user_message: str) -> dict:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )
    return _parse_plan(response.choices[0].message.content or "")


def _audit_plan(plan: dict, original_request: str, provider: str) -> dict:
    """Run the auditor loop to refine the plan."""
    for i in range(AUDITOR_MAX_ITERATIONS):
        audit_prompt = (
            f"Original user request: {original_request}\n\n"
            f"Diagram plan to review:\n{json.dumps(plan, indent=2)}"
        )

        if provider == "gemini":
            audit_result = _audit_with_gemini(audit_prompt)
        else:
            audit_result = _audit_with_openai(audit_prompt)

        if audit_result.get("status") == "approved":
            print(f"[planner] Plan approved by auditor (iteration {i + 1})")
            break

        feedback = audit_result.get("feedback", "")
        if feedback:
            print(f"[planner] Auditor feedback (iteration {i + 1}): {feedback}")
            # Re-generate with feedback
            fix_prompt = (
                f"Original request: {original_request}\n\n"
                f"Previous plan:\n{json.dumps(plan, indent=2)}\n\n"
                f"Auditor feedback: {feedback}\n\n"
                f"Please generate a corrected diagram plan addressing the feedback."
            )
            if provider == "gemini":
                plan = _plan_with_gemini(fix_prompt)
            else:
                plan = _plan_with_openai(fix_prompt)

    return plan


@retry_on_rate_limit(max_retries=3, initial_wait=10)
def _audit_with_gemini(prompt: str) -> dict:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=AUDITOR_SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    return _parse_json(response.text or "")


def _audit_with_openai(prompt: str) -> dict:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return _parse_json(response.choices[0].message.content or "")


def _parse_plan(text: str) -> dict:
    """Parse LLM response into a plan dict."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {
            "drawing_prompt": text,
            "labels": [],
            "label_side": "right",
            "style": "colored_diagram",
            "description": text[:100],
            "diagram_type": "anatomy",
        }

    # Validate/default required keys
    defaults = {
        "drawing_prompt": "",
        "labels": [],
        "label_side": "right",
        "style": "colored_diagram",
        "description": "",
        "diagram_type": "anatomy",
    }
    for key, default in defaults.items():
        if key not in result:
            result[key] = default

    return result


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"status": "approved", "feedback": ""}
