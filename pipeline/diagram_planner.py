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

1. "drawing_prompt": A SHORT, purely descriptive prompt for an image-generation model.
   RULES FOR THE DRAWING PROMPT:
   - Maximum 60 words. Be concise.
   - Start with "A single scientific medical illustration of..."
   - Describe ONLY what to draw: the subject, anatomy, visual style, colors, viewpoint.
   - Mention: clean white background, clear outlines, clinical textbook style.
   - Include the phrase "centered in the left two-thirds of the canvas".
   - NEVER include prohibitions or negative phrases (no "do not", "no text", "without", "never", "must not").
   - NEVER mention text, labels, letters, words, annotations, arrows, lines, watermarks, captions — not even to forbid them.
   - Write ONLY positive visual description.

2. "labels": An array of anatomical/scientific terms for annotation (rendered separately after image generation).
   - Order logically top-to-bottom as they appear in the image.
   - 3-15 labels depending on complexity.

3. "label_side": "right" (default), "left", "top", "bottom", or "around".

4. "style": One of: "line_diagram", "colored_diagram", "cross_section", "flowchart", "comparison".

5. "description": 1-sentence summary of the figure.

6. "diagram_type": "anatomy" or "relational".

Respond ONLY with valid JSON. No markdown, no code fences.

Example input: "draw a human hand anatomy"
Example output:
{
  "drawing_prompt": "A single scientific medical illustration of a human right hand viewed from the dorsal side showing complete skeletal anatomy with all 27 bones clearly visible. Distal phalanges, middle phalanges, proximal phalanges, metacarpals, and eight carpal bones. Clear outlines, light bone-yellow tones, clean white background, clinical textbook style, centered in the left two-thirds of the canvas.",
  "labels": ["Distal Phalanges", "Middle Phalanges", "Proximal Phalanges", "Metacarpals", "Trapezium", "Trapezoid", "Capitate", "Hamate", "Scaphoid", "Lunate", "Triquetrum", "Pisiform"],
  "label_side": "right",
  "style": "colored_diagram",
  "description": "Dorsal view of a human right hand showing all 27 skeletal bones.",
  "diagram_type": "anatomy"
}

MULTI-VIEW DIAGRAMS:
If the user asks for multiple views, layers, or types of the same subject side by side
(e.g., "3 types of hand anatomy", "show skeletal and muscular views", "comparison of layers"):
- Set "diagram_type" to "multi_view"
- Set "label_side" to "both"
- Start the drawing_prompt with "A wide landscape scientific medical illustration showing three views of..."
- Maximum 100 words for multi-view prompts (more detail needed)
- Describe each view (left, center, right) and what anatomy it shows
- All other rules still apply (positive-only language, purely visual description)
- Labels should cover structures from ALL views, ordered left-to-right then top-to-bottom

Example multi-view input: "show me 3 types of hand anatomy"
Example multi-view output:
{
  "drawing_prompt": "A wide landscape scientific medical illustration showing three dorsal views of a human right hand aligned side by side. Left hand showing tendons and extensor muscles with visible muscle fibers in natural tones. Center hand showing deep anatomy with muscles, red arteries, and yellow nerves. Right hand showing complete skeletal anatomy with all 27 bones in light bone-yellow. Same hand size in each view, vertically aligned. Clean white background, clinical textbook style, clear outlines.",
  "labels": ["Extensor pollicis brevis", "Extensor digitorum", "Interossei", "Abductor pollicis brevis", "Extensor carpi ulnaris", "Extensor carpi radialis longus", "Extensor carpi radialis brevis", "Abductor pollicis longus", "DIP joint", "Distal phalanx", "PIP joint", "Extensor pollicis longus", "Middle phalanx", "Proximal phalanx", "MCP joint", "Metacarpals", "Carpals", "Ulna", "Radius"],
  "label_side": "both",
  "style": "colored_diagram",
  "description": "Three views of a human right hand: tendons, deep anatomy, and skeletal structure.",
  "diagram_type": "multi_view"
}"""


AUDITOR_SYSTEM_PROMPT = """You are a diagram plan auditor. Review the following diagram plan and check for:

1. Is the drawing_prompt concise and purely descriptive? (Under 60 words for single view, under 100 for multi_view)
2. Does the drawing_prompt contain ONLY positive descriptions? It must NOT contain any prohibitions ("no", "do not", "without", "never", "must not") or mentions of text/labels/annotations.
3. Are the labels comprehensive? Are any obvious anatomical/scientific parts missing?
4. Are labels ordered logically (as they'd appear in the image)?
5. If diagram_type is "multi_view", is label_side set to "both"? If single view, does the prompt include "centered in the left two-thirds of the canvas"?

IMPORTANT NOTES:
- "label_side" and "labels" are RENDERING parameters used AFTER image generation. Do NOT suggest removing them.
- If the drawing_prompt contains any negative phrases like "no text", "do not", "without labels", etc., mark it as needs_fixes — the prompt must be purely positive.

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
