"""
Medical Figure Generator - FastAPI Chat Server
Orchestrates the 5-stage pipeline:
  1. Diagram Planner (LLM)
  2. Image Generator (API)
  3. Vision Analyzer (VLM) — confirms which structures are visible
  4. Vision Label Placer (VLM) — locates confirmed structures
  5. Label Renderer (Pillow/SVG)
"""

import uuid
import traceback
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import GENERATED_DIR, PROJECT_ROOT, DEFAULT_LABEL_STYLE, LLM_PROVIDER
from pipeline.diagram_planner import create_plan
from pipeline.image_generator import generate_image
from pipeline.vision_analyzer import analyze_image
from pipeline.vision_label_placer import locate_labels
from pipeline.label_renderer import (
    compute_label_layout, render_labels_on_png, render_labels_as_svg,
)
from pipeline.refiner import RefineSession

app = FastAPI(title="Medical Figure Generator")

# Static files & templates
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

# In-memory session store  {session_id: {...}}
sessions: dict[str, dict] = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/generate")
async def generate(request: Request):
    """
    Full pipeline: plan → image → vision place → render labels.
    Body: { "message": "...", "session_id": "..." (optional),
            "label_style": "boxed_text", "skip_annotate": false,
            "llm_provider": "gemini", "image_provider": "gemini" }
    """
    body = await request.json()
    message = body.get("message", "").strip()
    session_id = body.get("session_id") or str(uuid.uuid4())[:8]
    skip_annotate = body.get("skip_annotate", False)
    label_style = body.get("label_style") or DEFAULT_LABEL_STYLE
    llm_provider = body.get("llm_provider")
    image_provider = body.get("image_provider")

    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    steps_completed: list[dict] = []

    try:
        # ── Stage 1: Diagram Planner ──────────────────────────
        plan = create_plan(message, provider=llm_provider)
        suggested_labels = plan.get("labels", [])
        steps_completed.append({
            "stage": "diagram_planner",
            "result": {
                "suggested_labels": len(suggested_labels),
                "type": plan.get("diagram_type"),
            },
        })

        # ── Stage 2: Generate raster image ───────────────────
        raster_path = generate_image(plan, session_id, provider=image_provider)
        raster_url = f"/generated/{session_id}/raster.png"
        steps_completed.append({
            "stage": "image_generator",
            "result": raster_url,
        })

        annotated_png_url = None
        svg_url = None
        label_positions: list[dict] = []

        if not skip_annotate:
            # ── Stage 3: Vision Analyzer ──────────────────────
            # Analyze the ACTUAL generated image to find which
            # structures are truly visible (not just what planner guessed)
            analysis = analyze_image(
                image_path=raster_path,
                description=plan.get("description", "medical diagram"),
                suggested_labels=suggested_labels,
                provider=llm_provider,
            )
            confirmed_labels = analysis.get("labels", [])
            # Use analyzer's label_side recommendation if planner didn't set multi_view
            if plan.get("diagram_type") != "multi_view":
                plan["label_side"] = analysis.get("label_side", plan.get("label_side", "right"))
            # Update plan with confirmed labels (these are what actually exist in the image)
            plan["labels"] = confirmed_labels

            steps_completed.append({
                "stage": "vision_analyzer",
                "result": f"Confirmed {len(confirmed_labels)} of {len(suggested_labels)} suggested labels",
            })

            if confirmed_labels:
                # ── Stage 4: Vision Label Placer ──────────────────
                points = locate_labels(
                    image_path=raster_path,
                    labels=confirmed_labels,
                    description=plan.get("description", "medical diagram"),
                    provider=llm_provider,
                )
                steps_completed.append({
                    "stage": "vision_label_placer",
                    "result": f"Located {len(points)} labels",
                })

                # ── Stage 5: Label Renderer ───────────────────────
                from PIL import Image
                with Image.open(raster_path) as img:
                    w, h = img.size

                label_positions = compute_label_layout(
                    points=points,
                    image_width=w,
                    image_height=h,
                    label_side=plan.get("label_side", "right"),
                )

                # Render annotated PNG
                annotated_png_path = render_labels_on_png(
                    image_path=raster_path,
                    label_positions=label_positions,
                    style=label_style,  # type: ignore[arg-type]
                    session_id=session_id,
                )
                annotated_png_url = f"/generated/{session_id}/annotated.png"

                # Render annotated SVG
                svg_path = render_labels_as_svg(
                    image_path=raster_path,
                    label_positions=label_positions,
                    style=label_style,  # type: ignore[arg-type]
                    session_id=session_id,
                )
                svg_url = f"/generated/{session_id}/annotated.svg"

                steps_completed.append({
                    "stage": "label_renderer",
                    "result": f"{label_style} style, {len(label_positions)} labels",
                })

        # ── Store session for refinement ──────────────────────
        sessions[session_id] = {
            "plan": plan,
            "label_positions": label_positions,
            "label_style": label_style,
            "raster_path": raster_path,
            "llm_provider": llm_provider,
            "image_provider": image_provider,
            "refiner": RefineSession(session_id, plan, label_positions, label_style),
        }

        return JSONResponse({
            "session_id": session_id,
            "steps": steps_completed,
            "raster_url": raster_url,
            "annotated_url": annotated_png_url,
            "svg_url": svg_url,
            "plan": {
                "labels": plan.get("labels", []),
                "description": plan.get("description", ""),
                "style": plan.get("style", ""),
                "diagram_type": plan.get("diagram_type", ""),
                "label_side": plan.get("label_side", ""),
            },
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({
            "error": str(e),
            "steps": steps_completed,
            "session_id": session_id,
        }, status_code=500)


@app.post("/api/refine")
async def refine(request: Request):
    """
    Refine an existing figure.
    Body: { "session_id": "...", "message": "...", "label_style": "..." }
    """
    body = await request.json()
    session_id = body.get("session_id", "")
    message = body.get("message", "").strip()
    new_style = body.get("label_style")
    llm_provider = body.get("llm_provider")

    if session_id not in sessions:
        return JSONResponse(
            {"error": "Session not found. Generate a figure first."},
            status_code=404,
        )

    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    session = sessions[session_id]
    refiner: RefineSession = session["refiner"]

    try:
        result = refiner.refine(message, new_style=new_style)
        print(f"[app] Refine result: action={result.get('action')}, needs_regen={result.get('needs_regeneration')}, needs_vision={result.get('needs_vision_rerun')}")

        if result.get("needs_regeneration") or result.get("needs_vision_rerun"):
            new_plan = result.get("new_plan", session["plan"])
            img_provider = body.get("image_provider") or session.get("image_provider")
            vision_provider = llm_provider or session.get("llm_provider")

            if result.get("action") == "regenerate":
                # Full regeneration: re-plan + re-generate image + re-analyze
                refinement_req = new_plan.pop("_refinement_request", message)
                combined_prompt = (
                    session["plan"].get("description", "")
                    + ". " + refinement_req
                )
                print(f"[app] Re-running planner for: {combined_prompt[:80]}...")
                new_plan = create_plan(combined_prompt, provider=vision_provider)
                suggested_labels = new_plan.get("labels", [])
                print(f"[app] New plan: {len(suggested_labels)} suggested labels")

                raster_path = generate_image(
                    new_plan, session_id, provider=img_provider
                )
                raster_url = f"/generated/{session_id}/raster.png"

                # Analyze the NEW image to confirm visible structures
                analysis = analyze_image(
                    image_path=raster_path,
                    description=new_plan.get("description", "medical diagram"),
                    suggested_labels=suggested_labels,
                    provider=vision_provider,
                )
                confirmed_labels = analysis.get("labels", [])
                if new_plan.get("diagram_type") != "multi_view":
                    new_plan["label_side"] = analysis.get("label_side", new_plan.get("label_side", "right"))
                new_plan["labels"] = confirmed_labels

            elif result.get("needs_vision_rerun"):
                # Re-run vision placer on existing image (reposition / add_label)
                # Re-analyze to confirm which structures are visible
                raster_path = session["raster_path"]
                raster_url = f"/generated/{session_id}/raster.png"
                suggested_labels = new_plan.get("labels", [])

                analysis = analyze_image(
                    image_path=raster_path,
                    description=new_plan.get("description", "medical diagram"),
                    suggested_labels=suggested_labels,
                    provider=vision_provider,
                )
                confirmed_labels = analysis.get("labels", [])
                if new_plan.get("diagram_type") != "multi_view":
                    new_plan["label_side"] = analysis.get("label_side", new_plan.get("label_side", "right"))
                new_plan["labels"] = confirmed_labels

            else:
                raster_path = generate_image(
                    new_plan, session_id, provider=img_provider
                )
                raster_url = f"/generated/{session_id}/raster.png"
                # Re-analyze new image
                suggested_labels = new_plan.get("labels", [])
                analysis = analyze_image(
                    image_path=raster_path,
                    description=new_plan.get("description", "medical diagram"),
                    suggested_labels=suggested_labels,
                    provider=vision_provider or LLM_PROVIDER,
                )
                confirmed_labels = analysis.get("labels", [])
                new_plan["labels"] = confirmed_labels

            # Re-run vision placer + renderer with confirmed labels
            points = locate_labels(
                image_path=raster_path,
                labels=new_plan.get("labels", []),
                description=new_plan.get("description", "medical diagram"),
                provider=vision_provider,
            )

            from PIL import Image
            with Image.open(raster_path) as img:
                w, h = img.size

            style = new_style or session["label_style"]
            label_positions = compute_label_layout(
                points=points,
                image_width=w,
                image_height=h,
                label_side=new_plan.get("label_side", "right"),
            )

            render_labels_on_png(
                image_path=raster_path,
                label_positions=label_positions,
                style=style,  # type: ignore[arg-type]
                session_id=session_id,
            )
            render_labels_as_svg(
                image_path=raster_path,
                label_positions=label_positions,
                style=style,  # type: ignore[arg-type]
                session_id=session_id,
            )

            # Update session
            session["plan"] = new_plan
            session["label_positions"] = label_positions
            session["label_style"] = style
            session["raster_path"] = raster_path
            refiner.plan = new_plan
            refiner.label_positions = label_positions
            refiner.label_style = style

            return JSONResponse({
                "session_id": session_id,
                "action": "regenerated",
                "explanation": result["explanation"],
                "raster_url": raster_url,
                "annotated_url": f"/generated/{session_id}/annotated.png",
                "svg_url": f"/generated/{session_id}/annotated.svg",
            })
        else:
            # Label-only or style-only edits — just re-render
            annotated_url = None
            svg_url_out = None
            if result.get("annotated_png"):
                annotated_url = f"/generated/{session_id}/annotated.png"
            if result.get("annotated_svg"):
                svg_url_out = f"/generated/{session_id}/annotated.svg"

            return JSONResponse({
                "session_id": session_id,
                "action": result["action"],
                "explanation": result["explanation"],
                "annotated_url": annotated_url,
                "svg_url": svg_url_out,
            })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/sessions")
async def list_sessions():
    """List active sessions."""
    return JSONResponse({
        sid: {
            "labels": s["plan"].get("labels", []),
            "description": s["plan"].get("description", ""),
        }
        for sid, s in sessions.items()
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
