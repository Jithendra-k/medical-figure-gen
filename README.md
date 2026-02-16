# Medical Figure Generator

A chat-based tool for generating labeled medical and scientific textbook-style figures. It combines LLM-driven diagram planning, cloud image generation APIs (Gemini Imagen / DALL-E 3), vision-model image analysis and label placement, and Pillow/SVG rendering — all orchestrated through a FastAPI web interface with iterative refinement.

---

## Architecture

The system is a **5-stage pipeline** with an optional **refinement loop**, exposed through a chat UI:

```
User Message
    │
    ▼
┌──────────────────────────┐
│ 1. Diagram Planner       │  LLM generates drawing prompt + suggested labels + auditor QA
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 2. Image Generator       │  Gemini Imagen or DALL-E 3 produces a clean raster PNG
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 3. Vision Analyzer       │  VLM inspects the actual image → confirms which structures
│                          │  are truly visible (planner labels are hints, not forced)
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 4. Vision Label Placer   │  2-pass zone-classification via VLM locates confirmed structures
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 5. Label Renderer        │  Pillow renders annotated PNG + embedded-raster SVG
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 6. Refiner (chat loop)   │  Hybrid intent classifier routes edits, style changes,
│                          │  label repositioning, or full regeneration
└──────────────────────────┘
```

### Key Design Decisions

- **Image-first, then analyze.** Labels are NOT pre-decided. The planner suggests labels as hints, but the **Vision Analyzer** inspects the actual generated image and confirms only the structures it can truly see. This prevents labeling structures that don't exist in the image.
- **No local GPU required.** All heavy lifting (image generation, vision analysis) uses cloud APIs.
- **Positive-only image prompts.** The planner writes purely descriptive prompts (no prohibitions). A regex safety net strips any negations before the prompt reaches the image model, preventing "attention to forbidden concepts" artifacts.
- **Classification over regression.** Vision models are poor at estimating exact pixel coordinates. The label placer converts coordinate estimation into a two-pass classification task using numbered zone overlays, which VLMs handle reliably.
- **4×4 fine grid.** The fine localisation pass uses a 4×4 grid (16 sub-cells A1–D4) per zone, achieving ~64px precision on a 1024×1024 image.
- **Hybrid intent classification.** The refiner uses 23+ deterministic regex patterns before falling back to an LLM, ensuring fast and reliable intent routing for common refinement requests.
- **Multi-view support.** The planner can generate landscape multi-panel illustrations (e.g., 3 views of hand anatomy) with bilateral label placement.

### Project Structure

```
medical-figure-gen/
├── .env                          # API keys (not committed)
├── .env.example                  # Template for .env
├── .gitignore
├── config.py                     # Central config: paths, models, settings
├── app.py                        # FastAPI server — orchestrates pipeline + serves UI
├── requirements.txt
├── pipeline/
│   ├── __init__.py
│   ├── diagram_planner.py        # Stage 1 — LLM plan generation + auditor loop
│   ├── image_generator.py        # Stage 2 — Cloud image generation (Gemini / OpenAI)
│   ├── vision_analyzer.py        # Stage 3 — VLM confirms visible structures in image
│   ├── vision_label_placer.py    # Stage 4 — 2-pass zone-classification label placement
│   ├── label_renderer.py         # Stage 5 — Pillow PNG + SVG annotation rendering
│   ├── refiner.py                # Stage 6 — Multi-action refinement router
│   └── utils.py                  # Shared utilities (retry decorator, etc.)
├── templates/
│   └── index.html                # Chat UI (split panel: chat + three preview tabs)
└── static/
    └── generated/                # Runtime output (per-session folders)
```

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10 (conda env recommended) |
| **API keys** | At least one of: Google Gemini API key, OpenAI API key |
| **No GPU required** | All computation is done via cloud APIs |

---

## Setup

### 1. Clone & enter the project

```bash
git clone <your-repo-url> medical-figure-gen
cd medical-figure-gen
```

### 2. Create conda environment

```bash
conda create -n svgrender python=3.10 -y
conda activate svgrender
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
GOOGLE_GENERATIVE_AI_API_KEY=your-google-api-key
OPENAI_API_KEY=your-openai-api-key

IMAGE_GEN_PROVIDER=gemini    # or "openai"
LLM_PROVIDER=gemini          # or "openai"
```

---

## Running

```bash
# From the medical-figure-gen directory:
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

> **Windows note:** If `conda activate` doesn't work in VS Code terminal, use the full path:
> ```
> C:\Users\<you>\anaconda3\envs\svgrender\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
> ```

Open **http://127.0.0.1:8000** in your browser.

### Quick Start

1. Type: `"Draw a human hand anatomy diagram"`
2. Wait for the pipeline (20–40 seconds depending on API — 5 stages run sequentially)
3. The annotated PNG and SVG appear in the preview tabs
4. Refine: `"Move the labels to the left"`, `"Use numbered style"`, `"Add a label for Ulna"`
5. Regenerate: `"Give me a better image"` or `"Regenerate with more detail"`
6. Multi-view: `"Show me 3 types of hand anatomy"` — generates a landscape image with bilateral labels

### UI Controls

| Control | Effect |
|---|---|
| **LLM Provider** dropdown | Gemini or OpenAI for planning, analysis & vision tasks |
| **Image Provider** dropdown | Gemini Imagen or DALL-E 3 for image generation |
| **Label Style** dropdown | `boxed_text` (default), `plain_text`, `numbered`, `color_coded`, `minimal`, `textbook` |
| **Skip Annotation** checkbox | Returns only the raw raster image (no labels) |

---

## Configuration Reference

All settings live in [config.py](config.py):

| Setting | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `"gemini"` | LLM for planning, vision analysis, & refinement (`"gemini"` or `"openai"`) |
| `IMAGE_GEN_PROVIDER` | `"gemini"` | Image generation API (`"gemini"` or `"openai"`) |
| `GEMINI_LLM_MODEL` | `"gemini-3-pro-preview"` | Gemini model for text & vision tasks |
| `GEMINI_IMAGE_MODEL` | `"gemini-2.5-flash-image"` | Gemini model for image generation |
| `OPENAI_IMAGE_MODEL` | `"dall-e-3"` | OpenAI image model |
| `OPENAI_LLM_MODEL` | `"gpt-4o"` | OpenAI text & vision model |
| `DEFAULT_LABEL_STYLE` | `"boxed_text"` | Default annotation style |
| `AUDITOR_MAX_ITERATIONS` | `2` | Max planner↔auditor feedback loops (0 = skip) |
| `FONT_PATH` | `arial.ttf` | Font for label rendering (Windows default) |
| `FONT_BOLD_PATH` | `arialbd.ttf` | Bold font for label rendering |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the chat UI |
| `POST` | `/api/generate` | Full pipeline run (5 stages) |
| `POST` | `/api/refine` | Refine existing figure |
| `GET` | `/api/sessions` | List active sessions |
| `GET` | `/generated/{session_id}/{file}` | Serve generated files (PNG, SVG) |

### Generate request

```json
{
  "message": "Draw a human hand anatomy diagram",
  "label_style": "boxed_text",
  "skip_annotate": false,
  "llm_provider": "gemini",
  "image_provider": "gemini"
}
```

### Generate response

```json
{
  "session_id": "a1b2c3d4",
  "steps": [
    { "stage": "diagram_planner", "result": { "suggested_labels": 12, "type": "anatomy" } },
    { "stage": "image_generator", "result": "/generated/a1b2c3d4/raster.png" },
    { "stage": "vision_analyzer", "result": "Confirmed 10 of 12 suggested labels" },
    { "stage": "vision_label_placer", "result": "Located 10 labels" },
    { "stage": "label_renderer", "result": "boxed_text style, 10 labels" }
  ],
  "raster_url": "/generated/a1b2c3d4/raster.png",
  "annotated_url": "/generated/a1b2c3d4/annotated.png",
  "svg_url": "/generated/a1b2c3d4/annotated.svg",
  "plan": {
    "labels": ["Distal Phalanges", "Middle Phalanges", "..."],
    "description": "Dorsal view of a human right hand showing all 27 skeletal bones.",
    "style": "colored_diagram",
    "diagram_type": "anatomy",
    "label_side": "right"
  }
}
```

### Refine request

```json
{
  "session_id": "a1b2c3d4",
  "message": "Change labels to numbered style",
  "label_style": "numbered",
  "llm_provider": "gemini",
  "image_provider": "gemini"
}
```

---

## How Each Stage Works

### Stage 1 — Diagram Planner (`pipeline/diagram_planner.py`)

Sends the user message to an LLM with a system prompt that produces structured JSON:

```json
{
  "drawing_prompt": "A single scientific medical illustration of a human right hand...",
  "labels": ["Distal Phalanges", "Middle Phalanges", "..."],
  "label_side": "right",
  "style": "colored_diagram",
  "description": "Dorsal view of a human right hand showing all 27 skeletal bones.",
  "diagram_type": "anatomy"
}
```

The `labels` array here are **suggestions** — hints to guide the Vision Analyzer in Stage 3. They are not forced onto the final output.

Key rules enforced by the planner prompt:
- **Drawing prompts are purely positive** — no prohibitions ("do not", "no text", etc.)
- **Max ~60 words** (single view) or **~100 words** (multi-view) — concise descriptions generate cleaner images
- An **auditor** LLM reviews the plan and can request fixes (up to `AUDITOR_MAX_ITERATIONS` rounds)
- **Multi-view support** — if the user asks for multiple views (e.g., "3 types of hand anatomy"), sets `diagram_type: "multi_view"` and `label_side: "both"` for bilateral label layout

### Stage 2 — Image Generator (`pipeline/image_generator.py`)

Calls the configured image API (Gemini Imagen or DALL-E 3) with the drawing prompt.

Before sending, the generator:
1. **Strips negations** via regex — removes any "do not", "no text", etc. the planner may have included
2. **Appends a short suffix** — `". Single illustration, pure visual artwork, no text."` (minimal, effective)
3. **Saves** the result as `raster.png` (1024×1024 for single view, 1792×1024 for multi-view with DALL-E 3)

### Stage 3 — Vision Analyzer (`pipeline/vision_analyzer.py`) ✨ NEW

**Why this stage exists:** Without it, labels are pre-decided before the image exists. If the image generator doesn't render a structure the planner expected, that label would point to nothing.

The Vision Analyzer inspects the **actual generated image** and confirms which structures are truly visible:

1. Receives the raster image + planner's suggested labels as hints
2. Sends both to a VLM (Gemini or GPT-4o) with a structured prompt
3. The VLM identifies all distinct visible structures, using the hints as guidance but not forced to include them
4. Returns `{labels: [...confirmed...], label_side: "right"|"left"|"both"}`
5. Falls back to planner suggestions if VLM returns nothing (graceful degradation)

The confirmed labels overwrite the plan's label list — from this point on, only structures that exist in the image are labeled.

Uses `response_mime_type="application/json"` with Gemini for reliable structured output, plus a multi-strategy JSON parser as a safety net.

### Stage 4 — Vision Label Placer (`pipeline/vision_label_placer.py`)

Locates where each confirmed structure appears in the generated image using a **two-pass zone-classification** approach:

**Pass 1 — Coarse (zone identification):**
- Overlays an N×N grid of **numbered zones** (red circles with white numbers) onto the image
- Asks the vision model: *"Which zone number contains each structure?"* — a classification task
- Grid size adapts to label count (e.g., 5×5 = 25 zones for 12 labels)

**Pass 2 — Fine (sub-cell localisation):**
- For each occupied zone, crops that region and overlays a **4×4 labeled grid** (A1–D4, 16 sub-cells)
- Asks the vision model: *"Which sub-cell contains the center of the structure?"*
- Maps zone + sub-cell back to full-image pixel coordinates

This approach yields **~64px precision** on a 1024×1024 image and guarantees structures in different zones get different coordinates — solving the "everything clusters at center" problem that plagued direct pixel-coordinate estimation.

### Stage 5 — Label Renderer (`pipeline/label_renderer.py`)

Takes the located points and renders annotations in two formats:

- **Annotated PNG** — Pillow draws labels with leader lines directly on the raster image
- **Annotated SVG** — Embeds the raster as a `<image>` element with vector `<text>` and `<line>` overlays

Supports 6 annotation styles: `plain_text`, `boxed_text`, `numbered`, `color_coded`, `minimal`, `textbook`.

Layout features:
- Sorts labels by y-coordinate for clean vertical ordering
- Resolves overlapping labels with automatic spacing
- Configurable label side (right, left, top, bottom, around)
- **Bilateral layout** — when `label_side: "both"`, labels are split by x-position: left-half structures go to the left margin, right-half structures go to the right margin (ideal for multi-view diagrams)

### Stage 6 — Refiner (`pipeline/refiner.py`)

Handles iterative chat-based refinement with a **hybrid intent classifier**:

1. **Deterministic patterns** — 23+ regex rules match common requests instantly:
   - `"change style to numbered"` → `style_change`
   - `"remove the capitate label"` → `remove_label`
   - `"add a label for ulna"` → `add_label`
   - `"move the labels"` → `reposition_labels`
   - `"give me a better image"` → `regenerate`

2. **LLM fallback** — Only used when no regex matches; classifies into one of 6 action types

Supported actions:

| Action | What happens |
|---|---|
| `style_change` | Re-renders labels with new style (no API calls) |
| `label_edit` | Modifies label text, re-renders |
| `remove_label` | Removes a label, re-renders |
| `add_label` | Adds label to plan, re-runs vision analyzer + placer + renderer |
| `reposition_labels` | Re-runs vision analyzer + placer on existing image |
| `regenerate` | Re-runs full pipeline (planner → image gen → analyzer → placer → render) |

All refinement paths that re-run the vision pipeline now go through the Vision Analyzer first, ensuring only visible structures are labeled even after refinement.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `conda activate` doesn't work in VS Code terminal | Use full Python path: `C:\Users\...\anaconda3\envs\svgrender\python.exe` |
| "Gemini did not return an image" | The model may not support your prompt. Try `IMAGE_GEN_PROVIDER=openai` |
| Image has multiple panels / collage | Check that `_NEGATION_RE` in `image_generator.py` is stripping prohibitions. The planner prompt should produce only positive descriptions |
| Labels all cluster at center | Ensure `vision_label_placer.py` is using the 2-pass zone approach (check server logs for "Pass 1: NxN zone grid") |
| Labels point to non-existent structures | The Vision Analyzer should prevent this. Check logs for `[vision_analyzer] Confirmed X of Y suggested labels` |
| Rate limit errors (429) | Built-in retry with exponential backoff handles this. Reduce request frequency or check API quotas |
| Font not found on Linux/macOS | Set `FONT_PATH` and `FONT_BOLD_PATH` in `.env` to valid TTF paths |
| VLM returns prose instead of JSON | Gemini uses `response_mime_type="application/json"` to force JSON. All parsers have multi-strategy fallbacks |

---

## Adding a New Provider

To add a new LLM or image generation provider:

1. Add the API key to `.env` and `config.py`
2. Add a new branch in the relevant pipeline file:
   - **LLM tasks:** `diagram_planner.py`, `vision_analyzer.py`, `vision_label_placer.py`, `refiner.py`
   - **Image generation:** `image_generator.py`
3. Add the provider name to `IMAGE_GEN_PROVIDER` / `LLM_PROVIDER` options

---

## License

MIT
