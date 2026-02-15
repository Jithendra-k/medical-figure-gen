# Medical Figure Generator — Technical Report

## 1. Project Overview

**Medical Figure Generator** is a web application that generates **labeled medical/scientific diagrams** from natural-language descriptions. A user types a request like *"draw a human hand anatomy"* in a chat interface, and the system produces a labeled, textbook-style figure through a multi-stage AI pipeline.

The project is inspired by [figurelabs.ai](https://figurelabs.ai) and targets use cases such as medical textbooks, educational materials, and clinical presentations.

---

## 2. Architecture

### 2.1 High-Level Flow

```
┌───────────────┐     ┌────────────────────┐     ┌────────────────┐     ┌──────────────┐     ┌────────────────┐
│  User Chat UI │────▶│ Stage 1: Prompt     │────▶│ Stage 2: Image │────▶│ Stage 3:     │────▶│ Stage 4:       │
│  (Browser)    │     │ Structurer (LLM)    │     │ Generator (API)│     │ Vectorizer   │     │ Annotator(LLM) │
└───────────────┘     └────────────────────┘     └────────────────┘     │ (LIVE/DiffVG)│     └──────────┬─────┘
       ▲                                                                 └────────────────┘              │
       │                                                                                                 │
       │              ┌────────────────────┐                                                             │
       └──────────────│ Stage 5: Refiner   │◀────────────────────────────────────────────────────────────┘
                      │ (Chat Refinement)  │
                      └────────────────────┘
```

### 2.2 Pipeline Stages

| Stage | Name | Technology | Purpose |
|-------|------|-----------|---------|
| 1 | **Prompt Structurer** | Gemini Flash / GPT-4o | Converts natural-language input into a structured JSON spec (drawing prompt, labels, style, description) |
| 2 | **Image Generator** | Gemini Imagen / DALL-E 3 | Generates a raster PNG image from the structured drawing prompt |
| 3 | **Vectorizer** | PyTorch-SVGRender (LIVE) | Converts raster PNG → scalable SVG using differentiable rendering (Bézier path optimization) |
| 4 | **Annotator** | Gemini Flash / GPT-4o | Uses LLM to estimate label positions, then programmatically injects `<text>` elements and leader lines into the SVG |
| 5 | **Refiner** | Gemini Flash / GPT-4o | Handles iterative chat refinement — classifies requests (label edits, color changes, style changes, or regeneration) and applies SVG modifications or re-triggers the pipeline |

### 2.3 Technology Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Single-page HTML/CSS/JS (no framework), Jinja2 templates |
| **Backend** | Python 3.10, FastAPI, Uvicorn |
| **LLM APIs** | Google Gemini (`gemini-2.0-flash`), OpenAI (`gpt-4o`) |
| **Image Generation APIs** | Google Gemini Imagen (`gemini-2.0-flash-exp-image-generation`), OpenAI DALL-E 3 |
| **Vectorization** | PyTorch-SVGRender (LIVE pipeline) with DiffVG differentiable renderer |
| **Deep Learning** | PyTorch 2.6.0 + CUDA 12.4 |
| **SVG Manipulation** | Python `xml.etree.ElementTree` |
| **Configuration** | `.env` files, `python-dotenv` |
| **Retry/Resilience** | Custom exponential-backoff decorator for API rate limits |

---

## 3. Detailed Stage Descriptions

### 3.1 Stage 1: Prompt Structurer (`pipeline/prompt_structurer.py`)

**Input:** Raw user message (e.g., *"draw a human hand anatomy"*)

**Output:** Structured JSON:
```json
{
  "drawing_prompt": "Scientific medical textbook illustration of human hand skeletal anatomy... Absolutely no text, no labels...",
  "labels": ["Phalanges", "Metacarpals", "Carpals"],
  "style": "colored_diagram",
  "description": "Labeled diagram of the human hand showing the skeletal anatomy."
}
```

**Key Design Decisions:**
- The system prompt explicitly instructs the LLM to generate a `drawing_prompt` that **forbids any text/labels** in the image. This is critical because image generation models produce garbled, unreadable text. Labels are added programmatically in Stage 4 instead.
- Supports both Gemini and OpenAI as LLM providers, selectable from the UI.

### 3.2 Stage 2: Image Generator (`pipeline/image_generator.py`)

**Input:** Structured spec from Stage 1

**Output:** Raster PNG image saved to `static/generated/<session_id>/raster.png`

**Providers:**
- **Gemini Imagen** (`gemini-2.0-flash-exp-image-generation`): Uses the new `google-genai` SDK with `response_modalities=["image", "text"]` to request image output.
- **OpenAI DALL-E 3**: Generates 1024×1024 images, returned as base64-encoded PNG.

**Resilience:** Both providers are wrapped with a `@retry_on_rate_limit` decorator that handles HTTP 429 / `RESOURCE_EXHAUSTED` errors with exponential backoff (default: 3 retries, 10–15s initial wait).

### 3.3 Stage 3: Vectorizer (`pipeline/vectorizer.py`)

**Input:** Raster PNG from Stage 2

**Output:** Scalable SVG file at `static/generated/<session_id>/vector.svg`

**How it works:**
1. Invokes `PyTorch-SVGRender/svg_render.py` as a **subprocess** (to avoid Hydra configuration conflicts with the main app).
2. Uses the **LIVE** (Layer-wise Image Vectorization) method, a research technique that optimizes Bézier curve paths to approximate the raster image using differentiable rendering via DiffVG.
3. The process is GPU-accelerated (CUDA), running path optimization over multiple iterations.

**Configuration parameters:**
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `NUM_PATHS` | 16 (test) / 128 (quality) | Number of Bézier paths to optimize |
| `NUM_ITER` | 100 (test) / 500 (quality) | Optimization iterations per stage |
| `IMAGE_SIZE` | 480 | Input image resize target |
| `NUM_STAGES` | 1 | Number of path-addition stages |

**⚠️ Known Issue — Performance Bottleneck:**
On the test GPU (GTX 1050 Ti, 4GB VRAM), vectorization with production settings (128 paths, 500 iterations) exceeded the 10-minute timeout (~30+ minutes). Even with reduced settings (16 paths, 100 iterations), it takes several minutes. This is the **primary limitation** of the current approach. See Section 6 for alternatives.

### 3.4 Stage 4: Annotator (`pipeline/annotator.py`)

**Input:** SVG from Stage 3 + label list from Stage 1

**Output:** Annotated SVG with `<text>` elements and dashed leader lines

**How it works:**
1. Parses the SVG to extract canvas dimensions (viewBox or width/height).
2. Sends the label list, canvas dimensions, and figure description to the LLM.
3. The LLM returns estimated `(x, y)` coordinates for each label, plus `(leader_x, leader_y)` endpoints for leader lines.
4. Programmatically injects into the SVG DOM:
   - Dashed leader lines (`<line>` with `stroke-dasharray`)
   - Small dot markers (`<circle>`) at pointed locations
   - Semi-transparent background rectangles for text readability
   - Bold text elements (`<text>`) with the label content

### 3.5 Stage 5: Refiner (`pipeline/refiner.py`)

**Input:** User refinement message + current SVG

**Output:** Modified SVG or instruction to regenerate

**Refinement categories (LLM-classified):**
| Category | Action |
|----------|--------|
| `label_edit` | Change label text, position, size, add/remove labels |
| `color_edit` | Change colors of diagram parts |
| `style_edit` | Change stroke widths, fonts, opacity |
| `add_element` | Add arrows, boxes, highlights |
| `regenerate` | Re-run the full pipeline with modified prompt |

For non-regeneration requests, the refiner directly manipulates the SVG DOM (modifying attributes, adding/removing elements). For regeneration requests, it builds a modified spec and returns it to the orchestrator to re-run Stages 1–4.

---

## 4. Application Layer

### 4.1 Backend (`app.py`)

- **Framework:** FastAPI (async)
- **Endpoints:**
  | Endpoint | Method | Purpose |
  |----------|--------|---------|
  | `/` | GET | Serves the chat UI (HTML) |
  | `/api/generate` | POST | Runs the full 5-stage pipeline |
  | `/api/refine` | POST | Handles refinement requests |
  | `/api/sessions` | GET | Lists active sessions |
  | `/generated/<path>` | GET | Serves generated static files (images, SVGs) |

- **Session Management:** In-memory dict storing spec, refiner state, and file paths per session ID.
- **Error Handling:** Vectorizer errors are caught gracefully — if vectorization fails or times out, the raster image is still returned to the user.

### 4.2 Frontend (`templates/index.html`)

- **Split-panel layout:** Chat panel (left) + Preview panel (right)
- **Provider selectors:** Dropdowns to choose LLM (Gemini / OpenAI) and Image provider (Gemini Imagen / DALL-E 3)
- **Toggle options:** Skip vectorization, Skip annotation (checkboxes)
- **Preview tabs:** Raster PNG view and Vector SVG view with download buttons
- **Chat refinement:** After initial generation, subsequent messages are routed to the `/api/refine` endpoint
- **Session management:** Type `/new` to start a fresh figure

---

## 5. Infrastructure & Dependencies

### 5.1 Environment

| Component | Version/Detail |
|-----------|---------------|
| OS | Windows 10/11 |
| Python | 3.10 (conda env: `svgrender`) |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| GPU | NVIDIA GTX 1050 Ti (4GB VRAM) |

### 5.2 Key Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` + `uvicorn` | Web server |
| `google-genai` | Gemini API (new SDK) |
| `openai` | OpenAI API |
| `Pillow` | Image processing |
| `python-dotenv` | Environment variable loading |
| `jinja2` | HTML templating |
| `torch` + `torchvision` | Deep learning (for vectorizer) |
| `pydiffvg` | Differentiable SVG renderer (compiled from source) |
| `opencv-python-headless` | Image utilities for LIVE pipeline |
| `scikit-fmm` | Fast marching method (LIVE distance weighting) |
| `clip` (OpenAI) | CLIP loss for perceptual optimization |
| `scipy` | Scientific computing utilities |
| `hydra-core` + `omegaconf` | Configuration management (PyTorch-SVGRender) |

### 5.3 DiffVG Compilation (Windows)

DiffVG required **4 platform-specific patches** to compile on Windows:

1. **LIBDIR `None` fix** — `setup.py` crashed when `torch.utils.cpp_extension.COMMON_MSVC_FLAGS` returned `None` for lib directory
2. **`log2` MSVC fix** — MSVC doesn't support `log2()` on integer types; cast to `double`
3. **CMake Python path fix** — CMake couldn't find the correct Python executable in the conda env
4. **`.pyd` rename** — Windows uses `.pyd` not `.so` for Python extensions; added post-build rename

### 5.4 Configuration (`config.py` + `.env`)

```
.env keys:
  GOOGLE_GENERATIVE_AI_API_KEY  — Gemini API key
  OPENAI_API_KEY                — OpenAI API key
  ANTHROPIC_API_KEY             — (reserved, not yet used)
  IMAGE_GEN_PROVIDER            — "gemini" or "openai" (default)
  LLM_PROVIDER                  — "gemini" or "openai" (default)
```

---

## 6. Known Issues & Limitations

### 6.1 Vectorization Performance (Critical)

**Problem:** The LIVE vectorization pipeline is extremely slow on consumer GPUs. With 128 paths and 500 iterations on a GTX 1050 Ti, it takes **30+ minutes** — far exceeding the 10-minute subprocess timeout.

**Impact:** The vectorization step (Stage 3) times out, so the user only receives the raster PNG. The SVG output and annotation stages are skipped.

**Mitigations attempted:**
- Reduced parameters (16 paths, 100 iterations) — produces output but with low quality
- Set `num_stages=1` to minimize path-addition rounds

**Potential solutions:**
| Approach | Tradeoff |
|----------|----------|
| Use a cloud GPU (A100/V100) | Fast (~2-5 min for 128 paths), but adds cloud cost |
| Use `vtracer` or `potrace` libraries | Near-instant (~seconds), but simpler vectorization (not neural-optimized) |
| Pre-generate SVGs with a queued worker | Decouple vectorization from request/response cycle |
| Skip vectorization entirely | Serve raster PNG only; add labels directly on the PNG using Pillow |
| Use DALL-E 3 + post-process | DALL-E 3 produces high-quality images; overlay labels on raster instead |

### 6.2 Garbled Text on Generated Images

**Problem:** Despite aggressive prompt engineering to prohibit text, image generation models (especially DALL-E 3) sometimes still render garbled, unreadable text on the images. This creates visual artifacts in the final output.

**Mitigations attempted:**
- System prompt explicitly ends every drawing prompt with a long "absolutely no text" instruction
- Multiple reinforcement phrases in the system prompt

**Potential solutions:**
- Post-process images with inpainting to remove text regions before vectorization
- Use image segmentation to detect and mask text-like areas
- Fine-tune prompts per-model (DALL-E 3 vs Gemini Imagen respond differently)

### 6.3 Label Positioning Accuracy

The LLM estimates label positions based on textual description alone (it doesn't see the actual image). This means label placement can be inaccurate — a label for "Metacarpals" might not point exactly to the metacarpal region.

**Potential solutions:**
- Use a vision model (GPT-4V, Gemini Vision) to analyze the generated image and determine precise label positions
- Use object detection / segmentation to identify anatomical regions

### 6.4 Single-Machine Architecture

The current system runs all stages (LLM calls, image generation, GPU-heavy vectorization) on a single machine synchronously. This blocks the web server during long operations.

**Potential solutions:**
- Task queue (Celery/Redis) for async processing
- WebSocket-based progress updates
- Microservice architecture separating GPU workloads

---

## 7. Project Structure

```
medical-figure-gen/
├── .env                    # API keys (not committed)
├── .env.example            # Template for collaborators
├── .gitignore
├── config.py               # Central configuration
├── app.py                  # FastAPI server (orchestrator)
├── requirements.txt        # Python dependencies
├── pyrightconfig.json      # Type-checking config
├── report.md               # This document
├── pipeline/
│   ├── __init__.py
│   ├── prompt_structurer.py   # Stage 1: LLM prompt → structured spec
│   ├── image_generator.py     # Stage 2: Spec → raster PNG
│   ├── vectorizer.py          # Stage 3: PNG → SVG (LIVE/DiffVG)
│   ├── annotator.py           # Stage 4: SVG + labels → annotated SVG
│   ├── refiner.py             # Stage 5: Chat refinement
│   └── utils.py               # Shared utilities (retry decorator)
├── templates/
│   └── index.html             # Chat UI (single-page app)
├── static/
│   └── generated/             # Output directory per session
│       └── <session_id>/
│           ├── raster.png
│           ├── vector.svg
│           ├── annotated.svg
│           └── vectorize/     # LIVE intermediate outputs
└── README.md
```

**External dependency:**
```
../PyTorch-SVGRender/          # Research codebase (subprocessed for vectorization)
```

---

## 8. What Works Today

| Feature | Status | Notes |
|---------|--------|-------|
| Chat UI with provider selection | ✅ Working | Gemini / OpenAI selectable from dropdown |
| Stage 1: Prompt structuring | ✅ Working | Both Gemini and OpenAI providers tested |
| Stage 2: Image generation | ✅ Working | Both Gemini Imagen and DALL-E 3 produce output |
| Stage 3: Vectorization | ⚠️ Partial | Works but exceeds timeout on consumer GPU (30+ min); succeeds with very low quality settings |
| Stage 4: Annotation | ✅ Working (when SVG available) | Labels + leader lines injected correctly |
| Stage 5: Chat refinement | ✅ Working | Label edits, color changes, regeneration all functional |
| Rate limit handling | ✅ Working | Exponential backoff on 429 errors |
| Error resilience | ✅ Working | Vectorizer timeout caught gracefully; raster still delivered |
| Skip vectorization option | ✅ Working | Checkbox in UI to bypass slow Stage 3 |

---

## 9. Recommendations for Next Steps

### Short-term (make it usable now)
1. **Replace LIVE vectorizer with `vtracer`** — Python binding for a fast Rust-based vectorizer; produces SVGs in seconds instead of minutes.
2. **Add vision-based label placement** — Send the generated raster to GPT-4V / Gemini Vision to get accurate label coordinates.
3. **Add progress updates via WebSocket** — Show real-time "Stage 2 complete, starting vectorization..." in the chat.

### Medium-term (production readiness)
4. **Move to task queue (Celery + Redis)** — Decouple long-running stages from the web request lifecycle.
5. **Add user accounts and persistence** — Replace in-memory session store with a database.
6. **Deploy on cloud GPU** — Run vectorization on an A100/V100 for 10–50× speedup.

### Long-term (product differentiation)
7. **Fine-tune a medical illustration model** — LoRA / DreamBooth on textbook illustration datasets.
8. **Interactive SVG editor** — Allow drag-and-drop label repositioning in the browser.
9. **Multi-format export** — PDF, PowerPoint, EPS for academic publishing.

---

## 10. Summary

The Medical Figure Generator demonstrates a viable **5-stage pipeline** for producing labeled medical diagrams from text:

1. **LLM → structured spec** (works well)
2. **Image API → raster PNG** (works well)
3. **Neural vectorization → SVG** (works but too slow for interactive use on consumer hardware)
4. **LLM + programmatic → annotated SVG** (works well)
5. **Chat refinement → iterative edits** (works well)

The **primary bottleneck** is Stage 3 (vectorization). The LIVE pipeline from PyTorch-SVGRender produces beautiful neural-optimized SVGs, but its runtime on a GTX 1050 Ti (30+ minutes) makes it impractical for interactive use. Replacing it with a fast vectorizer (e.g., `vtracer`, ~2 seconds) or running on cloud GPUs would make the full pipeline viable.

Everything else — the LLM integration, image generation, annotation, chat refinement, and web UI — is functional and ready for iteration.
