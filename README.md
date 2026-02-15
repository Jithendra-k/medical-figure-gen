# Medical Figure Generator

A chat-based tool for generating labeled medical and scientific textbook-style figures. It combines LLM-driven prompt structuring, cloud image generation APIs, local SVG vectorization (via [PyTorch-SVGRender](https://github.com/ximinng/PyTorch-SVGRender)), and LLM-powered annotation — all orchestrated through a FastAPI web interface.

---

## Architecture

The system is a **5-stage pipeline** exposed through a chat UI:

```
User Message
    │
    ▼
┌──────────────────────┐
│ 1. Prompt Structurer │  LLM decomposes request into drawing prompt + label list
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 2. Image Generator   │  Gemini Imagen or DALL-E 3 produces a raster PNG
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 3. Vectorizer        │  PyTorch-SVGRender LIVE converts PNG → SVG (runs locally on GPU)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 4. Annotator         │  LLM places text labels + leader lines onto the SVG
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 5. Refiner           │  Chat loop: user asks for changes → SVG edits or re-generation
└──────────────────────┘
```

### Project Structure

```
medical-figure-gen/
├── .env                          # API keys (not committed)
├── .gitignore
├── config.py                     # Central config: paths, models, settings
├── app.py                        # FastAPI server — orchestrates pipeline + serves UI
├── requirements.txt
├── pipeline/
│   ├── __init__.py
│   ├── prompt_structurer.py      # Stage 1 — LLM prompt decomposition
│   ├── image_generator.py        # Stage 2 — Cloud image generation
│   ├── vectorizer.py             # Stage 3 — Raster→SVG via PyTorch-SVGRender
│   ├── annotator.py              # Stage 4 — LLM label placement
│   └── refiner.py                # Stage 5 — Iterative refinement
├── templates/
│   └── index.html                # Chat UI (split panel: chat + preview)
└── static/
    └── generated/                # Runtime output (per-session folders)
```

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10 (conda env recommended) |
| **CUDA GPU** | Any NVIDIA GPU with ≥2 GB VRAM (vectorization only) |
| **PyTorch-SVGRender** | Cloned and working as a sibling directory (see below) |
| **DiffVG** | Compiled from source inside PyTorch-SVGRender (see below) |
| **API keys** | At least one of: Google Gemini, OpenAI |

### Expected directory layout

```
svg-render/                       # parent folder
├── PyTorch-SVGRender/            # cloned repo with DiffVG compiled
│   ├── svg_render.py
│   ├── diffvg/                   # DiffVG source (compiled)
│   └── ...
└── medical-figure-gen/           # this project
    └── ...
```

`config.py` resolves `SVGRENDER_ROOT` as `../PyTorch-SVGRender` relative to this project.

---

## Setup

### 1. Clone & enter the project

```bash
git clone <your-repo-url> medical-figure-gen
cd medical-figure-gen
```

### 2. Create conda environment (if not already done)

```bash
conda create -n svgrender python=3.10 -y
conda activate svgrender
```

### 3. Install PyTorch (match your CUDA version)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install PyTorch-SVGRender + DiffVG

Follow the [PyTorch-SVGRender installation guide](https://github.com/ximinng/PyTorch-SVGRender#installation). Key steps:

```bash
cd ../PyTorch-SVGRender
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git --no-build-isolation

# DiffVG (compile from source)
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
python setup.py install
cd ../..
```

> **Windows notes:** DiffVG may need patches for MSVC. See the "Windows DiffVG Fixes" section below.

### 5. Install this project's dependencies

```bash
cd medical-figure-gen
pip install -r requirements.txt
```

### 6. Configure API keys

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
GOOGLE_GENERATIVE_AI_API_KEY=your-google-api-key
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key      # optional, reserved for future use

IMAGE_GEN_PROVIDER=gemini    # or "openai"
LLM_PROVIDER=gemini          # or "openai"
```

### 7. Update `config.py` paths (if needed)

Open [config.py](config.py) and verify:

- `PYTHON_EXE` — absolute path to the conda env's Python executable
- `SVGRENDER_ROOT` — auto-resolved to `../PyTorch-SVGRender`, adjust if your layout differs

---

## Running

```bash
# From the medical-figure-gen directory:
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in your browser.

### Quick start

1. Type: `"Draw a labeled diagram of a human neuron"`
2. Wait for the pipeline to complete (Stage 1–4)
3. The raster PNG and annotated SVG appear in the preview panel
4. Refine: `"Make the labels larger"` or `"Change label color to blue"`
5. Type `/new` to start a fresh figure

### UI Options

| Control | Effect |
|---|---|
| **Skip vectorization** checkbox | Returns only the raster PNG (fast, no GPU needed) |
| **Skip annotation** checkbox | Returns SVG without text labels |
| `/new` command | Resets the session for a new figure |

---

## Configuration Reference

All settings live in [config.py](config.py):

| Setting | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `"gemini"` | LLM for prompt structuring & annotation (`"gemini"` or `"openai"`) |
| `IMAGE_GEN_PROVIDER` | `"gemini"` | Image generation API (`"gemini"` or `"openai"`) |
| `GEMINI_LLM_MODEL` | `"gemini-2.0-flash"` | Gemini model for text tasks |
| `GEMINI_IMAGE_MODEL` | `"gemini-2.0-flash-exp-image-generation"` | Gemini model for image generation |
| `OPENAI_IMAGE_MODEL` | `"dall-e-3"` | OpenAI image model |
| `OPENAI_LLM_MODEL` | `"gpt-4o"` | OpenAI text model |
| `VECTORIZER_METHOD` | `"live"` | SVGRender method: `"live"` (recommended) or `"diffvg"` |
| `NUM_PATHS` | `128` | Number of SVG paths (more = more detail, slower) |
| `NUM_ITER` | `500` | Optimization iterations per path group |
| `IMAGE_SIZE` | `480` | Input image resize before vectorization |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the chat UI |
| `POST` | `/api/generate` | Full pipeline run. Body: `{ "message": "...", "skip_vectorize": false, "skip_annotate": false }` |
| `POST` | `/api/refine` | Refine existing figure. Body: `{ "session_id": "...", "message": "..." }` |
| `GET` | `/api/sessions` | List active sessions |
| `GET` | `/generated/{session_id}/{file}` | Serve generated files (PNG, SVG) |

### Generate response shape

```json
{
  "session_id": "a1b2c3d4",
  "steps": [
    { "stage": "prompt_structurer", "result": { "drawing_prompt": "...", "labels": [...], "style": "...", "description": "..." } },
    { "stage": "image_generator", "result": "/generated/a1b2c3d4/raster.png" },
    { "stage": "vectorizer", "result": "/generated/a1b2c3d4/vector.svg" },
    { "stage": "annotator", "result": "/generated/a1b2c3d4/annotated.svg" }
  ],
  "raster_url": "/generated/a1b2c3d4/raster.png",
  "svg_url": "/generated/a1b2c3d4/annotated.svg",
  "spec": { "drawing_prompt": "...", "labels": [...], "style": "...", "description": "..." }
}
```

---

## How Each Stage Works

### Stage 1 — Prompt Structurer (`pipeline/prompt_structurer.py`)

Sends the raw user message to an LLM with a system prompt that instructs it to return structured JSON:

```json
{
  "drawing_prompt": "Scientific medical textbook illustration of...",
  "labels": ["Dendrites", "Axon", "Soma", ...],
  "style": "colored_diagram",
  "description": "Labeled diagram of a neuron."
}
```

### Stage 2 — Image Generator (`pipeline/image_generator.py`)

Calls the configured image generation API (Gemini Imagen or DALL-E 3) with the `drawing_prompt`. Saves the result as `raster.png` in the session folder.

### Stage 3 — Vectorizer (`pipeline/vectorizer.py`)

Runs PyTorch-SVGRender's LIVE pipeline as a **subprocess**:

```
python svg_render.py x=live target=<path_to_png> x.num_paths=128 x.num_iter=500 ...
```

This avoids Hydra config conflicts and keeps the vectorizer isolated. Requires a CUDA GPU. Output: `vector.svg`.

### Stage 4 — Annotator (`pipeline/annotator.py`)

1. Parses the SVG to get canvas dimensions
2. Sends the label list + canvas size to an LLM
3. LLM returns coordinates for each label
4. Injects `<text>` elements and dashed leader lines into the SVG DOM

Output: `annotated.svg`.

### Stage 5 — Refiner (`pipeline/refiner.py`)

Classifies user refinement requests into categories:

- **label_edit** — change text, size, position → direct SVG manipulation
- **color_edit** — change colors → direct SVG manipulation
- **style_edit** — change strokes, fonts → direct SVG manipulation
- **regenerate** — re-run the full pipeline with a modified prompt
- **add_element** — add arrows, highlights → direct SVG manipulation

---

## Windows DiffVG Fixes

If compiling DiffVG on Windows with MSVC, you may need these patches (already applied in our setup):

1. **`diffvg/setup.py`** — `LIBDIR` is `None` on Windows:
   ```python
   # Add fallback when LIBDIR is None
   if libdir is None:
       libdir = os.path.join(sys.prefix, 'libs')
   ```

2. **`diffvg/diffvg.h`** — `log2` conflicts with MSVC intrinsic:
   ```cpp
   #ifndef _MSC_VER
   inline double log2(double x) { return log(x) / log(2); }
   #endif
   ```

3. **`diffvg/setup.py`** — CMake finds wrong Python version:
   ```python
   # Add explicit Python path hints to cmake_args
   cmake_args += [
       f'-DPython_EXECUTABLE={sys.executable}',
       f'-DPython_INCLUDE_DIR={sysconfig.get_path("include")}',
       f'-DPython_LIBRARY={os.path.join(libdir, "python310.lib")}',
   ]
   ```

4. **Post-install** — compiled binary may lack `.pyd` extension:
   ```bash
   # In site-packages, rename diffvg → diffvg.pyd
   ```

---

## Adding a New Provider

To add a new LLM or image generation provider:

1. Add the API key to `.env` and `config.py`
2. Add a new branch in the relevant pipeline file:
   - LLM: `prompt_structurer.py`, `annotator.py`, `refiner.py`
   - Image: `image_generator.py`
3. Add the provider name to the `IMAGE_GEN_PROVIDER` / `LLM_PROVIDER` options in `.env`

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `conda activate` doesn't work in VS Code terminal | Use the full Python path: `C:\Users\...\anaconda3\envs\svgrender\python.exe` |
| Tkinter warnings during vectorization | Set `MPLBACKEND=Agg` (already handled in `vectorizer.py`) |
| Vectorizer timeout | Increase `timeout` in `vectorizer.py` (default: 600s). Reduce `NUM_PATHS` or `NUM_ITER` for faster runs |
| "Gemini did not return an image" | The Gemini image model may not support your prompt. Try switching to `IMAGE_GEN_PROVIDER=openai` |
| DiffVG import error | Verify `diffvg.pyd` exists in `site-packages/`. Re-run `python setup.py install` in the diffvg directory |
| No SVG output found | Check the `static/generated/<session>/vectorize/` folder for Hydra output logs |

---

## License

This project uses PyTorch-SVGRender (MIT License) for vectorization. See [PyTorch-SVGRender/LICENSE](../PyTorch-SVGRender/LICENSE) for details.
