"""
Configuration loader for medical-figure-gen.
Reads API keys from .env and provides model/path settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.resolve()
load_dotenv(PROJECT_ROOT / ".env")

# ── API Keys ──────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Provider Selection ────────────────────────────────────
# "gemini" | "openai"
IMAGE_GEN_PROVIDER = os.getenv("IMAGE_GEN_PROVIDER", "gemini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# ── Paths ─────────────────────────────────────────────────
STATIC_DIR = PROJECT_ROOT / "static"
GENERATED_DIR = STATIC_DIR / "generated"

# Ensure output dirs exist
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# ── Model Settings ────────────────────────────────────────
GEMINI_LLM_MODEL = "gemini-3-pro-preview"
GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"  # supports image output
OPENAI_IMAGE_MODEL = "dall-e-3"
OPENAI_LLM_MODEL = "gpt-4o"

# ── Label / Annotation Settings ──────────────────────────
DEFAULT_LABEL_STYLE = "boxed_text"  # plain_text | boxed_text | numbered | color_coded | minimal | textbook
AUDITOR_MAX_ITERATIONS = 2          # max auditor feedback loops (0 = skip auditing)

# ── Font Settings ─────────────────────────────────────────
# Pillow font paths (Windows default; override in .env if needed)
FONT_PATH = os.getenv("FONT_PATH", r"C:\Windows\Fonts\arial.ttf")
FONT_BOLD_PATH = os.getenv("FONT_BOLD_PATH", r"C:\Windows\Fonts\arialbd.ttf")
