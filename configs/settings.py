"""
Central settings for the pipeline.
Reads from environment variables (.env locally, or Colab userdata/secrets).
"""

import os

def get_env(key: str, default: str = None):
    """
    Get a config value. Works both locally (.env via python-dotenv)
    and in Google Colab (via userdata.get).
    """
    # Try Colab secrets first
    try:
        from google.colab import userdata
        val = userdata.get(key)
        if val:
            return val
    except Exception:
        pass

    # Fallback to regular environment variables (.env)
    return os.environ.get(key, default)


# --- API Keys ---
GEMINI_API_KEY = get_env("GEMINI_API_KEY")
KIMI_API_KEY = get_env("KIMI_API_KEY")
PERPLEXITY_API_KEY = get_env("PERPLEXITY_API_KEY")

# --- Script generation provider: "gemini" or "kimi" ---
SCRIPT_PROVIDER = get_env("SCRIPT_PROVIDER", "gemini")

# --- Video settings ---
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
FPS = 30

# --- Paths ---
OUTPUT_DIR = "outputs"
MUSIC_DIR = "assets/music"
FONTS_DIR = "assets/fonts"
