"""
Visuals Module (Pexels + Text-Card Fallback)
-----------------------------------------------
Turns a script into a list of visual "scenes" — one per narration segment
(hook, each body sentence, cta) — each backed by a video clip.

Strategy:
  1. For each segment, search Pexels Videos for a relevant stock clip.
  2. If Pexels finds nothing suitable, generate a simple text-card video
     (colored background + the sentence text) as a fallback using PIL +
     moviepy — no heavy dependencies required (Manim needs system-level
     libraries that are unreliable to install on Colab, so this is used
     instead for the fallback case).

Usage:
    from src.visuals import generate_visuals
    scenes = generate_visuals(script, output_dir="outputs/clips")
    # scenes -> [{"text": ..., "clip_path": ..., "source": "pexels"|"fallback"}, ...]
"""

import os
import re
import requests

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import PEXELS_API_KEY, VIDEO_WIDTH, VIDEO_HEIGHT


PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"

# Common words to strip out when building a search query from a sentence,
# so Pexels gets meaningful nouns/keywords rather than filler words.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "of",
    "in", "on", "at", "to", "for", "with", "did", "you", "know", "that",
    "this", "it", "its", "their", "his", "her", "he", "she", "they", "we",
    "i", "as", "by", "from", "has", "have", "had", "be", "been", "being",
}


def _sentence_to_query(sentence: str, max_words: int = 6) -> str:
    """Extract a short, keyword-focused search query from a sentence."""
    words = re.findall(r"[A-Za-z']+", sentence.lower())
    keywords = [w for w in words if w not in _STOPWORDS]
    if not keywords:
        keywords = words
    return " ".join(keywords[:max_words]) or sentence[:40]


def _search_pexels_video(query: str) -> str | None:
    """
    Search Pexels for a video matching the query.
    Returns a direct downloadable video file URL, or None if nothing found.
    """
    if not PEXELS_API_KEY:
        return None

    try:
        response = requests.get(
            PEXELS_SEARCH_URL,
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": query, "per_page": 3, "orientation": "landscape"},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"  [visuals] Pexels search failed for '{query}': {e}")
        return None

    videos = data.get("videos", [])
    if not videos:
        return None

    # Pick the first video's best-quality file close to our target resolution
    video_files = videos[0].get("video_files", [])
    if not video_files:
        return None

    # Prefer HD (~1920x1080), fall back to the largest available
    hd_files = [f for f in video_files if f.get("width", 0) >= 1280]
    chosen = (hd_files or video_files)
    chosen.sort(key=lambda f: f.get("width", 0), reverse=True)
    return chosen[0]["link"]


def _download_video(url: str, output_path: str) -> str:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    return output_path


def _make_fallback_clip(text: str, output_path: str, duration: float = 4.0) -> str:
    """
    Generate a simple text-card video clip as a fallback when no stock
    footage is found. Dark background + centered wrapped text.
    """
    from PIL import Image, ImageDraw, ImageFont
    from moviepy.editor import ImageClip
    import textwrap

    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60
        )
    except Exception:
        font = ImageFont.load_default()

    wrapped = textwrap.fill(text, width=30)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    position = ((VIDEO_WIDTH - text_w) / 2, (VIDEO_HEIGHT - text_h) / 2)

    draw.multiline_text(
        position, wrapped, font=font, fill=(255, 255, 255), align="center"
    )

    frame_path = output_path.replace(".mp4", "_frame.png")
    img.save(frame_path)

    clip = ImageClip(frame_path).set_duration(duration)
    clip.write_videofile(output_path, fps=24, codec="libx264", audio=False, logger=None)

    os.remove(frame_path)
    return output_path


def generate_visuals(script: dict, output_dir: str = "outputs/clips") -> list:
    """
    Generate one video clip per narration segment in the script.

    Args:
        script: dict with keys hook, body (list), cta
        output_dir: folder to save downloaded/generated clips

    Returns:
        List of dicts: [{"text": str, "clip_path": str, "source": "pexels"|"fallback"}, ...]
        in narration order.
    """
    os.makedirs(output_dir, exist_ok=True)

    segments = [script.get("hook", "")]
    segments.extend(script.get("body", []))
    if script.get("cta"):
        segments.append(script["cta"])
    segments = [s.strip() for s in segments if s.strip()]

    scenes = []

    for i, sentence in enumerate(segments):
        query = _sentence_to_query(sentence)
        clip_path = os.path.join(output_dir, f"scene_{i:02d}.mp4")

        print(f"[visuals] Scene {i + 1}/{len(segments)} — query: '{query}'")
        video_url = _search_pexels_video(query)

        if video_url:
            try:
                _download_video(video_url, clip_path)
                scenes.append({"text": sentence, "clip_path": clip_path, "source": "pexels"})
                print(f"  -> downloaded from Pexels: {clip_path}")
                continue
            except Exception as e:
                print(f"  -> Pexels download failed, using fallback: {e}")

        _make_fallback_clip(sentence, clip_path)
        scenes.append({"text": sentence, "clip_path": clip_path, "source": "fallback"})
        print(f"  -> generated fallback text-card: {clip_path}")

    return scenes


if __name__ == "__main__":
    test_script = {
        "hook": "Did you know ancient Rome had a working vending machine?",
        "body": [
            "It dispensed holy water when you dropped in a coin.",
            "The design dates back over two thousand years.",
        ],
        "cta": "Follow for more surprising history facts.",
    }
    result = generate_visuals(test_script)
    for r in result:
        print(r)
