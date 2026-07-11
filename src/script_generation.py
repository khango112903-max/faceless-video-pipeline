"""
Script Generation Module
-------------------------
Generates a structured video script (hook + body + CTA) from a topic.

Supports two providers, with AUTOMATIC FALLBACK:
  - Preferred provider is set via configs/settings.py -> SCRIPT_PROVIDER
    (defaults to "gemini")
  - If the preferred provider fails for any reason (quota exceeded,
    missing key, network error), it automatically tries the OTHER
    provider so the pipeline doesn't break.
  - Requires at least one of GEMINI_API_KEY / KIMI_API_KEY to be set.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import GEMINI_API_KEY, KIMI_API_KEY, SCRIPT_PROVIDER


SCRIPT_PROMPT_TEMPLATE = """You are a scriptwriter and visual director for a faceless YouTube channel about facts, motivation, and history.

Write a video script for this topic: "{topic}"

Structure it as JSON with these exact keys:
{{
  "title": "catchy YouTube title (under 70 chars)",
  "hook": "1-2 punchy sentences to grab attention in the first 3 seconds",
  "body": ["sentence 1", "sentence 2", "..."],
  "cta": "short call-to-action for the end (like/subscribe/follow)",
  "estimated_duration_seconds": 60,
  "visual_queries": ["query for hook", "query for body sentence 1", "query for body sentence 2", "...", "query for cta"]
}}

Rules for the script:
- Keep total spoken length around 45-75 seconds when read aloud.
- Simple, clear, engaging spoken language (not written/formal).
- Body should be broken into short sentences (good for subtitle timing).

Rules for "visual_queries" (VERY IMPORTANT):
- There must be EXACTLY one visual_queries entry per narration segment, in the
  same order as: [hook, body[0], body[1], ..., body[last], cta].
- Each entry is a 3-5 word STOCK FOOTAGE search query for a real, professional
  stock video site (like Pexels), describing a concrete, specific, filmable
  visual that matches what that sentence is about.
- Be concrete and specific: use real places, objects, materials, actions, or
  historical subjects mentioned or implied by the sentence (e.g. "Roman
  aqueduct stone arches", "volcanic ash texture close up", "ancient
  Colosseum ruins aerial"), NOT vague or abstract terms (avoid words like
  "history", "concept", "idea", "surprising").
- Do NOT use generic queries like "person talking" or "man face" unless the
  sentence is specifically about a named historical figure.
- Always anchor each query to the overall topic ("{topic}") even if the
  sentence itself is a short transition (e.g. for a transition sentence,
  still describe a relevant establishing visual for the topic).

No emojis, no markdown, only valid JSON. Return ONLY the JSON object, nothing else.
"""


def _generate_with_gemini(topic: str) -> str:
    import google.generativeai as genai

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set. Add it to .env or Colab secrets.")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-3.5-flash")

    prompt = SCRIPT_PROMPT_TEMPLATE.format(topic=topic)
    response = model.generate_content(prompt)
    return response.text


def _generate_with_kimi(topic: str) -> str:
    import requests

    if not KIMI_API_KEY:
        raise ValueError("KIMI_API_KEY not set. Add it to .env or Colab secrets.")

    prompt = SCRIPT_PROMPT_TEMPLATE.format(topic=topic)

    response = requests.post(
        "https://api.moonshot.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {KIMI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "moonshot-v1-8k",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _parse_script_json(raw_text: str) -> dict:
    """Clean and parse the model's raw text output into a script dict."""
    import json
    import re

    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    cleaned = cleaned.strip("`\n ")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON. Raw output:\n{raw_text}"
        ) from e


def generate_script(topic: str, provider: str = None) -> dict:
    """
    Generate a structured script dict for the given topic.

    Tries the preferred provider first. If it fails for ANY reason
    (quota exceeded, missing key, network error, etc.), it automatically
    falls back to the other provider so the pipeline doesn't break.

    Args:
        topic: The video topic/subject.
        provider: "gemini" or "kimi". Defaults to SCRIPT_PROVIDER in settings.

    Returns:
        dict with keys: title, hook, body (list), cta, estimated_duration_seconds
    """
    preferred = (provider or SCRIPT_PROVIDER).lower()
    fallback = "kimi" if preferred == "gemini" else "gemini"

    generators = {
        "gemini": _generate_with_gemini,
        "kimi": _generate_with_kimi,
    }

    errors = {}

    for name in (preferred, fallback):
        try:
            print(f"[script_generation] Trying provider: {name}...")
            raw_text = generators[name](topic)
            script = _parse_script_json(raw_text)
            print(f"[script_generation] Success with provider: {name}")
            return script
        except Exception as e:
            print(f"[script_generation] Provider '{name}' failed: {e}")
            errors[name] = str(e)

    raise RuntimeError(
        f"Both providers failed.\nGemini error: {errors.get('gemini')}\n"
        f"Kimi error: {errors.get('kimi')}"
    )


if __name__ == "__main__":
    # Quick manual test:
    # python src/script_generation.py "The mystery of the Antikythera mechanism"
    topic = sys.argv[1] if len(sys.argv) > 1 else "A surprising fact about ancient Rome"
    result = generate_script(topic)
    print(result)
