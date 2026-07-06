"""
Script Generation Module
-------------------------
Generates a structured video script (hook + body + CTA) from a topic.

Supports two providers, switchable via configs/settings.py -> SCRIPT_PROVIDER:
  - "gemini" (active, ready to use)
  - "kimi"   (implemented, needs KIMI_API_KEY to activate)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import GEMINI_API_KEY, KIMI_API_KEY, SCRIPT_PROVIDER


SCRIPT_PROMPT_TEMPLATE = """You are a scriptwriter for a faceless YouTube channel about facts, motivation, and history.

Write a video script for this topic: "{topic}"

Structure it as JSON with these exact keys:
{{
  "title": "catchy YouTube title (under 70 chars)",
  "hook": "1-2 punchy sentences to grab attention in the first 3 seconds",
  "body": ["sentence 1", "sentence 2", "..."],
  "cta": "short call-to-action for the end (like/subscribe/follow)",
  "estimated_duration_seconds": 60
}}

Rules:
- Keep total spoken length around 45-75 seconds when read aloud.
- Simple, clear, engaging spoken language (not written/formal).
- No emojis, no markdown, only valid JSON.
- Body should be broken into short sentences (good for subtitle timing).

Return ONLY the JSON object, nothing else.
"""


def _generate_with_gemini(topic: str) -> str:
    import google.generativeai as genai

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set. Add it to .env or Colab secrets.")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")

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


def generate_script(topic: str, provider: str = None) -> dict:
    """
    Generate a structured script dict for the given topic.

    Args:
        topic: The video topic/subject.
        provider: "gemini" or "kimi". Defaults to SCRIPT_PROVIDER in settings.

    Returns:
        dict with keys: title, hook, body (list), cta, estimated_duration_seconds
    """
    import json
    import re

    provider = (provider or SCRIPT_PROVIDER).lower()

    if provider == "gemini":
        raw_text = _generate_with_gemini(topic)
    elif provider == "kimi":
        raw_text = _generate_with_kimi(topic)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'gemini' or 'kimi'.")

    # Clean up in case the model wraps JSON in markdown fences
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    cleaned = cleaned.strip("`\n ")

    try:
        script = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON. Raw output:\n{raw_text}"
        ) from e

    return script


if __name__ == "__main__":
    # Quick manual test:
    # python src/script_generation.py "The mystery of the Antikythera mechanism"
    topic = sys.argv[1] if len(sys.argv) > 1 else "A surprising fact about ancient Rome"
    result = generate_script(topic)
    print(result)
