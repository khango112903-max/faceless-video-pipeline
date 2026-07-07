"""
Voice Generation Module (Bark TTS)
------------------------------------
Converts a script (dict from script_generation.py) into a single narrated
audio file using Bark.

Bark limitations handled here:
- Bark generates ~13-14 seconds of audio per call, so long scripts must be
  split into sentence-level chunks and generated one at a time.
- Each chunk is generated with the SAME voice preset (history_prompt) so the
  voice stays consistent across the whole video.
- A short silence is inserted between chunks for natural pacing.

Usage:
    from src.voice_generation import generate_voice
    audio_path = generate_voice(script, output_path="outputs/voice.wav")
"""

import os
import re
import numpy as np


# Bark voice presets — pick one consistent narrator voice.
# Full list: https://github.com/suno-ai/bark/tree/main/bark/assets/prompts
DEFAULT_VOICE_PRESET = "v2/en_speaker_6"  # calm, clear male voice - good for narration

_SILENCE_SECONDS = 0.35


def _split_into_chunks(text: str, max_chars: int = 200):
    """
    Split text into chunks that are safe for Bark's ~13s generation limit.
    Splits on sentence boundaries, then groups sentences up to max_chars.
    """
    sentences = re.split(r"(?<=[.!?]) +", text.strip())
    chunks = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def _script_to_full_text(script: dict) -> str:
    """Combine hook + body + cta into one narration text."""
    parts = [script.get("hook", "")]
    parts.extend(script.get("body", []))
    if script.get("cta"):
        parts.append(script["cta"])
    return " ".join(p.strip() for p in parts if p.strip())


def generate_voice(
    script: dict,
    output_path: str = "outputs/voice.wav",
    voice_preset: str = DEFAULT_VOICE_PRESET,
) -> str:
    """
    Generate a single narrated audio file from a script dict.

    Args:
        script: dict with keys hook, body (list), cta (from script_generation.py)
        output_path: where to save the final .wav file
        voice_preset: Bark voice preset name for a consistent narrator voice

    Returns:
        The output_path where the audio was saved.
    """
    # --- Compatibility patch for PyTorch 2.6+ ---
    # PyTorch changed torch.load's default to weights_only=True, which breaks
    # Bark's older checkpoint format. Bark's checkpoints are from a trusted
    # source (suno-ai official releases), so we safely relax this default.
    import torch
    if not getattr(torch.load, "_bark_patched", False):
        _original_torch_load = torch.load

        def _patched_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _original_torch_load(*args, **kwargs)

        _patched_torch_load._bark_patched = True
        torch.load = _patched_torch_load

    from bark import SAMPLE_RATE, generate_audio, preload_models
    from scipy.io.wavfile import write as write_wav

    # Load Bark models into memory (first call downloads weights, ~ a few GB)
    preload_models()

    full_text = _script_to_full_text(script)
    chunks = _split_into_chunks(full_text)

    print(f"Narration split into {len(chunks)} chunk(s) for Bark generation.")

    silence = np.zeros(int(_SILENCE_SECONDS * SAMPLE_RATE), dtype=np.float32)
    audio_pieces = []

    for i, chunk in enumerate(chunks):
        print(f"  Generating chunk {i + 1}/{len(chunks)}: {chunk[:60]}...")
        audio_array = generate_audio(chunk, history_prompt=voice_preset)
        audio_pieces.append(audio_array)
        audio_pieces.append(silence)

    full_audio = np.concatenate(audio_pieces)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    write_wav(output_path, SAMPLE_RATE, full_audio)

    print(f"Voice saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Quick manual test with a fake script
    test_script = {
        "hook": "Did you know ancient Rome had a working vending machine?",
        "body": [
            "It dispensed holy water when you dropped in a coin.",
            "The design dates back over two thousand years.",
        ],
        "cta": "Follow for more surprising history facts.",
    }
    generate_voice(test_script, output_path="outputs/test_voice.wav")
