"""
Subtitles Module (Whisper)
---------------------------
Transcribes the narration audio into timed subtitle segments using
OpenAI Whisper (runs locally, no API key needed).

Usage:
    from src.subtitles import generate_subtitles
    result = generate_subtitles("outputs/voice.wav")
    # result -> {"srt_path": "outputs/subtitles.srt", "segments": [...]}
"""

import os


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_srt(segments: list, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = _format_timestamp(seg["start"])
            end = _format_timestamp(seg["end"])
            text = seg["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def generate_subtitles(audio_path: str, output_dir: str = "outputs", model_size: str = "base") -> dict:
    """
    Transcribe audio into subtitle segments and write an .srt file.

    Args:
        audio_path: path to the narration .wav file
        output_dir: where to save subtitles.srt
        model_size: Whisper model size ("tiny", "base", "small", ...).
                    "base" is a good speed/accuracy tradeoff for Colab.

    Returns:
        dict with "srt_path" and "segments" (list of {start, end, text})
    """
    import whisper
    import torch

    # Same GPU as Bark (GPU 0) — SadTalker uses GPU 1 (see src/avatar.py).
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"[subtitles] Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size, device=device)

    print(f"[subtitles] Transcribing {audio_path}...")
    result = model.transcribe(audio_path)

    segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result["segments"]
    ]

    os.makedirs(output_dir, exist_ok=True)
    srt_path = os.path.join(output_dir, "subtitles.srt")
    _write_srt(segments, srt_path)

    print(f"[subtitles] Saved {len(segments)} segments to {srt_path}")
    return {"srt_path": srt_path, "segments": segments}


if __name__ == "__main__":
    import sys
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/voice.wav"
    result = generate_subtitles(audio_path)
    print(result)
