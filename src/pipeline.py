"""
Pipeline Orchestrator
---------------------
Runs all stages in order: script -> voice -> visuals -> subtitles -> assembly

STATUS: Full pipeline wired up. Research stage is optional and not included
by default.
"""

import os
import glob

from src.script_generation import generate_script
from src.voice_generation import generate_voice
from src.visuals import generate_visuals
from src.subtitles import generate_subtitles
from src.assembly import assemble_video


def _find_music_file():
    """Return the first music file found in assets/music/, if any."""
    for ext in ("*.mp3", "*.wav", "*.m4a"):
        matches = glob.glob(os.path.join("assets", "music", ext))
        if matches:
            return matches[0]
    return None


def run_pipeline(topic: str):
    print(f"[1/5] Generating script for topic: {topic}")
    script = generate_script(topic)
    print("Script generated:")
    print(script)

    print("[2/5] Generating narration audio (Bark)...")
    audio_path = generate_voice(script)
    print(f"Narration saved at: {audio_path}")

    print("[3/5] Generating visuals (Pexels + fallback)...")
    scenes = generate_visuals(script)
    print(f"Generated {len(scenes)} scene clips.")

    print("[4/5] Generating subtitles (Whisper)...")
    subtitle_result = generate_subtitles(audio_path)
    print(f"Subtitles saved at: {subtitle_result['srt_path']}")

    print("[5/5] Assembling final video...")
    music_path = _find_music_file()
    if music_path:
        print(f"Using background music: {music_path}")
    else:
        print("No background music found in assets/music/ — skipping music.")

    final_path = assemble_video(
        scenes=scenes,
        audio_path=audio_path,
        subtitle_segments=subtitle_result["segments"],
        music_path=music_path,
    )
    print(f"Final video saved at: {final_path}")

    return {
        "script": script,
        "audio_path": audio_path,
        "scenes": scenes,
        "subtitles": subtitle_result,
        "final_video_path": final_path,
    }


if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "A surprising historical fact"
    run_pipeline(topic)
