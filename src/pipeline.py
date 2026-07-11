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


def run_pipeline(topic: str, avatar_photo: str = None):
    """
    Run the full pipeline: script -> voice -> visuals -> subtitles -> assembly.

    Args:
        topic: the video topic/subject
        avatar_photo: optional path to a photo. If provided, a lip-synced
            talking-head overlay (Wav2Lip) is generated from this photo and
            the narration audio, and shown as a small corner box in the
            final video (news-anchor style). Requires a GPU and downloads
            a ~436MB model on first use.
    """
    total_steps = 6 if avatar_photo else 5
    step = 1

    print(f"[{step}/{total_steps}] Generating script for topic: {topic}")
    script = generate_script(topic)
    print("Script generated:")
    print(script)
    step += 1

    print(f"[{step}/{total_steps}] Generating narration audio (Bark)...")
    audio_path = generate_voice(script)
    print(f"Narration saved at: {audio_path}")
    step += 1

    print(f"[{step}/{total_steps}] Generating visuals (Pexels + fallback)...")
    scenes = generate_visuals(script)
    print(f"Generated {len(scenes)} scene clips.")
    step += 1

    print(f"[{step}/{total_steps}] Generating subtitles (Whisper)...")
    subtitle_result = generate_subtitles(audio_path)
    print(f"Subtitles saved at: {subtitle_result['srt_path']}")
    step += 1

    avatar_path = None
    if avatar_photo:
        print(f"[{step}/{total_steps}] Generating avatar lip-sync (Wav2Lip)...")
        from src.avatar import generate_avatar_lipsync
        avatar_path = generate_avatar_lipsync(avatar_photo, audio_path)
        print(f"Avatar video saved at: {avatar_path}")
        step += 1

    print(f"[{step}/{total_steps}] Assembling final video...")
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
        avatar_clip_path=avatar_path,
    )
    print(f"Final video saved at: {final_path}")

    return {
        "script": script,
        "audio_path": audio_path,
        "scenes": scenes,
        "subtitles": subtitle_result,
        "avatar_path": avatar_path,
        "final_video_path": final_path,
    }


if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "A surprising historical fact"
    run_pipeline(topic)
