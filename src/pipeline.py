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


def _splice_avatar_into_one_scene(avatar_full_path, audio_path, scenes, scene_index=None):
    """
    Take the full-length avatar video (same duration as the narration) and
    cut out just the time-slice that corresponds to ONE scene, then swap
    that scene's background clip for this avatar slice.

    This makes the avatar appear naturally in exactly one scene, precisely
    synced to that portion of the narration (since it's a direct time-slice
    of a video that was generated against the full audio), instead of a
    persistent overlay across the whole video.
    """
    from moviepy.editor import VideoFileClip, AudioFileClip
    from src.assembly import _safe_write_videofile
    from configs.settings import FPS

    total_duration = AudioFileClip(audio_path).duration
    n = len(scenes)
    per_scene_duration = total_duration / max(n, 1)

    if scene_index is None:
        scene_index = n // 2  # default: a scene roughly in the middle
    scene_index = max(0, min(scene_index, n - 1))

    start = scene_index * per_scene_duration
    end = min((scene_index + 1) * per_scene_duration, total_duration)

    print(f"[avatar] Splicing avatar into scene {scene_index + 1}/{n} ({start:.1f}s-{end:.1f}s)...")
    avatar_clip = VideoFileClip(avatar_full_path)
    end = min(end, avatar_clip.duration)
    sliced = avatar_clip.subclip(start, end).without_audio()

    sliced_path = os.path.join("outputs", "clips", f"scene_{scene_index:02d}_avatar.mp4")
    _safe_write_videofile(sliced, sliced_path, fps=FPS, codec="libx264", threads=4)

    scenes[scene_index]["clip_path"] = sliced_path
    scenes[scene_index]["source"] = "avatar"
    return scenes


def run_pipeline(topic: str, avatar_photo: str = None, avatar_scene_index: int = None):
    """
    Run the full pipeline: script -> voice -> visuals -> subtitles -> assembly.

    Args:
        topic: the video topic/subject
        avatar_photo: optional path to a photo. If provided, a natural
            talking-head video (head motion + lip-sync, via SadTalker) is
            generated from this photo and the full narration audio. A
            time-sliced portion of it then replaces ONE scene's background
            clip in the final video (rather than a persistent overlay).
            Requires a GPU and downloads ~4GB of models on first use.
        avatar_scene_index: which scene (0-indexed) should show the avatar.
            Defaults to a scene roughly in the middle of the video.
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

    if avatar_photo:
        print(f"[{step}/{total_steps}] Generating avatar scene (SadTalker)...")
        from src.avatar import generate_avatar_video
        avatar_full_path = generate_avatar_video(avatar_photo, audio_path)
        scenes = _splice_avatar_into_one_scene(avatar_full_path, audio_path, scenes, avatar_scene_index)
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
