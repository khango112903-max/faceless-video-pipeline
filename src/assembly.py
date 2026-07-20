"""
Assembly Module (moviepy + PIL)
---------------------------------
Combines everything into the final video:
  - Scene video clips (from src/visuals.py), stretched/looped to fill the
    total narration length
  - Narration audio (from src/voice_generation.py)
  - Burned-in subtitles (from src/subtitles.py), rendered as transparent
    PNG overlays (no ImageMagick dependency needed)
  - Optional background music, mixed in quietly under the narration

Usage:
    from src.assembly import assemble_video
    output_path = assemble_video(scenes, audio_path, subtitle_segments)
"""

import os
import math
import textwrap

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import VIDEO_WIDTH, VIDEO_HEIGHT, FPS

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_SUBTITLE_BAND_HEIGHT = 280  # height of the transparent strip subtitles sit in


def _fit_clip_to_duration(clip, duration: float):
    """Loop (if too short) or trim (if too long) a clip to an exact duration,
    and resize/crop it to the target video resolution."""
    from moviepy.editor import CompositeVideoClip

    if clip.duration < duration:
        # Loop by layering time-shifted copies in a composite (more
        # reliable in this moviepy version than concatenate_videoclips —
        # see the note in assemble_video() for why we avoid it).
        loops_needed = math.ceil(duration / clip.duration)
        loop_layers = []
        t = 0.0
        for _ in range(loops_needed):
            loop_layers.append(clip.set_start(t))
            t += clip.duration
        clip = CompositeVideoClip(loop_layers, size=(clip.w, clip.h))

    clip = clip.subclip(0, duration)
    clip = clip.without_audio()

    # Resize to fill target resolution, cropping any overflow (cover-fit)
    target_ratio = VIDEO_WIDTH / VIDEO_HEIGHT
    clip_ratio = clip.w / clip.h

    if clip_ratio > target_ratio:
        clip = clip.resize(height=VIDEO_HEIGHT)
        clip = clip.crop(x_center=clip.w / 2, width=VIDEO_WIDTH)
    else:
        clip = clip.resize(width=VIDEO_WIDTH)
        clip = clip.crop(y_center=clip.h / 2, height=VIDEO_HEIGHT)

    return clip


def _make_subtitle_png(text: str, output_path: str):
    """Render a subtitle line as a transparent PNG with a black outline
    (for readability over any background), positioned for a bottom overlay."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGBA", (VIDEO_WIDTH, _SUBTITLE_BAND_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(_FONT_PATH, 64)
    except Exception:
        font = ImageFont.load_default()

    wrapped = textwrap.fill(text, width=28)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (VIDEO_WIDTH - text_w) / 2
    y = (_SUBTITLE_BAND_HEIGHT - text_h) / 2

    # Black outline for readability (draw text offset in a ring, then white on top)
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if dx or dy:
                draw.multiline_text(
                    (x + dx, y + dy), wrapped, font=font, fill=(0, 0, 0, 255), align="center"
                )
    draw.multiline_text((x, y), wrapped, font=font, fill=(255, 255, 255, 255), align="center")

    img.save(output_path)
    return output_path


def assemble_video(
    scenes: list,
    audio_path: str,
    subtitle_segments: list,
    music_path: str = None,
    avatar_clip_path: str = None,
    output_path: str = "outputs/final_video.mp4",
    tmp_dir: str = "outputs/tmp",
) -> str:
    """
    Assemble the final video from all pipeline outputs.

    Args:
        scenes: list of {"clip_path": ...} dicts from src/visuals.py
        audio_path: path to the narration .wav from src/voice_generation.py
        subtitle_segments: list of {"start", "end", "text"} from src/subtitles.py
        music_path: optional path to a background music file
        avatar_clip_path: optional path to a lip-synced talking-head video
            (from src/avatar.py) to overlay as a small corner box
        output_path: where to save the final .mp4
        tmp_dir: scratch folder for temporary subtitle images

    Returns:
        The output_path of the final assembled video.
    """
    from moviepy.editor import (
        VideoFileClip,
        AudioFileClip,
        CompositeVideoClip,
        CompositeAudioClip,
        ImageClip,
        concatenate_videoclips,
    )

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print("[assembly] Loading narration audio...")
    narration = AudioFileClip(audio_path)
    total_duration = narration.duration

    # --- 1. Build the visual track from scene clips ---
    print(f"[assembly] Building visual track from {len(scenes)} scene(s)...")
    per_scene_duration = total_duration / max(len(scenes), 1)

    # NOTE: We deliberately do NOT use concatenate_videoclips() here. In this
    # moviepy version it has proven unreliable when clips have differing
    # source fps/resolution (common with mixed Pexels + SadTalker clips) —
    # it can silently collapse the sequence down to only the first few
    # clips' worth of visible content. Instead we explicitly place each
    # fitted clip at its own start time within one CompositeVideoClip, the
    # same reliable approach already used for the subtitle overlays below.
    video_layers = []
    current_t = 0.0
    for scene in scenes:
        raw_clip = VideoFileClip(scene["clip_path"])
        fitted = _fit_clip_to_duration(raw_clip, per_scene_duration)
        fitted = fitted.set_start(current_t).set_duration(per_scene_duration)
        video_layers.append(fitted)
        current_t += per_scene_duration

    video_track = CompositeVideoClip(video_layers, size=(VIDEO_WIDTH, VIDEO_HEIGHT))
    video_track = video_track.set_duration(total_duration)

    # --- 2. Build subtitle overlays ---
    print(f"[assembly] Rendering {len(subtitle_segments)} subtitle overlay(s)...")
    subtitle_clips = []
    for i, seg in enumerate(subtitle_segments):
        png_path = os.path.join(tmp_dir, f"subtitle_{i:03d}.png")
        _make_subtitle_png(seg["text"], png_path)

        duration = max(seg["end"] - seg["start"], 0.1)
        sub_clip = (
            ImageClip(png_path)
            .set_start(seg["start"])
            .set_duration(duration)
            .set_position(("center", VIDEO_HEIGHT - _SUBTITLE_BAND_HEIGHT - 40))
        )
        subtitle_clips.append(sub_clip)

    # --- 2b. Avatar overlay (optional, lip-synced talking head, top-right corner) ---
    avatar_layer = []
    if avatar_clip_path and os.path.exists(avatar_clip_path):
        print("[assembly] Adding avatar overlay...")
        avatar_width = int(VIDEO_WIDTH * 0.28)
        avatar_clip = VideoFileClip(avatar_clip_path).without_audio()
        avatar_clip = avatar_clip.resize(width=avatar_width)

        if avatar_clip.duration < total_duration:
            loops_needed = math.ceil(total_duration / avatar_clip.duration)
            loop_layers = []
            t = 0.0
            for _ in range(loops_needed):
                loop_layers.append(avatar_clip.set_start(t))
                t += avatar_clip.duration
            avatar_clip = CompositeVideoClip(loop_layers, size=(avatar_clip.w, avatar_clip.h))
        avatar_clip = avatar_clip.subclip(0, total_duration)

        margin = 40
        avatar_clip = avatar_clip.set_position((VIDEO_WIDTH - avatar_width - margin, margin))
        avatar_layer = [avatar_clip]

    # --- 3. Composite video + avatar + subtitles ---
    final_video = CompositeVideoClip(
        [video_track] + avatar_layer + subtitle_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT)
    )
    final_video = final_video.set_duration(total_duration)
    final_video = final_video.set_fps(FPS)  # ensure fps is never None for the writer

    # --- 4. Mix audio (narration + optional background music) ---
    if music_path and os.path.exists(music_path):
        print("[assembly] Mixing in background music...")
        from moviepy.audio.fx.all import audio_loop

        music = AudioFileClip(music_path).volumex(0.12)
        if music.duration < total_duration:
            music = audio_loop(music, duration=total_duration)
        else:
            music = music.subclip(0, total_duration)
        final_audio = CompositeAudioClip([narration, music])
    else:
        final_audio = narration

    final_video = final_video.set_audio(final_audio)

    # --- 5. Export ---
    # NOTE: We bypass clip.write_videofile() here and call moviepy's internal
    # writer functions directly. Newer versions of the `decorator` package
    # break moviepy 1.0.3's fps-resolution wrapper (causes
    # "TypeError: must be real number, not NoneType"), and pip version
    # pinning isn't always reliable across Colab/Kaggle sessions. Calling
    # the internal writers directly sidesteps the broken decorator entirely.
    print(f"[assembly] Writing final video to {output_path}...")
    _safe_write_videofile(final_video, output_path, fps=FPS, codec="libx264",
                           audio_codec="aac", threads=4)

    print("[assembly] Done!")
    return output_path


def _safe_write_videofile(clip, output_path, fps, codec="libx264", audio_codec="aac",
                           threads=4, preset="veryfast"):
    """
    Write a video file bypassing moviepy's write_videofile() wrapper,
    which can break due to a decorator-library version incompatibility
    that causes fps to be silently passed as None.

    preset="veryfast" trades a little file-size efficiency for a large
    speedup vs the default "medium" preset — this matters a lot on CPU-only
    encoding, which is what these Kaggle/Colab environments use for the
    final ffmpeg write step (GPU is only used for the AI models, not encoding).
    """
    from moviepy.video.io.ffmpeg_writer import ffmpeg_write_video

    temp_audio_path = output_path.rsplit(".", 1)[0] + "_TEMP_audio.m4a"

    if clip.audio is not None:
        clip.audio.write_audiofile(
            temp_audio_path, codec=audio_codec, fps=44100,
            write_logfile=False, verbose=False, logger=None,
        )
    else:
        temp_audio_path = None

    ffmpeg_write_video(
        clip, output_path, fps, codec=codec, preset=preset,
        audiofile=temp_audio_path, threads=threads, logger=None,
    )

    if temp_audio_path and os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)


if __name__ == "__main__":
    # Quick manual test (requires outputs/voice.wav and outputs/clips/* to exist)
    scenes = [{"clip_path": p} for p in
              sorted(__import__("glob").glob("outputs/clips/*.mp4"))]
    from src.subtitles import generate_subtitles

    sub_result = generate_subtitles("outputs/voice.wav")
    assemble_video(scenes, "outputs/voice.wav", sub_result["segments"])
