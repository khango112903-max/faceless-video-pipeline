"""
Avatar Scene Module (SadTalker)
-----------------------------------
Generates a full-length talking-head video (natural head motion + blinking
+ lip-sync, from a single still photo) matching the ENTIRE narration audio,
using SadTalker.

This full-length video is then TIME-SLICED in src/pipeline.py to produce
just one scene's worth of footage, which replaces one background clip in
the final video — so the avatar appears in exactly one scene, well
synced to that portion of the narration, rather than as a persistent
overlay across the whole video.

NOTE: SadTalker is the heaviest, most fragile stage in this pipeline.
  - Clones the SadTalker repo (first run only).
  - Downloads ~4GB of pretrained checkpoints from a Hugging Face Space
    mirror (first run only) — the original Google Drive/Baidu links are
    unreliable for automated downloads.
  - Requires a GPU. Generating the full-length talking video can take
    several minutes depending on narration length.
  - SadTalker animates head pose, blinking, and lips from a single still
    photo. It does NOT animate hands/body — that would require a full
    source video of a person, which is out of scope here.

Usage:
    from src.avatar import generate_avatar_video
    avatar_path = generate_avatar_video("my_photo.jpg", "outputs/voice.wav")
"""

import os
import sys
import glob
import shutil
import subprocess

SADTALKER_DIR = "SadTalker"
CHECKPOINTS_DIR = os.path.join(SADTALKER_DIR, "checkpoints")

_HF_SPACE_REPO = "vinthony/SadTalker"


def _ensure_sadtalker_installed():
    """Clone the SadTalker repo and download required model checkpoints, if missing."""
    if not os.path.isdir(SADTALKER_DIR):
        print("[avatar] Cloning SadTalker repo (first time only)...")
        subprocess.run(
            ["git", "clone", "https://github.com/OpenTalker/SadTalker.git", SADTALKER_DIR],
            check=True,
        )

    if not os.path.isdir(CHECKPOINTS_DIR) or not os.listdir(CHECKPOINTS_DIR):
        print("[avatar] Downloading SadTalker checkpoints (~4GB, first time only)...")
        from huggingface_hub import snapshot_download

        downloaded_dir = snapshot_download(
            repo_id=_HF_SPACE_REPO,
            repo_type="space",
            allow_patterns=["checkpoints/**"],
        )
        src_checkpoints = os.path.join(downloaded_dir, "checkpoints")
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        shutil.copytree(src_checkpoints, CHECKPOINTS_DIR, dirs_exist_ok=True)


def generate_avatar_video(
    photo_path: str,
    audio_path: str,
    output_path: str = "outputs/avatar_full.mp4",
) -> str:
    """
    Generate a full-length talking-head video (head motion + lip-sync)
    matching the ENTIRE narration audio.

    Args:
        photo_path: path to a clear, front-facing photo
        audio_path: path to the narration audio (e.g. outputs/voice.wav)
        output_path: where to save the full-length avatar video

    Returns:
        The output_path of the generated video (same duration as the audio).
    """
    _ensure_sadtalker_installed()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result_dir = os.path.join("outputs", "sadtalker_raw")
    os.makedirs(result_dir, exist_ok=True)

    abs_photo = os.path.abspath(photo_path)
    abs_audio = os.path.abspath(audio_path)
    abs_result_dir = os.path.abspath(result_dir)

    cmd = [
        sys.executable, "inference.py",
        "--driven_audio", abs_audio,
        "--source_image", abs_photo,
        "--result_dir", abs_result_dir,
        "--still",
        "--preprocess", "crop",
    ]

    print("[avatar] Running SadTalker inference (this can take several minutes)...")
    result = subprocess.run(cmd, cwd=SADTALKER_DIR, capture_output=True, text=True)

    if result.returncode != 0:
        print("--- SadTalker stdout (tail) ---")
        print(result.stdout[-2500:])
        print("--- SadTalker stderr (tail) ---")
        print(result.stderr[-2500:])
        raise RuntimeError("SadTalker inference failed. See output above for details.")

    # SadTalker writes its output as <result_dir>/<timestamp>.mp4 (or nested)
    candidates = glob.glob(os.path.join(result_dir, "**", "*.mp4"), recursive=True)
    if not candidates:
        raise RuntimeError("SadTalker ran but no output video file was found.")
    latest = max(candidates, key=os.path.getmtime)

    shutil.copy(latest, output_path)
    print(f"[avatar] Full-length avatar video saved to {output_path}")
    return output_path


if __name__ == "__main__":
    import sys as _sys
    photo = _sys.argv[1] if len(_sys.argv) > 1 else "my_photo.jpg"
    audio = _sys.argv[2] if len(_sys.argv) > 2 else "outputs/voice.wav"
    generate_avatar_video(photo, audio)
