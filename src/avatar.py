"""
Avatar Lip-Sync Module (Wav2Lip)
-----------------------------------
Takes the user's own photo + the narration audio and produces a lip-synced
"talking head" video clip using the open-source Wav2Lip model. This is
then overlaid as a small corner "news-anchor style" box on the final video.

NOTE: This is the most complex, resource-heavy stage in the pipeline.
  - Clones the Wav2Lip repo (first run only).
  - Downloads a ~436MB pretrained checkpoint + face-detection model
    (first run only), from a Hugging Face mirror (more reliable than the
    original Google Drive links, which often require manual access).
  - Requires a GPU. Can take a few minutes per video.

Usage:
    from src.avatar import generate_avatar_lipsync
    avatar_path = generate_avatar_lipsync("my_photo.jpg", "outputs/voice.wav")
"""

import os
import sys
import subprocess
import shutil

WAV2LIP_DIR = "Wav2Lip"
CHECKPOINT_PATH = os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth")
S3FD_PATH = os.path.join(WAV2LIP_DIR, "face_detection", "detection", "sfd", "s3fd.pth")

_HF_REPO = "manavisrani07/gradio-lipsync-wav2lip"


def _ensure_wav2lip_installed():
    """Clone the Wav2Lip repo and download required model weights, if missing."""
    if not os.path.isdir(WAV2LIP_DIR):
        print("[avatar] Cloning Wav2Lip repo (first time only)...")
        subprocess.run(
            ["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git", WAV2LIP_DIR],
            check=True,
        )

    from huggingface_hub import hf_hub_download

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    if not os.path.exists(CHECKPOINT_PATH):
        print("[avatar] Downloading Wav2Lip GAN checkpoint (~436MB, first time only)...")
        downloaded = hf_hub_download(
            repo_id=_HF_REPO, repo_type="space",
            filename="checkpoints/wav2lip_gan.pth",
        )
        shutil.copy(downloaded, CHECKPOINT_PATH)

    os.makedirs(os.path.dirname(S3FD_PATH), exist_ok=True)
    if not os.path.exists(S3FD_PATH):
        print("[avatar] Downloading face detection model (first time only)...")
        downloaded = hf_hub_download(
            repo_id=_HF_REPO, repo_type="space",
            filename="face_detection/detection/sfd/s3fd.pth",
        )
        shutil.copy(downloaded, S3FD_PATH)


def generate_avatar_lipsync(
    photo_path: str,
    audio_path: str,
    output_path: str = "outputs/avatar_lipsync.mp4",
) -> str:
    """
    Generate a lip-synced talking-head video from a still photo + audio.

    Args:
        photo_path: path to the user's photo (a clear, front-facing face works best)
        audio_path: path to the narration audio (e.g. outputs/voice.wav)
        output_path: where to save the resulting lip-synced video

    Returns:
        The output_path of the generated video.
    """
    _ensure_wav2lip_installed()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    abs_output = os.path.abspath(output_path)
    abs_photo = os.path.abspath(photo_path)
    abs_audio = os.path.abspath(audio_path)
    abs_checkpoint = os.path.abspath(CHECKPOINT_PATH)

    cmd = [
        sys.executable, "inference.py",
        "--checkpoint_path", abs_checkpoint,
        "--face", abs_photo,
        "--audio", abs_audio,
        "--outfile", abs_output,
        "--pads", "0", "20", "0", "0",
    ]

    print("[avatar] Running Wav2Lip inference (this can take a few minutes)...")
    result = subprocess.run(cmd, cwd=WAV2LIP_DIR, capture_output=True, text=True)

    if result.returncode != 0:
        print("--- Wav2Lip stdout (tail) ---")
        print(result.stdout[-2000:])
        print("--- Wav2Lip stderr (tail) ---")
        print(result.stderr[-2000:])
        raise RuntimeError("Wav2Lip inference failed. See output above for details.")

    print(f"[avatar] Avatar lip-sync video saved to {output_path}")
    return output_path


if __name__ == "__main__":
    import sys as _sys
    photo = _sys.argv[1] if len(_sys.argv) > 1 else "my_photo.jpg"
    audio = _sys.argv[2] if len(_sys.argv) > 2 else "outputs/voice.wav"
    generate_avatar_lipsync(photo, audio)
