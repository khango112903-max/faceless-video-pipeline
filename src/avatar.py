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

_HF_SPACE_REPO = "barisaydin/sadtalker"
_HF_SPACE_REPO_TYPE = "model"


def _patch_numpy2_incompatibilities():
    """
    SadTalker's code was written for numpy 1.x and uses several attributes
    that were removed in numpy 2.x (e.g. np.VisibleDeprecationWarning,
    np.float, np.int, np.bool, np.object, np.str, np.complex). Rather than
    downgrading numpy globally (which breaks scipy/Bark, which need
    numpy 2.x), we patch these deprecated references directly in the
    cloned source so everything can share one numpy version.
    """
    import re

    # (pattern, replacement) — applied across every .py file in the repo
    replacements = [
        (r"\bnp\.VisibleDeprecationWarning\b", "DeprecationWarning"),
        (r"\bnp\.float\b(?!\d|_)", "float"),
        (r"\bnp\.int\b(?!\d|_|e)", "int"),
        (r"\bnp\.bool\b(?!\d|_)", "bool"),
        (r"\bnp\.object\b(?!\d|_)", "object"),
        (r"\bnp\.str\b(?!\d|_|i)", "str"),
        (r"\bnp\.complex\b(?!\d|_)", "complex"),
        # numpy 2.x raises a hard error (used to just warn) when building an
        # array from a mix of plain scalars and array-like elements (ragged
        # shape). SadTalker's align_img() does exactly this; cast t[0]/t[1]
        # to plain floats first so the array is a clean, uniform shape.
        (
            r"np\.array\(\[w0, h0, s, t\[0\], t\[1\]\]\)",
            "np.array([w0, h0, s, float(t[0]), float(t[1])])",
        ),
    ]

    patched_count = 0
    for root, _dirs, files in os.walk(SADTALKER_DIR):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            new_content = content
            for pattern, repl in replacements:
                new_content = re.sub(pattern, repl, new_content)

            if new_content != content:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(new_content)
                patched_count += 1

    if patched_count:
        print(f"[avatar] Patched numpy 2.x incompatibilities in {patched_count} file(s)")


def _patch_basicsr_torchvision_compat():
    """
    basicsr (a gfpgan dependency, imported unconditionally by SadTalker's
    face_enhancer module even when the enhancer isn't used) imports
    `torchvision.transforms.functional_tensor`, which was removed in newer
    torchvision versions. Patch the installed basicsr package directly.

    IMPORTANT: we locate the file via importlib.util.find_spec() instead of
    `import basicsr` — actually importing basicsr executes its __init__.py,
    which triggers this exact same crash before we get a chance to patch it.
    find_spec() locates the file without executing any of its code.
    """
    import importlib.util

    spec = importlib.util.find_spec("basicsr")
    if spec is None or not spec.submodule_search_locations:
        return

    basicsr_dir = list(spec.submodule_search_locations)[0]
    target_file = os.path.join(basicsr_dir, "data", "degradations.py")
    if not os.path.exists(target_file):
        return

    with open(target_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    patched = content.replace(
        "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
        "from torchvision.transforms.functional import rgb_to_grayscale",
    )

    if patched != content:
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(patched)
        print("[avatar] Patched basicsr/torchvision compatibility")


_TORCH_LOAD_SHIM_MARKER = "_patched_weights_only"

_TORCH_LOAD_SHIM = '''import torch as _torch_patch_shim
if not getattr(_torch_patch_shim.load, "_patched_weights_only", False):
    _orig_torch_load_shim = _torch_patch_shim.load
    def _patched_torch_load_shim(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_torch_load_shim(*a, **kw)
    _patched_torch_load_shim._patched_weights_only = True
    _torch_patch_shim.load = _patched_torch_load_shim
# --- end torch.load patch ---

'''


def _patch_torch_load_weights_only():
    """
    Newer PyTorch defaults torch.load's `weights_only` to True, which
    breaks loading SadTalker's older pretrained checkpoints (same class of
    issue we hit with Bark). Since SadTalker runs as a subprocess, we
    inject a small shim at the very top of inference.py that monkeypatches
    torch.load for that process, defaulting weights_only=False (safe here
    since these are trusted, official SadTalker checkpoints).
    """
    target_file = os.path.join(SADTALKER_DIR, "inference.py")
    if not os.path.exists(target_file):
        return

    with open(target_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    if _TORCH_LOAD_SHIM_MARKER in content:
        return  # already patched

    with open(target_file, "w", encoding="utf-8") as f:
        f.write(_TORCH_LOAD_SHIM + content)

    print("[avatar] Patched torch.load weights_only compatibility")


def _checkpoints_look_valid():
    """
    Check that the critical face-reconstruction checkpoint (epoch_20.pth)
    exists and is a reasonable size (~100MB+). This file has been seen to
    end up truncated/corrupted after an interrupted earlier download,
    which then causes a confusing "unpickling stack underflow" error much
    later during inference instead of a clear download error up front.
    """
    critical_file = os.path.join(CHECKPOINTS_DIR, "epoch_20.pth")
    if not os.path.exists(critical_file):
        return False
    size_mb = os.path.getsize(critical_file) / (1024 * 1024)
    return size_mb > 50  # real file is ~100MB+; a truncated one will be tiny


def _ensure_sadtalker_installed():
    """Clone the SadTalker repo and download required model checkpoints, if missing."""
    if not os.path.isdir(SADTALKER_DIR):
        print("[avatar] Cloning SadTalker repo (first time only)...")
        subprocess.run(
            ["git", "clone", "https://github.com/OpenTalker/SadTalker.git", SADTALKER_DIR],
            check=True,
        )

    _patch_numpy2_incompatibilities()
    _patch_basicsr_torchvision_compat()
    _patch_torch_load_weights_only()

    if not _checkpoints_look_valid():
        print("[avatar] Downloading SadTalker checkpoints (~4GB)...")
        if os.path.isdir(CHECKPOINTS_DIR):
            print("[avatar] Removing incomplete/corrupted checkpoints from a previous run...")
            shutil.rmtree(CHECKPOINTS_DIR)

        from huggingface_hub import snapshot_download

        downloaded_dir = snapshot_download(
            repo_id=_HF_SPACE_REPO,
            repo_type=_HF_SPACE_REPO_TYPE,
            allow_patterns=["checkpoints/**"],
            force_download=True,
        )
        src_checkpoints = os.path.join(downloaded_dir, "checkpoints")
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        shutil.copytree(src_checkpoints, CHECKPOINTS_DIR, dirs_exist_ok=True)

        if not _checkpoints_look_valid():
            raise RuntimeError(
                "SadTalker checkpoint download completed but epoch_20.pth still "
                "looks invalid/too small. The Hugging Face mirror may have changed "
                "— check https://huggingface.co/spaces/vinthony/SadTalker/tree/main/checkpoints"
            )


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

    # Re-save the photo as a clean, standard JPEG. This fixes a common class
    # of failures in SadTalker/face-detection preprocessing caused by EXIF
    # rotation tags, CMYK color mode, alpha channels, or unusual encodings
    # in the original upload — regardless of the original file's extension.
    from PIL import Image, ImageOps

    clean_photo_path = os.path.join("outputs", "avatar_source_photo.jpg")
    img = Image.open(photo_path)
    img = ImageOps.exif_transpose(img)  # apply any EXIF rotation, then drop it
    img = img.convert("RGB")
    img.save(clean_photo_path, "JPEG", quality=95)
    photo_path = clean_photo_path
    print(f"[avatar] Re-saved photo as clean JPEG: {clean_photo_path}")

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
