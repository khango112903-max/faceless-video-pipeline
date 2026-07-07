"""
Pipeline Orchestrator
---------------------
Runs all stages in order: script -> research -> voice -> visuals -> subtitles -> assembly

STATUS: Only script_generation is wired up so far. Other stages will be
plugged in as they're built.
"""

from src.script_generation import generate_script
from src.voice_generation import generate_voice
from src.visuals import generate_visuals


def run_pipeline(topic: str):
    print(f"[1/6] Generating script for topic: {topic}")
    script = generate_script(topic)
    print("Script generated:")
    print(script)

    print("[2/6] Generating narration audio (Bark)...")
    audio_path = generate_voice(script)
    print(f"Narration saved at: {audio_path}")

    print("[3/6] Generating visuals (Pexels + fallback)...")
    scenes = generate_visuals(script)
    print(f"Generated {len(scenes)} scene clips.")

    # TODO: research (optional, if you want fact-checking/deeper research)
    # TODO: subtitles.generate_subtitles(audio_path)
    # TODO: assembly.assemble_video(...)

    return {"script": script, "audio_path": audio_path, "scenes": scenes}


if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "A surprising historical fact"
    run_pipeline(topic)
