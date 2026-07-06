"""
Pipeline Orchestrator
---------------------
Runs all stages in order: script -> research -> voice -> visuals -> subtitles -> assembly

STATUS: Only script_generation is wired up so far. Other stages will be
plugged in as they're built.
"""

from src.script_generation import generate_script


def run_pipeline(topic: str):
    print(f"[1/6] Generating script for topic: {topic}")
    script = generate_script(topic)
    print("Script generated:")
    print(script)

    # TODO: research (optional, if you want fact-checking/deeper research)
    # TODO: voice_generation.generate_voice(script)
    # TODO: visuals.generate_visuals(script)
    # TODO: subtitles.generate_subtitles(voice_audio_path)
    # TODO: assembly.assemble_video(...)

    return script


if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "A surprising historical fact"
    run_pipeline(topic)
