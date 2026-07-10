# 🎬 Faceless YouTube Video Pipeline

Automated pipeline for creating faceless YouTube videos (facts / motivation / history)
using free-tier AI tools, running on Google Colab.

## 📦 Pipeline Stages

| Stage        | Tool(s)                              | File                        |
|--------------|---------------------------------------|-----------------------------|
| Script       | Gemini (active) / Kimi K2 (switchable)| `src/script_generation.py`  |
| Research     | Perplexity / Gemini                   | `src/research.py`           |
| Voice (TTS)  | Bark / Kokoro                         | `src/voice_generation.py`   |
| Visuals      | Pexels (stock clips) + PIL/moviepy (fallback) | `src/visuals.py`     |
| Subtitles    | Whisper                               | `src/subtitles.py`          |
| Music        | Pixabay / YouTube Audio Library       | `assets/music/`             |
| Assembly     | FFmpeg                                | `src/assembly.py`           |
| Orchestration| `pipeline.py`                         | `src/pipeline.py`           |

## 🚀 Setup (Google Colab)

1. Clone this repo in Colab:
   ```python
   !git clone <your-repo-url>
   %cd faceless-video-pipeline
   !pip install -r requirements.txt
   ```

2. Add your API keys in Colab secrets (or `.env`):
   - `GEMINI_API_KEY`
   - `KIMI_API_KEY` (fallback if Gemini quota is exceeded)
   - `PEXELS_API_KEY` (for stock video clips)
   - `PERPLEXITY_API_KEY` (optional)

3. Run the pipeline:
   ```python
   from src.pipeline import run_pipeline
   run_pipeline(topic="Interesting history fact about ancient Rome")
   ```

## 🚀 Setup (Kaggle — recommended, more RAM than Colab free tier)

1. Create a new Kaggle Notebook.
2. In Notebook settings (right panel): set **Accelerator = GPU T4 x2**,
   and turn **Internet = ON** (required for git clone / pip install).
3. Add your API keys via **Add-ons → Secrets** in the notebook menu:
   `GEMINI_API_KEY`, `KIMI_API_KEY`, `PEXELS_API_KEY`
4. Cell 1 (clone + install):
   ```python
   !git clone <your-repo-url>
   %cd faceless-video-pipeline
   !pip install -r requirements.txt
   ```
5. Cell 2 (run with your own topic):
   ```python
   from src.pipeline import run_pipeline
   my_topic = input("Enter your video topic: ")
   result = run_pipeline(topic=my_topic)
   ```

## 🔧 Current Status

- [x] Repo structure
- [x] Script generation (Gemini) — Kimi switch ready, key pending
- [x] Voice generation (Bark TTS) — requires GPU, run in Colab
- [ ] Research module
- [x] Visuals (Pexels stock clips + text-card fallback)
- [x] Subtitles (Whisper)
- [ ] Music mixing
- [x] Final assembly (moviepy, subtitles burned in + optional music)
- [x] Full pipeline orchestration (script → voice → visuals → subtitles → final video)

## 📁 Folder Structure

```
faceless-video-pipeline/
├── README.md
├── requirements.txt
├── .env.example
├── configs/
│   └── settings.py
├── src/
│   ├── script_generation.py
│   ├── research.py
│   ├── voice_generation.py
│   ├── visuals.py
│   ├── subtitles.py
│   ├── assembly.py
│   └── pipeline.py
├── assets/
│   ├── music/
│   └── fonts/
├── notebooks/
│   └── main_colab.ipynb
└── outputs/
```
