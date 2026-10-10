# Algerian Darja (Algiers Dialect) Text-to-Speech (TTS) Framework

An experimental, scalable, and continuous framework for scraping, processing, fine-tuning, and synthesizing **Algerian Darja** (starting with the **Algiers / العاصمة** dialect) using state-of-the-art zero-shot, multi-speaker, and emotion-conditioned Text-to-Speech models.

---

## Features
- **Modern SOTA TTS Architecture Support**: Adapters for F5-TTS (Flow Matching Transformer), Coqui XTTS v2 (Zero-shot cloning), and lightweight synthesis engines.
- **Emotion & Expression Conditioning**: Supports emotion tags in text prompts (`[happy]`, `[excited]`, `[angry]`, `[sad]`, `[sarcastic]`, `[calm]`).
- **Algerian Darja Text Processor**: Normalizes text, extracts expression tags, handles punctuation, and supports Arabic script & Arabizi mappings.
- **Automated Data Scraping Pipeline**: Built-in YouTube / Podcast crawler (`src/dataset/scraper.py`) for collecting public Algerian voice data.
- **Audio Preprocessing & Segmentation**: Automatic 24kHz resampling, silence-based segment splitting (`librosa`), peak amplitude normalization, and dataset manifest creation (`src/dataset/preprocess.py`).
- **Fine-Tuning Loop & DataLoader**: PyTorch training interface with checkpoint saving and loss tracking (`src/training/finetune.py`).

---

## Directory Structure
```
.
├── RESEARCH_MODELS.md              # SOTA TTS model evaluation & deep dives
├── EXPERIMENTAL_LOG.md             # Experiments, progress, challenges & test results log
├── README.md                       # Project overview and usage guide
├── data/                           # Raw & processed audio dataset directory
│   └── baseline_algiers_sample.wav # Generated sample baseline audio
├── src/
│   ├── dataset/
│   │   ├── scraper.py             # Algerian audio scraper (YouTube, Podcasts)
│   │   └── preprocess.py          # Resampling, silence splitting & normalization
│   ├── models/
│   │   └── tts_engine.py          # AlgerianDarjaTextProcessor & AlgerianTTSEngine
│   └── training/
│       └── finetune.py            # PyTorch Dataset loader & Fine-Tuner loop
└── tests/
    └── test_dataset_and_models.py # Unit and integration tests
```

---

## Quick Start

### 1. Installation
Ensure Python 3.10+ is installed along with dependencies:
```bash
pip install yt-dlp librosa soundfile scipy pydub pytest torch torchaudio transformers
```

### 2. Synthesize Algerian Darja Speech
```python
from src.models.tts_engine import AlgerianTTSEngine
import soundfile as sf

engine = AlgerianTTSEngine(backend_model="f5tts", sample_rate=24000)

# Generate speech with emotion tag
result = engine.synthesize(
    text="[happy] واش راك خويا، لاباس؟ مرحبتين بيك معنا!",
    speaker_id="algiers_speaker_01"
)

# Save output WAV file
sf.write("output_algiers.wav", result["audio"], result["sample_rate"])
print(f"Generated {result['duration']} seconds of speech.")
```

### 3. Run Automated Tests
```bash
PYTHONPATH=. pytest tests/test_dataset_and_models.py
```

---

## Roadmap & Future Steps
1. Scale automated scraping to collect 50+ hours of Algiers dialogue speech.
2. Integrate Whisper large-v3 for automated speech-to-text transcript alignment.
3. Conduct full GPU fine-tuning runs using F5-TTS flow matching models.
