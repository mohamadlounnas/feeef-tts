# Research & Evaluation: Modern SOTA Text-to-Speech (TTS) Models for Algerian Darja (Algiers Dialect)

## Overview & Goal
The objective is to establish an experimental, scalable, and high-quality Text-to-Speech (TTS) pipeline specifically tailored for **Algerian Darja** (starting with the **Algiers / العاصمة** dialect).

### Key Technical Requirements
1. **Dialect & Script Support**: Support Arabic script and Arabizi (Latin-scripted Darja) with Algiers phonetic nuances (e.g. French loanwords, unique phonemes like /g/, fast rhythm, code-switching).
2. **Emotion & Expression Conditioning**: Ability to generate speech with varied emotions (happy, angry, sad, neutral, excited, sarcastic) or prosodic tags.
3. **Multi-Speaker & Voice Cloning**: Zero-shot prompt-based voice cloning using short reference audio samples (3-10 seconds) or multi-speaker embedding adaptation.
4. **Fine-Tuning & Adaptation Efficiency**: Ability to fine-tune pre-trained multilingual or Arabic foundation models on custom Algerian Darja dataset clips scraped from public media (podcasts, YouTube, dialogue).

---

## Candidate SOTA Models Evaluation

| Model Architecture | License | Multi-Speaker | Voice Cloning | Emotion Control | Arabic / Multilingual Base | Fine-tuning Feasibility | Recommendation Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **F5-TTS / E2 TTS** (Flow Matching Transformer) | MIT | Yes | Zero-shot (3-10s prompt) | High (Inferred from prompt audio/text) | Strong (Multilingual / Non-autoregressive) | High (Fast training, robust to missing alignment) | **9.5 / 10 (Top Pick)** |
| **Coqui XTTS v2** (GPT + VQGAN + HiFi-GAN) | CPML / Custom | Yes | Zero-shot reference cloning | High (Expressive prompt copying) | Native Arabic support | High (Well-documented fine-tuning scripts) | **9.0 / 10 (Primary Baseline)** |
| **ChatTTS** (Autoregressive Flow / Conversational) | AGPLv3 | Yes | Speaker embedding / Prompting | Very High (Laughter, pauses, emotion tags) | Primary Chinese/English (Adaptable via fine-tuning) | Medium (Heavy compute for full fine-tuning) | **8.5 / 10** |
| **Bark** (Suno Transformer LLM TTS) | MIT | Yes | Prompt history cloning | Very High (`[laughter]`, `[sighs]`, etc.) | Multilingual (incl. Arabic) | Low/Medium (Large parameter count) | **7.5 / 10** |
| **StyleTTS2** (Style Diffusion / Adversarial) | MIT | Yes | Style vector cloning | High | Requires phonemizer/Arabic setup | Medium | **8.0 / 10** |
| **Piper TTS** (VITS-based Light On-device) | MIT | Yes | Speaker ID fine-tuning | Moderate | High (Pre-trained Arabic models available) | High (Very fast, CPU friendly) | **8.0 / 10 (Fast CPU Deployment)** |

---

## Detailed Model Deep-Dives

### 1. F5-TTS (Flow-Matching Speech Synthesizer)
- **Architecture**: Fast Diffusion / Flow Matching transformer.
- **Why it fits Algerian Darja**:
  - Non-autoregressive, meaning it avoids word repeating/skipping errors common in AR models when fed mixed languages (Arabic + French loanwords).
  - Can take a 5-10s clip of Algerian Darja speech as reference, and synthesize new Darja text in that exact voice and emotion without needing precise alignment/phonemization during inference.
- **Fine-Tuning Strategy**: Prepare 1-5 hours of clean Algerian Darja audio clips + transcripts.

### 2. Coqui XTTS v2
- **Architecture**: Autoregressive LM + Audio Tokenizer + HiFi-GAN decoder.
- **Why it fits Algerian Darja**:
  - Out-of-the-box support for Arabic text tokenization.
  - Excellent voice cloning with 3-6 second reference audio.
  - Can be fine-tuned using `TTS` library (`Trainer` API).

### 3. ChatTTS
- **Architecture**: Designed for natural conversational dialogue, pauses, emotion interjections (`[smile]`, `[laughter]`).
- **Why it fits Algerian Darja**:
  - Algerian Darja is heavily conversational and spoken in fast-paced dialogue formats. ChatTTS handles interjections exceptionally well.

---

## Speech Scraping & Dataset Acquisition Strategy (Algerian Algiers Dialect)

1. **Target Sources**:
   - Algerian Podcasts (YouTube, Anchor, Spotify public clips).
   - Local Radio/TV Talk Shows (radio interviews, street interviews - *Micro-Trottoir*).
   - Public Short-form content (TikTok/Reels with clear speech and Algiers accent).
2. **Audio Processing**:
   - Download audio via `yt-dlp`.
   - Silence-removal and audio segmentation into 3-12 second chunks (`pydub` / `librosa`).
   - Loudness normalization (EBU R128 to -23 dBFS).
   - Resampling to standard rate (24,000 Hz or 22,050 Hz).
3. **Transcription & Dialect Tagging**:
   - Automated transcription via Whisper (Arabic/multilingual fine-tune).
   - Manual/Rule-based post-normalization for Darja words (e.g. `واش راك`, `كيما`, `دير`, `شغال`, `بزاف`).

---

## Architectural Strategy for Algerian Darja TTS Engine
We build a unified Python library structure (`src/`) that exposes:
1. `DatasetPipeline`: Automated YouTube/podcast downloading, VAD/silence segmentation, audio normalization, transcription.
2. `DarjaTextProcessor`: Standardizes Algerian Darja text (Arabic script & Arabizi conversions, punctuation, emotion tags like `[happy]`, `[angry]`).
3. `AlgerianTTSEngine`: A multi-backend adapter supporting zero-shot cloning, emotion parameterization, and model inference.
4. `FineTuner`: Pipeline to fine-tune models on curated Darja voice datasets.
