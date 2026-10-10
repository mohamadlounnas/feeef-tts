# Algerian Darija (Algiers) Text-to-Speech (TTS) Architecture & Model Research

## Executive Summary
This document outlines the state-of-the-art (SOTA) open-source Text-to-Speech (TTS) models evaluated for porting and fine-tuning specifically for **Algerian Darija (Algiers dialect)**.
Our primary objectives are:
1. **Dialogue & Modern Dialect Support**: Robust handling of Algerian Algiers Darija (mixed Arabic-French-Berber vocabulary, code-switching, and local phonetic nuances).
2. **Emotion & Expressiveness**: Support for controllable emotional tone (happy, angry, sad, excited, neutral, whisper, energetic).
3. **Multi-Speaker & Zero-Shot Voice Cloning**: Capability to synthesize multiple voices or clone a speaker from 3-10 seconds of reference audio.
4. **Controllability & Parametric Tuning**: Speed, pitch, temperature, and stylistic prompt steering.

---

## Candidate SOTA TTS Architectures

| Architecture | Model Family / Paradigm | Emotion Support | Voice Cloning | Arabic / Dialect Fit | Training Efficiency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Parler-TTS** | Controllable Audio Language Model | **Native via Description** ("A energetic Algiers male voice...") | Via prompt/speaker embedding | High (uses EnCodec/DAC + Transformer) | Medium (Requires fine-tuning text decoder & audio tokens) |
| **F5-TTS / E2-TTS** | Flow Matching Audio Diffusion | **Inherent in Reference Audio / In-context** | **SOTA Zero-Shot (3s ref audio)** | Excellent (Phoneme-free / text-guided flow) | High (Fast training & convergence) |
| **Coqui XTTS v2** | Multi-lingual Autoregressive + HiFi-GAN | High (Conditioned on audio prompt) | Excellent (Zero-shot audio prompt) | Good (Pre-trained on Modern Standard Arabic) | Moderate |
| **Suno Bark** | Audio LM (GPT-style semantic + acoustic tokens) | High (Special emotion tags `[laughs]`, `[sighs]`) | Good (Voice preset / clone via prompt vectors) | Medium (Has Arabic baseline, prone to hallucination) | Low |
| **Kokoro / OuteTTS** | Lightweight Quantized Audio LM | Prompt-conditioned | Medium | Emerging (Lightweight deployment focused) | Very High |

---

## Recommended Architecture Strategy: Dual-Engine Pipeline

### Engine A: F5-TTS / Flow-Matching (Primary Voice Cloning & Dialect Synthesis Engine)
- **Why**: Flow-matching diffusion models (F5-TTS) bypass strict phoneme dictionary requirements, making them ideal for dialectal Darija, which lacks standard diacritics and standardized orthography.
- **Strength**: High fidelity zero-shot cloning from 3-10 second Algiers audio clips, preserving local accent inflections naturally.

### Engine B: Parler-TTS Style Controllable Transformer (Primary Emotion & Dialogue Engine)
- **Why**: Parler-TTS uses dual encoders (Text Description + Prompt Text). By conditioning speech generation on text prompts like *"A male speaker from Algiers speaking with an excited, emotional tone in a studio setting"*, we achieve fine-grained dialectal emotion control.

---

## Algerian Darija (Algiers) Linguistic & Phonetic Adaptation Strategy

1. **Orthographic Normalization**:
   - Standardize Algiers Darija text representations (handling Arabic script `الدارجة الجزائريـة` and Latin/Arabizi if applicable).
   - Normalize character variants (`أ, إ, آ -> ا`, `ة -> ه`, `ى -> ي`, retaining Algiers dialect specific usages like `ڤ` / `ڨ` for hard G sound).
2. **Audio Preprocessing & Segmentation**:
   - Audio downsampling to 24kHz / 22.05kHz mono.
   - Voice Activity Detection (VAD) and silence trimming.
   - Normalization of loudness to -20 dBFS.
   - Automatic chunking into 3-15 second clips with high-confidence transcript alignments.
3. **Synthetic / Fine-tuning Data Pipeline Loop**:
   - Raw audio acquisition -> VAD/Demucs noise removal -> Whisper Transcribe -> Standardized Normalization -> Alignment Quality Filtering -> Dataset Creation -> Model Fine-Tuning -> Synthetic Voice Generation -> WER/Audio Quality Evaluation.
