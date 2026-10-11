"""
Algerian Darja TTS engine.

Backends:
- ``mms-ara``: facebook/mms-tts-ara — MSA smoke / A-side baseline only (not Darja product).
- ``f5tts``: F5-TTS zero-shot / fine-tuned path (optional dependency).

The sine/harmonic fallback is never treated as a successful demo; it only runs when
``allow_harmonic_fallback=True`` for offline unit tests.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AlgerianDarjaTextProcessor:
    """
    Lightweight Darja text cleaner: emotion tags, Arabizi digit map, whitespace.
    Emotion tags do **not** drive acoustic expression beyond optional amplitude hacks
    (kept only for backward-compatible MMS experiments; not product emotion).
    """

    SUPPORTED_EMOTIONS = ["neutral", "happy", "excited", "sad", "angry", "sarcastic", "calm"]

    ARABIZI_MAP = {
        "3": "ع",
        "7": "ح",
        "9": "ق",
        "5": "خ",
        "2": "أ",
    }

    def extract_emotion_and_clean_text(self, text: str) -> Tuple[str, str]:
        emotion = "neutral"
        match = re.search(r"\[([a-zA-Z]+)\]", text)
        if match:
            found_tag = match.group(1).lower()
            if found_tag in self.SUPPORTED_EMOTIONS:
                emotion = found_tag
            text = re.sub(r"\[[a-zA-Z]+\]", "", text).strip()
        return emotion, text

    def normalize_darja_text(self, text: str) -> Tuple[str, str]:
        emotion, clean_text = self.extract_emotion_and_clean_text(text)
        for k, v in self.ARABIZI_MAP.items():
            clean_text = clean_text.replace(k, v)
        clean_text = re.sub(r"[\?\!]{2,}", "!", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        return clean_text, emotion


class AlgerianTTSEngine:
    """
    Multi-backend engine. Default is honest MMS-TTS Arabic (MSA baseline).
    """

    DEFAULT_MODEL_HUB = "facebook/mms-tts-ara"

    def __init__(
        self,
        backend_model: str = "mms-ara",
        sample_rate: int = 16000,
        allow_harmonic_fallback: bool = False,
        f5_ckpt: Optional[str] = None,
        f5_vocab: Optional[str] = None,
        f5_model_name: str = "F5TTS_v1_Base",
    ):
        self.backend_model = backend_model.lower()
        self.sample_rate = sample_rate
        self.allow_harmonic_fallback = allow_harmonic_fallback
        self.text_processor = AlgerianDarjaTextProcessor()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._f5 = None
        self.f5_ckpt = f5_ckpt
        self.f5_vocab = f5_vocab
        self.f5_model_name = f5_model_name

        if self.backend_model in ("mms-ara", "mms"):
            self._load_mms()
        elif self.backend_model in ("f5tts", "f5"):
            self._load_f5()
        else:
            raise ValueError(f"Unknown backend_model={backend_model!r}. Use 'mms-ara' or 'f5tts'.")

        logger.info(
            "Initialized AlgerianTTSEngine backend=%s device=%s",
            self.backend_model,
            self.device,
        )

    def _load_mms(self) -> None:
        try:
            from transformers import AutoTokenizer, VitsModel

            logger.info("Loading MMS Arabic baseline: %s", self.DEFAULT_MODEL_HUB)
            self.model = VitsModel.from_pretrained(self.DEFAULT_MODEL_HUB).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.DEFAULT_MODEL_HUB)
            self.sample_rate = self.model.config.sampling_rate
        except Exception as e:
            logger.error("Failed to load MMS model: %s", e)
            self.model = None
            self.tokenizer = None

    def _load_f5(self) -> None:
        try:
            from f5_tts.api import F5TTS

            kwargs = {"model": self.f5_model_name, "device": self.device}
            if self.f5_ckpt:
                kwargs["ckpt_file"] = self.f5_ckpt
            if self.f5_vocab:
                kwargs["vocab_file"] = self.f5_vocab
            self._f5 = F5TTS(**kwargs)
            self.sample_rate = 24000
            logger.info("F5-TTS loaded (model=%s).", self.f5_model_name)
        except Exception as e:
            logger.error("Failed to load F5-TTS: %s", e)
            self._f5 = None

    def synthesize(
        self,
        text: str,
        speaker_id: str = "algiers_male_01",
        emotion: Optional[str] = None,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        apply_emotion_gain: bool = False,
    ) -> Dict[str, Any]:
        clean_text, detected_emotion = self.text_processor.normalize_darja_text(text)
        selected_emotion = emotion if emotion else detected_emotion

        if self.backend_model in ("mms-ara", "mms"):
            if self.model is None or self.tokenizer is None:
                if self.allow_harmonic_fallback:
                    audio_data = self._synthesize_harmonic_fallback(clean_text, selected_emotion, speed)
                    backend = "harmonic_fallback"
                else:
                    raise RuntimeError(
                        "MMS model unavailable and harmonic fallback disabled "
                        "(fallback is not a valid Darja demo)."
                    )
            else:
                audio_data = self._synthesize_mms(clean_text, selected_emotion, apply_emotion_gain)
                backend = "mms-ara"
        else:
            if self._f5 is None:
                raise RuntimeError("F5-TTS backend not loaded.")
            if not reference_audio_path:
                raise ValueError("f5tts backend requires reference_audio_path for zero-shot cloning.")
            audio_data = self._synthesize_f5(
                clean_text,
                reference_audio_path=reference_audio_path,
                reference_text=reference_text or "",
                speed=speed,
            )
            backend = "f5tts"

        return {
            "audio": audio_data,
            "sample_rate": self.sample_rate,
            "text": clean_text,
            "speaker_id": speaker_id,
            "emotion": selected_emotion,
            "duration": round(len(audio_data) / self.sample_rate, 3),
            "backend": backend,
        }

    def _synthesize_mms(self, text: str, emotion: str, apply_emotion_gain: bool) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs).waveform
        waveform = output.squeeze().cpu().numpy()
        # Amplitude "emotion" is an explicit opt-in hack — not real expression.
        if apply_emotion_gain:
            if emotion in ["excited", "happy"]:
                waveform = waveform * 1.15
            elif emotion == "sad":
                waveform = waveform * 0.85
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = (waveform / max_val) * 0.90
        return waveform.astype(np.float32)

    def _synthesize_f5(
        self,
        text: str,
        reference_audio_path: str,
        reference_text: str,
        speed: float,
    ) -> np.ndarray:
        wav, sr, _ = self._f5.infer(
            ref_file=reference_audio_path,
            ref_text=reference_text,
            gen_text=text,
            speed=speed,
        )
        self.sample_rate = int(sr)
        return np.asarray(wav, dtype=np.float32)

    def _synthesize_harmonic_fallback(self, text: str, emotion: str, speed: float) -> np.ndarray:
        """Offline test-only tone generator — not speech."""
        words = text.split()
        num_words = max(1, len(words))
        base_dur = 0.35 / max(0.5, speed)
        total_dur = num_words * base_dur + 0.4
        t = np.linspace(0, total_dur, int(self.sample_rate * total_dur), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 220.0 * t)
        return audio.astype(np.float32)


if __name__ == "__main__":
    engine = AlgerianTTSEngine(backend_model="mms-ara")
    res = engine.synthesize("السلام عليكم", speaker_id="algiers_01")
    logger.info("Synthesized %s s via %s", res["duration"], res["backend"])
