"""
ASR draft transcription + light automatic QC for seed segments.

Uses faster-whisper (preferred) or openai-whisper. QC is heuristic — not a
substitute for human Maghrebi orthography review.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Arabic / Latin letters + digits + common punctuation (Darja + FR loanwords)
_AR_LATIN_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"
    r"A-Za-z0-9\s\.,;:!?\-'\"%\(\)/]+"
)


@dataclass
class AsrResult:
    text: str
    language: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    model_name: str


class SeedAsrTranscriber:
    """
    Segment ASR with Maghrebi-friendly settings.

    Prefer ``faster-whisper`` ``small`` / ``medium`` over Whisper-tiny for CAFE.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "ar",
        backend: str = "faster_whisper",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.backend = backend
        self._fw = None
        self._ow = None
        self._load()

    def _load(self) -> None:
        if self.backend == "faster_whisper":
            try:
                from faster_whisper import WhisperModel

                logger.info(
                    "Loading faster-whisper %s device=%s compute=%s",
                    self.model_size,
                    self.device,
                    self.compute_type,
                )
                self._fw = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                return
            except Exception as e:
                logger.warning("faster-whisper unavailable (%s); falling back to openai-whisper", e)
                self.backend = "openai_whisper"

        import whisper

        logger.info("Loading openai-whisper %s", self.model_size)
        self._ow = whisper.load_model(self.model_size)

    def transcribe_array(self, audio_16k: np.ndarray) -> AsrResult:
        audio_16k = np.asarray(audio_16k, dtype=np.float32)
        if self._fw is not None:
            segments, info = self._fw.transcribe(
                audio_16k,
                language=self.language,
                vad_filter=True,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=False,
            )
            parts: List[str] = []
            logprobs: List[float] = []
            no_speech: List[float] = []
            for seg in segments:
                parts.append(seg.text.strip())
                if seg.avg_logprob is not None:
                    logprobs.append(float(seg.avg_logprob))
                if seg.no_speech_prob is not None:
                    no_speech.append(float(seg.no_speech_prob))
            text = " ".join(p for p in parts if p).strip()
            text = re.sub(r"\s+", " ", text)
            return AsrResult(
                text=text,
                language=getattr(info, "language", self.language) or self.language,
                avg_logprob=float(np.mean(logprobs)) if logprobs else -99.0,
                no_speech_prob=float(np.mean(no_speech)) if no_speech else 1.0,
                compression_ratio=float(getattr(info, "compression_ratio", 0.0) or 0.0),
                model_name=f"faster-whisper-{self.model_size}",
            )

        assert self._ow is not None
        result = self._ow.transcribe(
            audio_16k,
            language=self.language,
            fp16=False,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        text = re.sub(r"\s+", " ", (result.get("text") or "").strip())
        # openai-whisper aggregates; pull segment stats when present
        segs = result.get("segments") or []
        logprobs = [float(s["avg_logprob"]) for s in segs if "avg_logprob" in s]
        no_speech = [float(s["no_speech_prob"]) for s in segs if "no_speech_prob" in s]
        return AsrResult(
            text=text,
            language=result.get("language") or self.language,
            avg_logprob=float(np.mean(logprobs)) if logprobs else -99.0,
            no_speech_prob=float(np.mean(no_speech)) if no_speech else 1.0,
            compression_ratio=float(result.get("compression_ratio") or 0.0),
            model_name=f"openai-whisper-{self.model_size}",
        )

    def transcribe_file(self, wav_path: str, target_sr: int = 16000) -> AsrResult:
        audio, _ = librosa.load(wav_path, sr=target_sr, mono=True)
        return self.transcribe_array(audio)


def arabic_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    ar = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    letters = sum(1 for c in text if c.isalpha() or ("\u0600" <= c <= "\u06FF"))
    return ar / max(1, letters)


def light_qc(
    text: str,
    duration: float,
    asr: Optional[AsrResult] = None,
    *,
    min_chars: int = 4,
    min_duration: float = 1.8,
    max_duration: float = 12.0,
    min_ar_ratio: float = 0.25,
    max_no_speech: float = 0.65,
    min_avg_logprob: float = -1.2,
) -> Tuple[bool, List[str]]:
    """
    Light automatic QC. Returns (pass, reasons).

    Designed to drop silence, music-ish dumps, and empty/garbage ASR — not to
    certify Darja orthography.
    """
    reasons: List[str] = []
    t = (text or "").strip()
    if len(t) < min_chars:
        reasons.append("too_short_text")
    if duration < min_duration:
        reasons.append("too_short_audio")
    if duration > max_duration:
        reasons.append("too_long_audio")
    # Reject pure noise tokens / brackets-only
    if re.fullmatch(r"[\W_\d]+", t or ""):
        reasons.append("non_linguistic")
    if arabic_char_ratio(t) < min_ar_ratio and not re.search(r"[A-Za-z]{3,}", t):
        # Allow FR-heavy code-switch if Latin words present; else require Arabic.
        reasons.append("low_arabic_ratio")
    # Collapse repeated char spam (hhhhhhh / ااااا)
    if re.search(r"(.)\1{8,}", t):
        reasons.append("char_spam")
    if asr is not None:
        if asr.no_speech_prob > max_no_speech:
            reasons.append("high_no_speech")
        if asr.avg_logprob < min_avg_logprob:
            reasons.append("low_avg_logprob")
    return (len(reasons) == 0, reasons)


def qc_row(row: Dict[str, Any], asr: Optional[AsrResult] = None) -> Dict[str, Any]:
    """Attach QC fields to a manifest row (does not drop)."""
    ok, reasons = light_qc(
        row.get("text") or "",
        float(row.get("duration") or 0.0),
        asr,
    )
    out = dict(row)
    out["qc_pass"] = ok
    out["qc_reasons"] = reasons
    if asr is not None:
        out["asr_model"] = asr.model_name
        out["asr_avg_logprob"] = asr.avg_logprob
        out["asr_no_speech_prob"] = asr.no_speech_prob
        out["asr_language"] = asr.language
    return out
