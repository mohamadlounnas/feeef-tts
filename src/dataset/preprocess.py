"""
Audio preprocessing and rights-aware manifest builders for Algerian Darja TTS.

Hard rules:
- Empty ``text`` fields are rejected from training manifests.
- Every row must carry license / allowed_use / source provenance.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import librosa
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ManifestRecord:
    """One speech segment with transcript and rights metadata."""

    audio_filepath: str
    text: str
    duration: float
    speaker_id: str = "unknown"
    dialect: str = "Algiers"
    emotion: str = "neutral"
    sample_rate: int = 24000
    license: str = "unknown"
    allowed_use: str = "unknown"
    source: str = "unknown"
    source_url: str = ""
    split: str = "train"
    transcript_status: str = "unverified"
    dialect_note: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra") or {}
        d.update(extra)
        return d


def reject_empty_transcripts(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drop (and log) any row without non-empty text.

    Training on empty transcripts is a hard ban for this project.
    """
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for row in records:
        text = (row.get("text") or "").strip()
        if not text:
            dropped += 1
            continue
        row = dict(row)
        row["text"] = text
        kept.append(row)
    if dropped:
        logger.warning("Rejected %d rows with empty transcripts", dropped)
    return kept


def require_rights_fields(row: Dict[str, Any]) -> None:
    """Raise if license / allowed_use / source are missing."""
    for key in ("license", "allowed_use", "source"):
        if not str(row.get(key) or "").strip():
            raise ValueError(f"Manifest row missing required rights field '{key}': {row.get('audio_filepath')}")


class AlgerianAudioPreprocessor:
    """
    Resample → peak-normalize → silence-split → write WAVs + rights-aware manifests.

    Transcripts are never invented here. Callers must supply ``text`` (human or ASR draft)
    via ``process_file(..., text=...)`` or merge ASR results before writing train manifests.
    """

    def __init__(self, target_sr: int = 24000, min_silence_len_sec: float = 0.4, top_db: int = 30):
        self.target_sr = target_sr
        self.min_silence_len_sec = min_silence_len_sec
        self.top_db = top_db

    def normalize_loudness(self, audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio * (target_peak / max_val)
        return audio

    def resample_audio(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self.target_sr:
            return audio
        return librosa.resample(y=audio, orig_sr=orig_sr, target_sr=self.target_sr)

    def split_on_silence(
        self,
        audio: np.ndarray,
        sr: int,
        min_segment_len_sec: float = 1.0,
        max_segment_len_sec: float = 12.0,
    ) -> List[np.ndarray]:
        intervals = librosa.effects.split(audio, top_db=self.top_db)
        segments: List[np.ndarray] = []
        for start, end in intervals:
            duration = (end - start) / sr
            if min_segment_len_sec <= duration <= max_segment_len_sec:
                segments.append(audio[start:end])
            elif duration > max_segment_len_sec:
                max_samples = int(max_segment_len_sec * sr)
                for sub_start in range(start, end, max_samples):
                    sub_end = min(sub_start + max_samples, end)
                    if (sub_end - sub_start) / sr >= min_segment_len_sec:
                        segments.append(audio[sub_start:sub_end])
        return segments

    def process_file(
        self,
        input_wav_path: str,
        output_dir: str,
        speaker_id: str = "dz_algiers_01",
        emotion: str = "neutral",
        text: str = "",
        license: str = "unknown",
        allowed_use: str = "unknown",
        source: str = "unknown",
        source_url: str = "",
        dialect: str = "Algiers",
        transcript_status: str = "unverified",
        allow_empty_text: bool = False,
        split: str = "train",
    ) -> List[Dict[str, str]]:
        """
        Process one WAV into segments.

        If ``text`` is empty and ``allow_empty_text`` is False, no train-ready rows are
        emitted (empty-transcript ban). Intermediate QC rows may still be written when
        ``allow_empty_text=True`` and must not be used for training.
        """
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(input_wav_path):
            raise FileNotFoundError(f"Input file not found: {input_wav_path}")

        audio, sr = librosa.load(input_wav_path, sr=None, mono=True)
        audio = self.resample_audio(audio, sr)
        audio = self.normalize_loudness(audio)
        segments = self.split_on_silence(audio, self.target_sr)
        base_name = os.path.splitext(os.path.basename(input_wav_path))[0]

        processed_manifests: List[Dict[str, Any]] = []
        clean_text = (text or "").strip()
        if not clean_text and not allow_empty_text:
            logger.error(
                "Refusing to emit train manifest for '%s' with empty text. "
                "Provide a transcript or set allow_empty_text=True for QC-only output.",
                input_wav_path,
            )
            return []

        for idx, seg in enumerate(segments):
            seg_filename = f"{base_name}_seg_{idx:03d}.wav"
            seg_path = os.path.join(output_dir, seg_filename)
            sf.write(seg_path, seg, self.target_sr)
            duration = len(seg) / self.target_sr
            # When a single parent transcript is supplied for a multi-segment file,
            # attach it only if there is exactly one segment; otherwise leave empty
            # for ASR alignment (caller must fill before training).
            seg_text = clean_text if (clean_text and len(segments) == 1) else (clean_text if len(segments) == 1 else "")
            if len(segments) > 1 and clean_text:
                # Parent transcript does not align 1:1 — require per-segment text later.
                seg_text = ""
                transcript_status_row = "needs_per_segment_transcript"
            else:
                transcript_status_row = transcript_status if seg_text else "empty"

            record = ManifestRecord(
                audio_filepath=seg_path,
                text=seg_text,
                duration=round(duration, 3),
                speaker_id=speaker_id,
                dialect=dialect,
                emotion=emotion,
                sample_rate=self.target_sr,
                license=license,
                allowed_use=allowed_use,
                source=source,
                source_url=source_url,
                split=split,
                transcript_status=transcript_status_row,
            )
            row = record.to_dict()
            if allow_empty_text or row["text"].strip():
                if row["text"].strip():
                    require_rights_fields(row)
                processed_manifests.append(row)

        # Enforce empty-transcript ban on returned train rows
        if not allow_empty_text:
            processed_manifests = reject_empty_transcripts(processed_manifests)

        logger.info("Processed '%s' into %d segments.", input_wav_path, len(processed_manifests))
        return processed_manifests

    @staticmethod
    def write_jsonl(path: str, rows: List[Dict[str, Any]], training: bool = True) -> int:
        """Write JSONL manifest; optionally ban empty text."""
        if training:
            rows = reject_empty_transcripts(rows)
            for row in rows:
                require_rights_fields(row)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote %d rows to %s", len(rows), path)
        return len(rows)


if __name__ == "__main__":
    processor = AlgerianAudioPreprocessor()
    logger.info("Preprocessing module operational (empty-transcript ban enabled).")
