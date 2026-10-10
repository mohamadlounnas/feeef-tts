import os
import json
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import soundfile as sf
import scipy.signal as signal
import librosa

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AlgerianAudioPreprocessor:
    """
    Audio preprocessing pipeline for Algerian Darja TTS training.
    Includes silence detection/splitting, sample rate conversion (24kHz/22.05kHz),
    loudness peak normalization, and metadata manifest creation.
    """

    def __init__(self, target_sr: int = 24000, min_silence_len_sec: float = 0.4, top_db: int = 30):
        self.target_sr = target_sr
        self.min_silence_len_sec = min_silence_len_sec
        self.top_db = top_db

    def normalize_loudness(self, audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
        """
        Normalizes peak amplitude of the audio array to target_peak.
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio * (target_peak / max_val)
        return audio

    def resample_audio(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """
        Resamples audio signal to self.target_sr using librosa.
        """
        if orig_sr == self.target_sr:
            return audio
        return librosa.resample(y=audio, orig_sr=orig_sr, target_sr=self.target_sr)

    def split_on_silence(
        self, audio: np.ndarray, sr: int, min_segment_len_sec: float = 1.0, max_segment_len_sec: float = 12.0
    ) -> List[np.ndarray]:
        """
        Splits audio signal into contiguous speech segments based on non-silent intervals.
        """
        intervals = librosa.effects.split(audio, top_db=self.top_db)
        segments = []

        for start, end in intervals:
            duration = (end - start) / sr
            if duration >= min_segment_len_sec and duration <= max_segment_len_sec:
                segment = audio[start:end]
                segments.append(segment)
            elif duration > max_segment_len_sec:
                # Sub-split long segments
                max_samples = int(max_segment_len_sec * sr)
                for sub_start in range(start, end, max_samples):
                    sub_end = min(sub_start + max_samples, end)
                    if (sub_end - sub_start) / sr >= min_segment_len_sec:
                        segments.append(audio[sub_start:sub_end])

        return segments

    def process_file(
        self, input_wav_path: str, output_dir: str, speaker_id: str = "dz_algiers_01", emotion: str = "neutral"
    ) -> List[Dict[str, str]]:
        """
        Processes an raw WAV file: resamples, normalizes, splits, saves output segments, and returns manifest list.
        """
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(input_wav_path):
            raise FileNotFoundError(f"Input file not found: {input_wav_path}")

        audio, sr = librosa.load(input_wav_path, sr=None, mono=True)
        audio = self.resample_audio(audio, sr)
        audio = self.normalize_loudness(audio)

        segments = self.split_on_silence(audio, self.target_sr)
        base_name = os.path.splitext(os.path.basename(input_wav_path))[0]

        processed_manifests = []
        for idx, seg in enumerate(segments):
            seg_filename = f"{base_name}_seg_{idx:03d}.wav"
            seg_path = os.path.join(output_dir, seg_filename)
            sf.write(seg_path, seg, self.target_sr)

            duration = len(seg) / self.target_sr
            manifest_entry = {
                "audio_filepath": seg_path,
                "duration": round(duration, 3),
                "speaker_id": speaker_id,
                "dialect": "Algiers",
                "emotion": emotion,
                "sample_rate": self.target_sr,
                "text": ""  # Placeholder for transcript / Whisper output
            }
            processed_manifests.append(manifest_entry)

        logger.info(f"Processed '{input_wav_path}' into {len(processed_manifests)} segments.")
        return processed_manifests

if __name__ == "__main__":
    processor = AlgerianAudioPreprocessor()
    logger.info("Preprocessing module operational.")
