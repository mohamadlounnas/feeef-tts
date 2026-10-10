#!/usr/bin/env python3
"""
Audio Processing & Normalization Module for Algerian Darija Speech Corpus.
Handles audio resampling (24kHz), mono conversion, peak/RMS normalization,
and energy/VAD chunking into optimal 3-15s audio segments.
"""

import os
import re
import numpy as np
import soundfile as sf
import scipy.signal
import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, target_sample_rate: int = 24000, max_duration: float = 15.0, min_duration: float = 2.0):
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration

    def load_and_resample(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Loads audio file, converts to mono, and resamples to target sample rate."""
        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1) # Convert stereo to mono

        if sr != self.target_sample_rate:
            # Resample signal
            num_samples = int(len(data) * self.target_sample_rate / sr)
            data = scipy.signal.resample(data, num_samples)

        # Normalize RMS / Peak
        peak = np.max(np.abs(data))
        if peak > 0:
            data = data / peak * 0.95

        return data.astype(np.float32), self.target_sample_rate

    def chunk_audio(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """Chunks long audio based on energy silent gaps into target clip durations."""
        chunk_samples_max = int(self.max_duration * sr)
        chunk_samples_min = int(self.min_duration * sr)

        if len(audio) <= chunk_samples_max:
            return [audio]

        chunks = []
        start = 0
        while start < len(audio):
            end = start + chunk_samples_max
            if end >= len(audio):
                if len(audio) - start >= chunk_samples_min:
                    chunks.append(audio[start:])
                break

            # Search for lowest energy frame in the last 2 seconds of the chunk window
            search_start = max(start + chunk_samples_min, end - int(2 * sr))
            search_window = audio[search_start:end]
            frame_size = int(0.05 * sr)
            num_frames = len(search_window) // frame_size

            if num_frames > 0:
                energies = [np.sum(search_window[i*frame_size:(i+1)*frame_size]**2) for i in range(num_frames)]
                min_frame_idx = np.argmin(energies)
                split_point = search_start + min_frame_idx * frame_size
            else:
                split_point = end

            chunks.append(audio[start:split_point])
            start = split_point

        return chunks

class DarijaTextNormalizer:
    """
    Normalizer for Algerian Darija Text (Algiers Dialect).
    Standardizes orthography, cleans diacritics, standardizes character variants,
    and preserves dialect specific constructs (e.g. ڤ/ڨ for hard G).
    """
    def __init__(self):
        # Diacritics regex
        self.arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670]')

    def normalize(self, text: str) -> str:
        if not text:
            return ""

        # Remove diacritics
        text = self.arabic_diacritics.sub('', text)

        # Normalize Alef variants
        text = re.sub(r'[\u0622\u0623\u0625]', '\u0627', text) # أ, إ, آ -> ا

        # Normalize Ya / Alef Maksura
        text = re.sub(r'\u0649', '\u064A', text) # ى -> ي

        # Normalize Teh Marbuta to Heh if context permits or standard
        text = re.sub(r'\u0629', '\u0647', text) # ة -> ه

        # Preserve / Standardize Algiers Darija specific letter ڨ / ڤ
        text = re.sub(r'\u06A4', '\u06AF', text) # ڤ -> ڨ

        # Clean multiple whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text
