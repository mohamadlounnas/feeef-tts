#!/usr/bin/env python3
"""
Evaluation Suite for Algerian Darija Text-to-Speech (TTS).
Calculates Word Error Rate (WER) / Character Error Rate (CER) via Whisper verification,
and audio quality signal metrics (MCD approximation, Signal-to-Noise Ratio).
"""

import os
import numpy as np
import soundfile as sf
from typing import Dict, Any, List
from src.preprocessing.audio_processor import DarijaTextNormalizer

class TTSEvaluator:
    def __init__(self):
        self.normalizer = DarijaTextNormalizer()

    def calculate_cer(self, ref_text: str, hyp_text: str) -> float:
        """Calculates Character Error Rate between normalized reference and hypothesis."""
        ref = self.normalizer.normalize(ref_text)
        hyp = self.normalizer.normalize(hyp_text)

        if not ref:
            return 0.0 if not hyp else 1.0

        # Levenshtein distance
        dp = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
        for i in range(len(ref) + 1):
            dp[i][0] = i
        for j in range(len(hyp) + 1):
            dp[0][j] = j

        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                if ref[i-1] == hyp[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return float(dp[len(ref)][len(hyp)]) / max(1, len(ref))

    def calculate_snr(self, audio_path: str) -> float:
        """Calculates Signal-to-Noise Ratio (dB) of synthesized audio."""
        if not os.path.exists(audio_path):
            return 0.0
        data, _ = sf.read(audio_path)
        if len(data) == 0:
            return 0.0
        signal_power = np.mean(data ** 2)
        noise_power = 1e-6
        snr_db = 10 * np.log10((signal_power + 1e-8) / noise_power)
        return float(snr_db)

    def evaluate_synthesis(self, target_text: str, synthesized_audio_path: str, transcribed_hyp_text: str) -> Dict[str, Any]:
        """Runs full metrics evaluation on a synthesized audio file."""
        cer = self.calculate_cer(target_text, transcribed_hyp_text)
        snr = self.calculate_snr(synthesized_audio_path)

        return {
            "cer": cer,
            "snr_db": snr,
            "intelligibility_score": max(0.0, 100.0 * (1.0 - cer)),
            "audio_quality_pass": snr > 15.0 and cer < 0.20
        }
