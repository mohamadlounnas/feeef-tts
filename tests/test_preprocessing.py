import os
import tempfile
import numpy as np
import soundfile as sf
import pytest
from src.preprocessing.audio_processor import AudioProcessor, DarijaTextNormalizer
from src.preprocessing.transcribe import DarijaTranscriber

def test_darija_text_normalizer():
    normalizer = DarijaTextNormalizer()
    raw = "أهلاً و سهلاً، واش راك خويا؟ ڨاع لاباس!"
    normalized = normalizer.normalize(raw)
    assert "أ" not in normalized  # 'أ' should be converted to 'ا'
    assert "ااهلا" in normalized or "اهلا" in normalized
    assert "ڨ" in normalized or "ڨ" in normalized

def test_audio_processor():
    processor = AudioProcessor(target_sample_rate=24000, max_duration=5.0, min_duration=1.0)
    # Generate 10s synthetic sine wave audio
    sr = 16000
    t = np.linspace(0, 10, sr * 10)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_file = os.path.join(tmp_dir, "sample.wav")
        sf.write(input_file, audio, sr)

        resampled, target_sr = processor.load_and_resample(input_file)
        assert target_sr == 24000
        assert len(resampled) == 24000 * 10

        chunks = processor.chunk_audio(resampled, target_sr)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 5.0 * 24000 + 100

def test_transcriber_packaging():
    transcriber = DarijaTranscriber()
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_wav = os.path.join(tmp_dir, "raw.wav")
        out_dir = os.path.join(tmp_dir, "processed")
        sf.write(raw_wav, np.zeros(24000 * 4, dtype=np.float32), 24000)

        meta = {
            "id": "test_sample_01",
            "speaker": "algiers_female_01",
            "emotion": "happy",
            "transcript": "واش راكي اختي، الحمد لله"
        }

        records = transcriber.transcribe_and_package(raw_wav, out_dir, meta)
        assert len(records) > 0
        assert records[0]["speaker"] == "algiers_female_01"
        assert records[0]["emotion"] == "happy"
        assert os.path.exists(records[0]["audio_path"])
