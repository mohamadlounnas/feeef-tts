"""Unit tests for preprocess rights rules, text processor, and trainer honesty."""

import json
import os

import numpy as np
import pytest
import soundfile as sf

from src.dataset.preprocess import AlgerianAudioPreprocessor, reject_empty_transcripts
from src.models.tts_engine import AlgerianDarjaTextProcessor, AlgerianTTSEngine
from src.training.finetune import AlgerianDarjaSpeechDataset, AlgerianTTSFineTuner


def test_text_processor_normalization():
    processor = AlgerianDarjaTextProcessor()
    text_with_tag = "[happy] واش راك 3في، لاباس؟"
    clean_text, emotion = processor.normalize_darja_text(text_with_tag)
    assert emotion == "happy"
    assert clean_text == "واش راك عفي، لاباس؟"

    plain_text = "صحا خويا بزاف شباب"
    clean_text_2, emotion_2 = processor.normalize_darja_text(plain_text)
    assert emotion_2 == "neutral"
    assert clean_text_2 == "صحا خويا بزاف شباب"


def test_audio_preprocessor_requires_text_for_train(tmp_path):
    sr_orig = 16000
    t = np.linspace(0, 2.0, sr_orig * 2, endpoint=False)
    dummy_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    input_wav = tmp_path / "test_input.wav"
    sf.write(str(input_wav), dummy_signal, sr_orig)

    preprocessor = AlgerianAudioPreprocessor(target_sr=24000)
    output_dir = tmp_path / "processed"

    # Empty text → no train rows
    empty = preprocessor.process_file(
        str(input_wav),
        output_dir=str(output_dir / "empty"),
        speaker_id="algiers_speaker_01",
        text="",
        license="CC-BY-4.0",
        allowed_use="research_and_commercial_with_attribution",
        source="unit-test",
    )
    assert empty == []

    manifests = preprocessor.process_file(
        str(input_wav),
        output_dir=str(output_dir / "ok"),
        speaker_id="algiers_speaker_01",
        emotion="happy",
        text="واش راك خويا",
        license="CC-BY-4.0",
        allowed_use="research_and_commercial_with_attribution",
        source="unit-test",
        source_url="https://example.com",
        transcript_status="human",
    )
    assert len(manifests) > 0
    assert os.path.exists(manifests[0]["audio_filepath"])
    assert manifests[0]["sample_rate"] == 24000
    assert manifests[0]["text"] == "واش راك خويا"
    assert manifests[0]["license"] == "CC-BY-4.0"


def test_reject_empty_transcripts():
    rows = [
        {"audio_filepath": "a.wav", "text": "مرحبا", "license": "x", "allowed_use": "y", "source": "z"},
        {"audio_filepath": "b.wav", "text": "", "license": "x", "allowed_use": "y", "source": "z"},
        {"audio_filepath": "c.wav", "text": "   ", "license": "x", "allowed_use": "y", "source": "z"},
    ]
    kept = reject_empty_transcripts(rows)
    assert len(kept) == 1
    assert kept[0]["audio_filepath"] == "a.wav"


def test_mms_engine_or_explicit_fallback():
    # Prefer real MMS when network/weights available; else allow explicit test fallback.
    try:
        engine = AlgerianTTSEngine(backend_model="mms-ara", allow_harmonic_fallback=False)
        result = engine.synthesize(text="السلام عليكم", speaker_id="algiers_01")
        assert result["backend"] == "mms-ara"
    except RuntimeError:
        engine = AlgerianTTSEngine(backend_model="mms-ara", allow_harmonic_fallback=True)
        result = engine.synthesize(text="السلام عليكم", speaker_id="algiers_01")
        assert result["backend"] == "harmonic_fallback"
    assert isinstance(result["audio"], np.ndarray)
    assert len(result["audio"]) > 0
    assert result["duration"] > 0


def test_finetuner_refuses_simulation(tmp_path):
    manifest_file = tmp_path / "train_manifest.json"
    data = [
        {
            "audio_filepath": str(tmp_path / "seg1.wav"),
            "text": "واش راك خويا",
            "duration": 1.5,
            "speaker_id": "spk1",
            "emotion": "neutral",
            "license": "CC-BY-4.0",
            "allowed_use": "research_and_commercial_with_attribution",
            "source": "unit-test",
        }
    ]
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    dataset = AlgerianDarjaSpeechDataset(str(manifest_file))
    assert len(dataset) == 1

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=1)
    tuner = AlgerianTTSFineTuner(model_name="f5tts", checkpoint_dir=str(tmp_path / "checkpoints"))
    with pytest.raises(NotImplementedError):
        tuner.train_epoch(dataloader, epoch=1)

    with pytest.raises(ValueError):
        tuner.save_checkpoint(epoch=1, metrics={"loss": 0.0}, state_dict=None)
