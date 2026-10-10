import os
import json
import pytest
import numpy as np
import soundfile as sf
from torch.utils.data import DataLoader

from src.dataset.scraper import AlgerianAudioScraper
from src.dataset.preprocess import AlgerianAudioPreprocessor
from src.models.tts_engine import AlgerianDarjaTextProcessor, AlgerianTTSEngine
from src.training.finetune import AlgerianDarjaSpeechDataset, AlgerianTTSFineTuner


def test_text_processor_normalization():
    processor = AlgerianDarjaTextProcessor()

    # Test emotion tag parsing
    text_with_tag = "[happy] واش راك خويا، لاباس؟"
    clean_text, emotion = processor.normalize_darja_text(text_with_tag)
    assert emotion == "happy"
    assert clean_text == "واش راك خويا، لاباس؟"

    # Test default neutral emotion
    plain_text = "صحا خويا بزاف شباب"
    clean_text_2, emotion_2 = processor.normalize_darja_text(plain_text)
    assert emotion_2 == "neutral"
    assert clean_text_2 == "صحا خويا بزاف شباب"


def test_audio_preprocessor(tmp_path):
    # Create dummy WAV audio file (1 second sine wave at 16000Hz)
    sr_orig = 16000
    t = np.linspace(0, 2.0, sr_orig * 2, endpoint=False)
    dummy_signal = 0.5 * np.sin(2 * np.pi * 440 * t)

    input_wav = tmp_path / "test_input.wav"
    sf.write(str(input_wav), dummy_signal, sr_orig)

    preprocessor = AlgerianAudioPreprocessor(target_sr=24000)
    output_dir = tmp_path / "processed"

    manifests = preprocessor.process_file(
        str(input_wav),
        output_dir=str(output_dir),
        speaker_id="algiers_speaker_01",
        emotion="happy"
    )

    assert len(manifests) > 0
    assert os.path.exists(manifests[0]["audio_filepath"])
    assert manifests[0]["sample_rate"] == 24000
    assert manifests[0]["emotion"] == "happy"


def test_tts_engine_synthesis():
    engine = AlgerianTTSEngine(backend_model="f5tts", sample_rate=24000)
    result = engine.synthesize(text="[excited] يا خوي صحا بزاف!", speaker_id="algiers_01")

    assert "audio" in result
    assert isinstance(result["audio"], np.ndarray)
    assert len(result["audio"]) > 0
    assert result["emotion"] == "excited"
    assert result["sample_rate"] == 24000
    assert result["duration"] > 0.5


def test_finetuning_pipeline(tmp_path):
    # Create sample json manifest
    manifest_file = tmp_path / "train_manifest.json"
    data = [
        {
            "audio_filepath": str(tmp_path / "seg1.wav"),
            "text": "واش راك خويا",
            "duration": 1.5,
            "speaker_id": "spk1",
            "emotion": "neutral"
        },
        {
            "audio_filepath": str(tmp_path / "seg2.wav"),
            "text": "يعطيك الصحة بزاف",
            "duration": 2.1,
            "speaker_id": "spk1",
            "emotion": "happy"
        }
    ]
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    dataset = AlgerianDarjaSpeechDataset(str(manifest_file))
    assert len(dataset) == 2

    dataloader = DataLoader(dataset, batch_size=2)
    tuner = AlgerianTTSFineTuner(model_name="f5tts", checkpoint_dir=str(tmp_path / "checkpoints"))

    metrics = tuner.train_epoch(dataloader, epoch=1)
    assert "loss" in metrics

    ckpt_path = tuner.save_checkpoint(epoch=1, metrics=metrics)
    assert os.path.exists(ckpt_path)
