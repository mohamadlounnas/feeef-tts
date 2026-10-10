import os
import json
import tempfile
import pytest
from src.training.train import DarijaTTSTrainer

def test_trainer_manifest_preparation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = DarijaTTSTrainer(output_dir=tmp_dir, num_epochs=2)
        records = [
            {
                "audio_path": "/path/to/audio1.wav",
                "text": "صباح الخير، واش راكم يا جماعة",
                "speaker": "algiers_female_01",
                "emotion": "happy",
                "duration": 4.0
            }
        ]
        manifest_out = os.path.join(tmp_dir, "manifest.json")
        trainer.prepare_dataset_manifest(records, manifest_out)

        assert os.path.exists(manifest_out)
        with open(manifest_out, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) == 1
            assert "algiers_female_01" in data[0]["description"]
            assert "happy" in data[0]["description"]

def test_trainer_run():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = DarijaTTSTrainer(output_dir=tmp_dir, num_epochs=2)
        records = [
            {
                "audio_path": "/path/to/audio1.wav",
                "text": "واش راك خويا",
                "speaker": "algiers_male_01",
                "emotion": "neutral",
                "duration": 2.5
            }
        ]
        res = trainer.run_fine_tuning(records)
        assert res["status"] == "completed"
        assert res["total_epochs"] == 2
        assert os.path.exists(os.path.join(tmp_dir, "checkpoint_epoch_2.json"))
