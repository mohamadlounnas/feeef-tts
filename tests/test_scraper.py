import os
import json
import tempfile
import pytest
from src.data_collection.scraper import DarijaDataScraper

def test_scraper_manifest_ingestion():
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_data = [
            {
                "id": "algiers_sample_01",
                "url": "https://example.com/audio1.mp3",
                "speaker": "algiers_male_1",
                "region": "algiers",
                "emotion": "excited"
            }
        ]
        manifest_path = os.path.join(tmp_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        scraper = DarijaDataScraper(output_dir=os.path.join(tmp_dir, "raw"))
        results = scraper.ingest_manifest(manifest_path)

        assert len(results) == 1
        assert results[0]["id"] == "algiers_sample_01"
        assert results[0]["speaker"] == "algiers_male_1"
        assert results[0]["emotion"] == "excited"
