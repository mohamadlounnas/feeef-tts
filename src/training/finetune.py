"""
Dataset loader and fine-tune entrypoints for Algerian Darja TTS.

Simulated ``random.uniform`` loss / empty checkpoints were removed in M1.
Real F5 training lives in :mod:`src.training.f5_finetune` (upstream CFM loss).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from src.dataset.preprocess import reject_empty_transcripts, require_rights_fields

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AlgerianDarjaSpeechDataset(Dataset):
    """
    Dataset loader for Algerian Darja voice fine-tuning.

    Reads JSON / JSONL manifests. Empty transcripts are rejected at load time.
    """

    def __init__(
        self,
        manifest_file: str,
        sample_rate: int = 24000,
        require_rights: bool = True,
        allowed_use_filter: Optional[List[str]] = None,
    ):
        self.sample_rate = sample_rate
        self.data: List[Dict[str, Any]] = []

        if os.path.exists(manifest_file):
            with open(manifest_file, "r", encoding="utf-8") as f:
                if manifest_file.endswith(".jsonl"):
                    raw = [json.loads(line) for line in f if line.strip()]
                else:
                    raw = json.load(f)
            raw = reject_empty_transcripts(raw)
            if require_rights:
                for row in raw:
                    require_rights_fields(row)
            if allowed_use_filter:
                allowed = set(allowed_use_filter)
                raw = [r for r in raw if r.get("allowed_use") in allowed]
            self.data = raw

        logger.info("Loaded %d speech items from %s", len(self.data), manifest_file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        return {
            "audio_filepath": item.get("audio_filepath", ""),
            "text": item.get("text", ""),
            "speaker_id": item.get("speaker_id", "dz_algiers_01"),
            "emotion": item.get("emotion", "neutral"),
            "duration": item.get("duration", 0.0),
            "license": item.get("license", ""),
            "allowed_use": item.get("allowed_use", ""),
            "source": item.get("source", ""),
            "split": item.get("split", "train"),
        }


class AlgerianTTSFineTuner:
    """
    Thin facade: points operators at the real F5 entrypoint.

    ``train_epoch`` still refuses simulation. Call
    ``python -m src.training.f5_finetune`` (or :func:`launch_f5_finetune`) instead.
    """

    def __init__(
        self,
        model_name: str = "f5tts",
        checkpoint_dir: str = "checkpoints",
        learning_rate: float = 1e-5,
        batch_size: int = 4,
    ):
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.lr = learning_rate
        self.batch_size = batch_size
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(
            "AlgerianTTSFineTuner facade ready (model=%s). "
            "Use src.training.f5_finetune for real training.",
            model_name,
        )

    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        raise NotImplementedError(
            "Simulated fine-tuning was removed. Run real F5 training via:\n"
            "  python -m src.training.f5_finetune --manifests <commercial.jsonl> --smoke\n"
            "  # GPU full train:\n"
            "  python -m src.training.f5_finetune --manifests ... --train\n"
            f"(model={self.model_name}, epoch={epoch}, batches={len(dataloader)})"
        )

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], state_dict: Optional[Dict] = None) -> str:
        if not state_dict:
            raise ValueError(
                "Refusing to write empty state_dict checkpoint. Pass real model weights."
            )
        import torch

        ckpt_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_algerian_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_name": self.model_name,
                "metrics": metrics,
                "dialect": "Algiers",
                "state_dict": state_dict,
            },
            ckpt_path,
        )
        logger.info("Checkpoint saved to %s", ckpt_path)
        return ckpt_path

    def launch_f5_finetune(self, argv: Optional[List[str]] = None) -> int:
        """Delegate to :mod:`src.training.f5_finetune`."""
        from src.training.f5_finetune import main as f5_main

        return f5_main(argv)


if __name__ == "__main__":
    tuner = AlgerianTTSFineTuner(model_name="f5tts")
    logger.info("Fine-tuner module loaded — real path: python -m src.training.f5_finetune")
