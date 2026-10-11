"""Device selection for CUDA / Apple MPS fine-tune path."""

from __future__ import annotations

from src.training.f5_finetune import select_train_device


def test_select_train_device_returns_known_label():
    device = select_train_device()
    assert device in {"cuda", "mps", "cpu"}


def test_select_train_device_prefer_cpu():
    assert select_train_device("cpu") == "cpu"
