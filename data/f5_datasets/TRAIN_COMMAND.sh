#!/usr/bin/env bash
set -euo pipefail
# Commercial Darja F5 fine-tune — requires GPU
# Train hours ≈ 1.264
# Place prepared data where f5_tts load_dataset expects (often <f5_tts_repo>/data/darja_commercial_char)
export F5_DATA_ROOT=/workspace/repos/feeef-tts/data/f5_datasets
/usr/bin/python3 -m f5_tts.train.finetune_cli --exp_name F5TTS_v1_Base --dataset_name darja_commercial --learning_rate 1e-05 --batch_size_per_gpu 800 --batch_size_type frame --max_samples 32 --epochs 50 --num_warmup_updates 300 --save_per_updates 500 --last_per_updates 200 --tokenizer custom --finetune --tokenizer_path /workspace/repos/feeef-tts/data/f5_datasets/darja_commercial_char/vocab.txt --logger tensorboard
