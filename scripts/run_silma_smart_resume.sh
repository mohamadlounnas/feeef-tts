#!/bin/zsh
# Resume SILMA MPS FT from model_last with Algerian-preferring smart seed.
set -euo pipefail
export PATH="/opt/homebrew/bin:/bin:/usr/bin:/usr/sbin:/sbin:$PATH"
cd /Users/mohamad/Desktop/projects/feeef-tts-m3
source .venv/bin/activate
export HF_HUB_DISABLE_XET=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

mkdir -p logs ckpts/silma_darja

MAX_UPDATES="${MAX_UPDATES:-7000}"
SAVE_EVERY="${SAVE_EVERY:-50}"
BATCH_FRAMES="${BATCH_FRAMES:-256}"

echo "=== SMART SILMA RESUME $(date) max_updates=$MAX_UPDATES ==="
echo "free disk: $(df -h /System/Volumes/Data | tail -1)"

# Keep Mac awake; resume-safe; Algerian smart manifest
caffeinate -dims python -u -m src.training.silma_finetune \
  --train-mps --resume \
  --prebuilt-manifest \
  --commercial-manifest data/seed/smart_train_manifest.jsonl \
  --dataset-name darja_smart \
  --max-updates "$MAX_UPDATES" \
  --batch-frames "$BATCH_FRAMES" \
  --save-every "$SAVE_EVERY" \
  --epochs 40 \
  --pretrain ckpts/silma_weights/model.pt \
  --save-dir ckpts/silma_darja \
  2>&1 | tee -a logs/silma_smart_train.log

echo "=== SMART TRAIN SCRIPT EXITED $(date) ==="
