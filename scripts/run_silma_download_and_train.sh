#!/bin/zsh
set -euo pipefail
export PATH="/opt/homebrew/bin:/bin:/usr/bin:/usr/sbin:/sbin:$PATH"
cd /Users/mohamad/Desktop/projects/feeef-tts-m3
source .venv/bin/activate
export HF_HUB_DISABLE_XET=1

BLOB_DIR="$HOME/.cache/huggingface/hub/models--silma-ai--silma-tts/blobs"
HASH=f43256d0b78b8803c638aed0875da5a4b372b4a784690a0156e5baff14f7336c
INC="$BLOB_DIR/${HASH}.38141e96.incomplete"
EXPECTED=2603209272
FINAL_BLOB="$BLOB_DIR/$HASH"
LOCAL=ckpts/silma_weights/model.pt
SNAP="$HOME/.cache/huggingface/hub/models--silma-ai--silma-tts/snapshots/226dd7a65cadf51f9a6dbe3953fc89003b3844d5"
mkdir -p ckpts/silma_weights logs data/f5_datasets ckpts/silma_darja

if [[ ! -f "$LOCAL" ]] || [[ $(stat -f%z "$LOCAL") -ne $EXPECTED ]]; then
  # Prefer continuing incomplete
  if [[ -f "$FINAL_BLOB" ]] && [[ $(stat -f%z "$FINAL_BLOB") -eq $EXPECTED ]]; then
    echo "blob already complete"
  else
    # Find largest incomplete for this hash
    CAND=$(ls -S "$BLOB_DIR"/${HASH}*incomplete 2>/dev/null | head -1 || true)
    if [[ -n "${CAND:-}" ]]; then
      INC="$CAND"
    fi
    echo "Resuming download into $INC size=$(stat -f%z "$INC" 2>/dev/null || echo 0)"
    while true; do
      SZ=$(stat -f%z "$INC" 2>/dev/null || echo 0)
      if [[ "$SZ" -ge "$EXPECTED" ]]; then break; fi
      URL=$(python - <<'PY'
from huggingface_hub import hf_hub_url, get_hf_file_metadata
import os
os.environ['HF_HUB_DISABLE_XET']='1'
url=hf_hub_url('silma-ai/silma-tts','model.pt')
meta=get_hf_file_metadata(url)
print(meta.location or url)
PY
)
      echo "curl resume from $SZ"
      curl -L --fail --retry 30 --retry-delay 2 -C - -o "$INC" "$URL" || true
      sleep 2
      SZ2=$(stat -f%z "$INC" 2>/dev/null || echo 0)
      echo "after curl size=$SZ2"
      if [[ "$SZ2" -eq "$EXPECTED" ]]; then break; fi
      if [[ "$SZ2" -le "$SZ" ]]; then echo "no progress; retry in 5s"; sleep 5; fi
    done
    mv "$INC" "$FINAL_BLOB"
  fi
  mkdir -p "$SNAP"
  ln -sfn "$FINAL_BLOB" "$SNAP/model.pt"
  rm -f "$LOCAL"
  ln "$FINAL_BLOB" "$LOCAL" 2>/dev/null || cp -c "$FINAL_BLOB" "$LOCAL"
fi

python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
v=Path(hf_hub_download('silma-ai/silma-tts','vocab.txt'))
Path('data/f5_datasets/silma_vocab.txt').write_text(v.read_text(encoding='utf-8'), encoding='utf-8')
print('vocab ok', len(v.read_text().splitlines()))
print('model', Path('ckpts/silma_weights/model.pt').stat().st_size)
PY

echo "=== STARTING SILMA MPS TRAIN ==="
# Keep Mac awake; resume-safe; long run
caffeinate -dims python -u -m src.training.silma_finetune \
  --train-mps --resume \
  --max-updates 2000 \
  --batch-frames 256 \
  --save-every 50 \
  --epochs 20 \
  --pretrain ckpts/silma_weights/model.pt \
  --save-dir ckpts/silma_darja \
  2>&1 | tee -a logs/silma_train.log

echo "=== TRAIN SCRIPT EXITED ==="
date
