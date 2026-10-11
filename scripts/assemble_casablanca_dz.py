"""
Recovered from transcript shell heredoc (msg 82).
Ephemeral M1 seed/ASR assembly — was not a committed scripts/ file.
"""

import pyarrow.parquet as pq
import soundfile as sf
import io, os, json, numpy as np
from huggingface_hub import hf_hub_download

out = '/workspace/repos/feeef-tts/data/seed/casablanca_dz'
os.makedirs(out+'/wavs', exist_ok=True)
files = [
  'Algeria/validation-00000-of-00002.parquet',
  'Algeria/validation-00001-of-00002.parquet',
  'Algeria/test-00000-of-00002.parquet',
  'Algeria/test-00001-of-00002.parquet',
]
manifest = []
total = 0.0
for f in files:
    local = hf_hub_download('UBC-NLP/Casablanca', f, repo_type='dataset')
    table = pq.read_table(local)
    cols = table.column_names
    print(f, 'cols', cols, 'rows', table.num_rows)
    data = table.to_pydict()
    for i in range(table.num_rows):
        text = (data['transcription'][i] or '').strip()
        dur = float(data['duration'][i])
        if not text or dur < 1.5 or dur > 12.0:
            continue
        if text in ('ضوضاء',) or text.startswith('['):
            continue
        audio_obj = data['audio'][i]
        # audio is struct with bytes/path
        if isinstance(audio_obj, dict):
            raw = audio_obj.get('bytes')
            path_hint = audio_obj.get('path')
        else:
            raw = None
            path_hint = None
        if not raw:
            continue
        arr, sr = sf.read(io.BytesIO(raw))
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        seg_id = data['seg_id'][i]
        gender = data.get('gender', [None]*table.num_rows)[i]
        split = 'validation' if 'validation' in f else 'test'
        fname = f"{split}_{seg_id}.wav"
        path = os.path.join(out, 'wavs', fname)
        sf.write(path, arr, sr)
        manifest.append({
            'audio_filepath': path,
            'text': text,
            'duration': round(float(len(arr)/sr), 3),
            'speaker_id': f"casablanca_{gender or 'unk'}",
            'dialect': 'Algerian',
            'dialect_note': 'Casablanca Algeria split; not Algiers-tagged; often TV/drama conversational register',
            'gender': gender,
            'split_source': split,
            'license': 'CC-BY-NC-ND-4.0',
            'allowed_use': 'research_eval_only',
            'source': 'UBC-NLP/Casablanca',
            'source_url': 'https://huggingface.co/datasets/UBC-NLP/Casablanca',
            'seg_id': seg_id,
        })
        total += dur
        if total >= 45*60:
            break
    if total >= 45*60:
        break

with open(os.path.join(out, 'manifest.jsonl'), 'w', encoding='utf-8') as f:
    for e in manifest:
        f.write(json.dumps(e, ensure_ascii=False)+'\n')
print(f'Casablanca kept {len(manifest)} clips, {total/60:.1f} min')
