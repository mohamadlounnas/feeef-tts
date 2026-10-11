"""
Recovered from transcript shell heredoc (msg 88).
Ephemeral M1 seed/ASR assembly — was not a committed scripts/ file.
"""

from huggingface_hub import hf_hub_download
import pandas as pd, soundfile as sf, os, json

out = '/workspace/repos/feeef-tts/data/seed/dvoice_ary'
os.makedirs(out+'/wavs', exist_ok=True)
meta_path = hf_hub_download('BrunoHays/DVOICEv2.0-Darija', 'metadata.csv', repo_type='dataset')
df = pd.read_csv(meta_path)
print(df.columns.tolist(), len(df))
print(df.head(2))
# filter 2-10s
df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
df = df.dropna(subset=['duration', 'words'])
df = df[(df['duration']>=2.0) & (df['duration']<=10.0) & (df['words'].astype(str).str.strip()!='')]
# Prefer train split if path has train
df['file_name'] = df['file_name'].astype(str)
train = df[df['file_name'].str.contains('train')]
pool = train if len(train) else df
# Take enough for ~20 min, diversify by sampling evenly
pool = pool.sort_values('duration')
# pick every Nth for variety
target_sec = 20*60
picked = []
acc = 0.0
step = max(1, len(pool)//400)
for i in range(0, len(pool), step):
    row = pool.iloc[i]
    picked.append(row)
    acc += float(row['duration'])
    if acc >= target_sec:
        break
print(f'picking {len(picked)} files ~{acc/60:.1f} min')

manifest = []
total = 0.0
for row in picked:
    # file_name like data/train/uuid.wav
    rel = row['file_name']
    if not rel.startswith('data/'):
        rel = 'data/train/' + os.path.basename(rel)
    local = hf_hub_download('BrunoHays/DVOICEv2.0-Darija', rel, repo_type='dataset')
    dest = os.path.join(out, 'wavs', os.path.basename(rel))
    if not os.path.exists(dest):
        import shutil
        shutil.copy2(local, dest)
    info = sf.info(dest)
    text = str(row['words']).strip()
    manifest.append({
        'audio_filepath': dest,
        'text': text,
        'duration': round(info.duration, 3),
        'speaker_id': 'dvoice_ary',
        'dialect': 'Moroccan_Darija',
        'dialect_note': 'NOT Algerian; Maghrebi conversational closest open CC-BY seed. Gap: Moroccan lexicon/prosody vs Algiers Darja.',
        'license': 'CC-BY-4.0',
        'allowed_use': 'research_and_commercial_with_attribution',
        'source': 'BrunoHays/DVOICEv2.0-Darija',
        'source_url': 'https://huggingface.co/datasets/BrunoHays/DVOICEv2.0-Darija',
        'upstream': 'https://doi.org/10.5281/zenodo.6342622',
        'audio_id': str(row.get('audio_id', os.path.basename(rel))),
    })
    total += info.duration

with open(os.path.join(out, 'manifest.jsonl'), 'w', encoding='utf-8') as f:
    for e in manifest:
        f.write(json.dumps(e, ensure_ascii=False)+'\n')
print(f'DVoice kept {len(manifest)} clips, {total/60:.1f} min')
