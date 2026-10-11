"""
Recovered from transcript shell heredoc (msg 68).
Ephemeral M1 seed/ASR assembly — was not a committed scripts/ file.
"""

from huggingface_hub import hf_hub_download, list_repo_files
import os, shutil
repo = "FatimahEmadEldin/cafe-algerian-codeswitch-speech"
out = "/workspace/repos/feeef-tts/data/seed/cafe_raw"
os.makedirs(out, exist_ok=True)
files = list_repo_files(repo, repo_type="dataset")
wanted = [f for f in files if 'CAFE-small-no-overlap' in f]
print(f"Downloading {len(wanted)} CAFE-small-no-overlap files...")
for i, f in enumerate(wanted):
    local = hf_hub_download(repo, f, repo_type="dataset")
    # flatten into out with basename (handle collisions)
    base = os.path.basename(f)
    dest = os.path.join(out, base)
    if os.path.exists(dest) and not os.path.samefile(local, dest):
        # keep unique name
        dest = os.path.join(out, f.replace('/', '__'))
    if not os.path.exists(dest):
        shutil.copy2(local, dest)
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(wanted)}")
print("done CAFE")
print(os.listdir(out)[:10], "count", len(os.listdir(out)))
