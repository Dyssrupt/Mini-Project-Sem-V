"""
Rebuild manifest.csv from curated_batch.jsonl, matching against whatever
.mp3 files already exist in audio_out/. Doesn't call the ElevenLabs API.

Run in the same folder as curated_batch.jsonl and audio_out/:
    python3 rebuild_manifest.py
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
BATCH_FILE = HERE / "curated_batch.jsonl"
AUDIO_DIR = HERE / "audio_out"
OUT_MANIFEST = AUDIO_DIR / "manifest.csv"

rows = [json.loads(l) for l in open(BATCH_FILE, encoding="utf-8")]

found, missing = [], []
for r in rows:
    fname = f"{r['id']}.mp3"
    if (AUDIO_DIR / fname).exists():
        found.append((fname, r))
    else:
        missing.append(fname)

AUDIO_DIR.mkdir(exist_ok=True)
with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "id", "emergency_type", "urgency_label", "urgency_score", "text"])
    for fname, r in found:
        writer.writerow([fname, r["id"], r["emergency_type"], r["urgency_label"], r["urgency_score"], r["text"]])

print(f"Rebuilt manifest.csv with {len(found)} clips.")
if missing:
    print(f"\n{len(missing)} clips from curated_batch.jsonl have no matching audio file "
          f"(not generated, or named differently):")
    for m in missing:
        print(f"  - {m}")
