"""
Emergency Call Audio Generator (ElevenLabs eleven_v3)
------------------------------------------------------
Run this LOCALLY, not in a restricted sandbox. It reads your API key from
an environment variable (never hardcode it here), and uses the eleven_v3
model, which supports inline emotional direction tags like [panicking],
[gasps], [sobbing] -- exactly what's already baked into curated_batch.jsonl.

Setup:
    pip install requests
    export ELEVENLABS_API_KEY="your-key-here"   # do NOT paste it into this file
    python3 generate_audio.py

Output:
    audio_out/<id>.mp3  for each clip
    audio_out/manifest.csv  mapping filename -> transcript/type/urgency

Free tier is 10,000 chars/month with no top-up -- this batch uses ~7,250,
so you have some room for a re-run if a couple of calls fail.
"""

import os
import sys
import csv
import json
import time
from pathlib import Path

import requests

API_KEY = os.environ.get("ELEVENLABS_API_KEY")
if not API_KEY:
    sys.exit("ERROR: Set the ELEVENLABS_API_KEY environment variable first "
              "(export ELEVENLABS_API_KEY=\"...\"). Do not hardcode it here.")

BASE_URL = "https://api.elevenlabs.io/v1"
HEADERS = {"xi-api-key": API_KEY}

BATCH_FILE = Path(__file__).parent / "curated_batch.jsonl"
OUT_DIR = Path(__file__).parent / "audio_out"
OUT_DIR.mkdir(exist_ok=True)

MODEL_ID = "eleven_v3"

# Voice settings: lower stability = more expressive/variable delivery,
# which suits panicked/distressed speech better than a flat, stable read.
VOICE_SETTINGS = {"stability": 0.35, "similarity_boost": 0.75, "style": 0.6}


def pick_voice():
    """List available voices and let the user choose, or auto-pick the first premade one."""
    resp = requests.get(f"{BASE_URL}/voices", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    voices = resp.json().get("voices", [])
    if not voices:
        sys.exit("No voices returned from your account -- check your ElevenLabs dashboard.")

    print("Available voices:")
    for i, v in enumerate(voices):
        print(f"  [{i}] {v['name']}  (id={v['voice_id']})  labels={v.get('labels', {})}")

    choice = input(f"\nPick a voice index [default 0]: ").strip()
    idx = int(choice) if choice else 0
    return voices[idx]["voice_id"], voices[idx]["name"]


def generate_clip(text, voice_id):
    url = f"{BASE_URL}/text-to-speech/{voice_id}"
    payload = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": VOICE_SETTINGS,
    }
    resp = requests.post(url, headers={**HEADERS, "Content-Type": "application/json"},
                          json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
    return resp.content


def main():
    rows = [json.loads(l) for l in open(BATCH_FILE, encoding="utf-8")]
    total_chars = sum(len(r["text"]) for r in rows)
    print(f"Loaded {len(rows)} clips, {total_chars} characters total.\n")

    voice_id, voice_name = pick_voice()
    print(f"\nUsing voice: {voice_name}\n")

    manifest_path = OUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(["filename", "id", "emergency_type", "urgency_label", "urgency_score", "text"])

        chars_used = 0
        for i, row in enumerate(rows, 1):
            out_path = OUT_DIR / f"{row['id']}.mp3"
            print(f"[{i}/{len(rows)}] {row['id']} ({row['emergency_type']}/{row['urgency_label']}) "
                  f"-- {len(row['text'])} chars ... ", end="", flush=True)
            try:
                audio = generate_clip(row["text"], voice_id)
                out_path.write_bytes(audio)
                chars_used += len(row["text"])
                writer.writerow([out_path.name, row["id"], row["emergency_type"],
                                  row["urgency_label"], row["urgency_score"], row["text"]])
                print(f"OK (running total: {chars_used} chars)")
            except Exception as e:
                print(f"FAILED: {e}")
            time.sleep(0.5)  # be gentle on rate limits

    print(f"\nDone. Audio in {OUT_DIR}/, manifest at {manifest_path}")


if __name__ == "__main__":
    main()
