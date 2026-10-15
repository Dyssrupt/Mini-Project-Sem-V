"""
ASR Module - Week 5/6 (Development + Testing & Tuning)
-------------------------------------------------------
Runs faster-whisper over audio_out/ clips and scores against the known scripts:
  * WER (word error rate)
  * PLACE-NAME RECALL -- of the Sangli place names that were actually said (from sangli_gazetteer.json),
    how many did Whisper get exactly / nearly / not at all. This matters more than WER for this project:
    a call with perfect WER but "Kanbhag" for "Khanbhag" still needs fuzzy gazetteer matching in the NER stage.

Setup (run LOCALLY, first run downloads model weights):
    pip install faster-whisper
    python3 transcribe.py

Week-6 tuning experiments (each run writes its own results files, compare them):
    python3 transcribe.py --tag baseline_small   --no-prompt
    python3 transcribe.py --tag prompt_small                         # default: Sangli place-name prompt
    python3 transcribe.py --tag prompt_vad       --vad               # skips long silences (helps dropped clauses)
    python3 transcribe.py --tag prompt_nocond    --no-condition      # stops a bad segment from derailing the next
    python3 transcribe.py --tag prompt_medium    --model medium
    python3 transcribe.py --model large-v3 --device cuda --compute-type float16
"""
import argparse, csv, difflib, json, re, shutil, subprocess
from pathlib import Path

HERE = Path(__file__).parent
AUDIO_DIR_DEFAULT = HERE / "audio_out"
GAZETTEER = HERE / "sangli_gazetteer.json"


def load_place_names():
    """All Sangli names that can appear in a script (areas, sub-areas, roads, chowks, landmarks)."""
    g = json.load(open(GAZETTEER, encoding="utf-8"))
    names = set()
    for area, d in g["areas"].items():
        names.add(area)
        for k in ("roads", "chowks", "landmarks", "subareas"): names.update(d[k])
    names.update(g["highways"]); names.discard("Main Road")          # too generic to score
    return sorted(names, key=len, reverse=True)


def build_prompt(names, max_chars=480):
    """Whisper only reads ~224 tokens of initial_prompt (and keeps the END if longer), so keep it short:
    areas first, then chowks/landmarks."""
    g = json.load(open(GAZETTEER, encoding="utf-8"))
    ordered = list(g["areas"]) + [c for d in g["areas"].values() for c in d["chowks"]] + \
              [l for d in g["areas"].values() for l in d["landmarks"]]
    out, n = "Emergency call from Sangli, Maharashtra. Places: ", ""
    for name in dict.fromkeys(ordered):
        if len(out) + len(n) + len(name) + 2 > max_chars: break
        n += name + ", "
    return (out + n).rstrip(", ") + "."


def strip_direction_tag(text):
    """The manifest text starts with an [emotion tag] -- a performance cue, not spoken content."""
    return re.sub(r"^\[.*?\]\s*", "", text).strip()


def normalize(text):
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def word_error_rate(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i-1][j-1] if ref[i-1] == hyp[j-1] else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m] / max(n, 1)


def compact(text):
    return "".join(normalize(text))


def place_names_in(reference, names):
    """Place names present in the reference script (longest first so 'Station Chowk' isn't also counted as 'Station')."""
    ref, found = compact(reference), []
    for nm in names:
        c = compact(nm)
        if c and c in ref and not any(c in compact(f) for f in found):
            found.append(nm)
    return found


def place_match(name, hypothesis, near=0.80):
    """'exact' = name found (ignoring spaces/case, so 'Madhav Nagar' == 'Madhavnagar'); 'near' = fuzzy match
    (e.g. 'Kanbhag' vs 'Khanbhag') -- recoverable with fuzzy matching in the NER stage; else 'miss'."""
    c, h = compact(name), compact(hypothesis)
    if c in h: return "exact"
    L = len(c)
    best = max((difflib.SequenceMatcher(None, c, h[i:i+L+1]).ratio() for i in range(max(1, len(h) - L + 1))), default=0)
    return "near" if best >= near else "miss"


def load_audio(path):
    """Decode any audio file to 16 kHz mono float32 using the ffmpeg command line, bypassing PyAV.
    (Newer PyAV versions break faster-whisper's own decoder: "open() got an unexpected keyword argument
    'metadata_errors'".) Falls back to faster-whisper's decoder if ffmpeg isn't installed."""
    if shutil.which("ffmpeg") is None:
        return str(path)
    import numpy as np
    out = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
                          "-f", "f32le", "-ac", "1", "-ar", "16000", "-"], capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {out.stderr.decode(errors='ignore')[:300]}")
    return np.frombuffer(out.stdout, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", help="tiny/base/small/medium/large-v3 (default: small)")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--audio-dir", default=str(AUDIO_DIR_DEFAULT))
    ap.add_argument("--no-prompt", action="store_true", help="disable the Sangli place-name initial_prompt (baseline)")
    ap.add_argument("--vad", action="store_true", help="enable voice-activity detection (skips long silences)")
    ap.add_argument("--no-condition", action="store_true", help="condition_on_previous_text=False")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. --tag prompt_small")
    args = ap.parse_args()
    from faster_whisper import WhisperModel                      # imported here so the helpers can be tested without it
    if shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg not found -- falling back to faster-whisper's decoder (may fail with newer PyAV). "
              "Install ffmpeg:  sudo apt install ffmpeg")

    audio_dir = Path(args.audio_dir); manifest = audio_dir / "manifest.csv"
    if not manifest.exists(): raise SystemExit(f"Can't find {manifest} -- run generate_audio.py / rebuild_manifest.py first.")
    suffix = f"_{args.tag}" if args.tag else ""
    out_json, out_csv = HERE / f"asr_results{suffix}.json", HERE / f"asr_results{suffix}.csv"

    names = load_place_names(); prompt = None if args.no_prompt else build_prompt(names)
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    print(f"Whisper {args.model} | {args.device}/{args.compute_type} | prompt={'off' if prompt is None else 'on'} "
          f"| vad={args.vad} | condition_on_prev={not args.no_condition}")
    if prompt: print("initial_prompt:", prompt)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    results, per_urgency, tally = [], {}, {"exact": 0, "near": 0, "miss": 0}
    for i, row in enumerate(rows, 1):
        path = audio_dir / row["filename"]
        if not path.exists(): print(f"[{i}/{len(rows)}] MISSING FILE: {path}"); continue
        segments, _ = model.transcribe(load_audio(path), beam_size=5, language="en", initial_prompt=prompt,
                                       vad_filter=args.vad, condition_on_previous_text=not args.no_condition)
        hyp = " ".join(s.text.strip() for s in segments)
        ref = strip_direction_tag(row["text"]); wer = word_error_rate(ref, hyp)
        said = place_names_in(ref, names); marks = {nm: place_match(nm, hyp) for nm in said}
        for v in marks.values(): tally[v] += 1
        per_urgency.setdefault(row["urgency_label"], []).append(wer)
        print(f"[{i}/{len(rows)}] {row['id']} ({row['emergency_type']}/{row['urgency_label']}) WER={wer:.2%}  places={marks}")
        results.append({"id": row["id"], "emergency_type": row["emergency_type"], "urgency_label": row["urgency_label"],
                        "reference": ref, "hypothesis": hyp, "wer": round(wer, 4), "places": json.dumps(marks)})

    n = max(len(results), 1); avg = sum(r["wer"] for r in results) / n; tot = max(sum(tally.values()), 1)
    print(f"\nOverall WER: {avg:.2%} across {len(results)} clips")
    for u, w in per_urgency.items(): print(f"  WER {u}: {sum(w)/len(w):.2%} (n={len(w)})")
    print(f"Place names said: {tot} -> exact {tally['exact']/tot:.1%} | near-miss {tally['near']/tot:.1%} | missed {tally['miss']/tot:.1%}")
    json.dump({"settings": vars(args), "overall_wer": avg, "place_names": tally, "results": results},
              open(out_json, "w", encoding="utf-8"), indent=2)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    print(f"Saved {out_json.name} and {out_csv.name}")


if __name__ == "__main__":
    main()
