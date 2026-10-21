"""
ner_eval.py -- evaluate a tagger on the held-out test scenarios.
    python3 ner_eval.py --tagger rule                       # rule-based baseline (no spaCy needed)
    python3 ner_eval.py --tagger spacy --model models/sangli_ner
    python3 ner_eval.py --tagger hybrid --model models/sangli_ner
    python3 ner_eval.py --tagger combo  --model models/sangli_ner   # rules + spaCy only for VICTIM_COUNT
    add  --split val  to compare on validation data (choose the config there, report test once)
Reports (clean text AND ASR-noisy text):
  * strict entity P/R/F1 per label (span AND label must match exactly)
  * KEYWORD generalisation: recall on keyword forms never seen in training vs seen
  * gazetteer linking accuracy on the noisy text (does "Vishram Bagh" resolve to Vishrambag?)
  * end-to-end area resolution accuracy (extractor output vs the call's true area)
"""
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path
import sangli_nlp as S

HERE = Path(__file__).parent
LABELS = ["AREA", "ROAD", "LANDMARK", "BUILDING", "CITY", "TIME_REF", "VICTIM_COUNT", "EMERGENCY_KEYWORD"]


def load(data_dir):
    rows = [json.loads(l) for l in open(Path(data_dir) / "emergency_calls_v3_full.jsonl", encoding="utf-8")]
    return {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}


def gold(r, noisy): return {(e["start"], e["end"], e["label"]) for e in r["entities_asr_noisy" if noisy else "entities"]}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(tagger, rows, noisy, seen_kw=None):
    tp, fp, fn = Counter(), Counter(), Counter(); kw_hit = Counter(); kw_tot = Counter()
    for r in rows:
        text = r["transcript_asr_noisy" if noisy else "transcript_clean"]
        g, p = gold(r, noisy), set(tagger.tag(text))
        for s in g & p: tp[s[2]] += 1
        for s in p - g: fp[s[2]] += 1
        for s in g - p: fn[s[2]] += 1
        if seen_kw is not None:
            for a, b, l in g:
                if l == "EMERGENCY_KEYWORD":
                    k = "seen" if text[a:b].lower() in seen_kw else "unseen"
                    kw_tot[k] += 1; kw_hit[k] += (a, b, l) in p
    per = {l: dict(zip("prf", prf(tp[l], fp[l], fn[l])), n=tp[l] + fn[l]) for l in LABELS}
    allp = dict(zip("prf", prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))))
    kw = {k: {"recall": kw_hit[k] / kw_tot[k] if kw_tot[k] else None, "n": kw_tot[k]} for k in ("seen", "unseen")} if seen_kw is not None else None
    return {"overall": allp, "per_label": per, "keyword_generalisation": kw}


def linking_accuracy(extractor, rows):
    """Gold place spans (noisy text) -> canonical name; compared with the clean gold name. Isolates the gazetteer."""
    ok = tot = 0; bad = Counter()
    for r in rows:
        c, n = r["entities"], r["entities_asr_noisy"]
        if len(c) != len(n): continue
        for ec, en in zip(c, n):
            if ec["label"] not in S.LABEL_KINDS or ec["label"] == "CITY": continue
            hit, _ = extractor.gaz.match(en["text"], en["label"]); tot += 1
            want = re.sub(r"^the ", "", ec["text"].lower())
            good = bool(hit) and re.sub(r"^the ", "", hit["name"].lower()) == want
            ok += good
            if not good: bad[(en["text"], hit["name"] if hit else None, ec["text"])] += 1
    return ok / max(tot, 1), tot, bad.most_common(8)


def area_resolution(extractor, rows, noisy):
    ok = 0
    for r in rows:
        out = extractor.extract(r["transcript_asr_noisy" if noisy else "transcript_clean"])
        ok += out["location"]["area"] == r["primary_area"]
    return ok / len(rows)


def build_tagger(kind, model, data):
    gaz = S.Gazetteer(); rule = S.RuleTagger.from_training_rows(gaz, data["train"])
    if kind == "rule": return gaz, rule
    sp = S.SpacyTagger(model)
    return gaz, {"spacy": sp, "hybrid": S.HybridTagger(sp, rule), "combo": S.ComboTagger(sp, rule)}[kind]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tagger", choices=["rule", "spacy", "hybrid", "combo"], default="rule")
    ap.add_argument("--model", default=str(HERE / "models" / "sangli_ner"))
    ap.add_argument("--data", default=str(HERE / "output_v3")); ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    data = load(a.data); rows = data[a.split]; gaz, tagger = build_tagger(a.tagger, a.model, data)
    ext = S.IncidentExtractor(tagger, gaz)
    seen = {e["text"].lower() for r in data["train"] for e in r["entities"] if e["label"] == "EMERGENCY_KEYWORD"}
    res = {"tagger": a.tagger, "split": a.split, "n_calls": len(rows),
           "n_distinct_scenarios": len({r["scenario_id"] for r in rows})}
    for noisy in (False, True):
        key = "asr_noisy" if noisy else "clean"; res[key] = evaluate(tagger, rows, noisy, seen)
        res[key]["area_resolution_acc"] = round(area_resolution(ext, rows, noisy), 4)
        o = res[key]["overall"]
        print(f"\n== {a.tagger.upper()} | {a.split} | {key} text  (n={len(rows)} calls, {res['n_distinct_scenarios']} scenarios)"
              f"   P={o['p']:.3f} R={o['r']:.3f} F1={o['f']:.3f}   area-resolution={res[key]['area_resolution_acc']:.1%}")
        print(f"   {'label':<19}{'P':>7}{'R':>7}{'F1':>7}{'gold':>7}")
        for l in LABELS:
            m = res[key]["per_label"][l]; print(f"   {l:<19}{m['p']:7.3f}{m['r']:7.3f}{m['f']:7.3f}{m['n']:7d}")
        k = res[key]["keyword_generalisation"]
        print(f"   keyword recall  seen-in-train: {k['seen']['recall']:.1%} (n={k['seen']['n']})   "
              f"NEVER-seen: {k['unseen']['recall'] if k['unseen']['recall'] is None else format(k['unseen']['recall'], '.1%')} (n={k['unseen']['n']})")
    acc, n, bad = linking_accuracy(ext, rows)
    res["gazetteer_linking_on_noisy_text"] = {"accuracy": round(acc, 4), "n_place_spans": n, "worst": bad}
    print(f"\n== Gazetteer linking on ASR-noisy gold place spans: {acc:.1%} of {n}")
    for b in bad: print("   miss:", b)
    out = a.out or f"ner_results_{a.tagger}_{a.split}.json"
    json.dump(res, open(HERE / out, "w"), indent=2, default=float); print("\nsaved", out)


if __name__ == "__main__":
    main()
