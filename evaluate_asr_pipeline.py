"""
evaluate_asr_pipeline.py -- END-TO-END check: real Whisper transcripts -> NER -> structured incident.
    python3 evaluate_asr_pipeline.py --asr asr_results_prompt_vad.json --tagger rule
    python3 evaluate_asr_pipeline.py --asr asr_results_prompt_vad.json --tagger hybrid --model models/sangli_ner
Gold = the entities of the dataset row with the same id (the script that was read aloud).
Scores per clip and overall: place names linked to the right gazetteer entry, area resolved, time, keyword recall, count.
"""
import argparse, csv, json
from pathlib import Path
import ner_eval as E, sangli_nlp as S

HERE = Path(__file__).parent


def gold_info(row, gaz):
    ents = row["entities"]; text = row["transcript_clean"]
    places = set()
    for e in ents:
        if e["label"] in S.LABEL_KINDS and e["label"] != "CITY":
            hit, _ = gaz.match(e["text"], e["label"])
            if hit: places.add(hit["name"])
    t = next((S.parse_time(e["text"]) for e in ents if e["label"] == "TIME_REF"), None)
    c = next((S.parse_count(e["text"]) for e in ents if e["label"] == "VICTIM_COUNT"), None)
    return {"places": places, "area": row["primary_area"], "time": t, "count": c,
            "keywords": {e["text"].lower() for e in ents if e["label"] == "EMERGENCY_KEYWORD"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", required=True); ap.add_argument("--tagger", default="rule", choices=["rule", "spacy", "hybrid", "combo"])
    ap.add_argument("--model", default=str(HERE / "models" / "sangli_ner")); ap.add_argument("--data", default=str(HERE / "output_v3"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    data = E.load(a.data); rows = {r["id"]: r for s in data.values() for r in s}
    gaz, tagger = E.build_tagger(a.tagger, a.model, data); ext = S.IncidentExtractor(tagger, gaz)
    results = json.load(open(a.asr, encoding="utf-8"))["results"]
    out, tot = [], dict(p_tp=0, p_fp=0, p_fn=0, area=0, time=0, time_n=0, kw_hit=0, kw_n=0, cnt=0, cnt_n=0)
    for r in results:
        g = gold_info(rows[r["id"]], gaz); x = ext.extract(r["hypothesis"])
        pred_places = {e["canonical"] for e in x["entities"] if e["label"] in S.LABEL_KINDS and e["label"] != "CITY" and e.get("canonical")}
        tp, fp, fn = len(g["places"] & pred_places), len(pred_places - g["places"]), len(g["places"] - pred_places)
        area_ok = x["location"]["area"] == g["area"]
        time_ok = None if g["time"] is None else (x["time"] is not None and x["time"]["minutes_ago"] == g["time"]["minutes_ago"])
        kw = {k.lower() for k in x["keywords"]}; kw_hit = len(g["keywords"] & kw)
        cnt_ok = None if g["count"] is None else (x["victim_count"] is not None and x["victim_count"]["count"] == g["count"]["count"])
        tot["p_tp"] += tp; tot["p_fp"] += fp; tot["p_fn"] += fn; tot["area"] += area_ok
        tot["kw_hit"] += kw_hit; tot["kw_n"] += len(g["keywords"])
        if time_ok is not None: tot["time"] += time_ok; tot["time_n"] += 1
        if cnt_ok is not None: tot["cnt"] += cnt_ok; tot["cnt_n"] += 1
        out.append({"id": r["id"], "wer": r.get("wer"), "area_ok": area_ok, "place_tp": tp, "place_fp": fp, "place_fn": fn,
                    "missing_places": sorted(g["places"] - pred_places), "wrong_places": sorted(pred_places - g["places"]),
                    "time_ok": time_ok, "keywords_found": f"{kw_hit}/{len(g['keywords'])}", "missing_keywords": sorted(g["keywords"] - kw),
                    "count_ok": cnt_ok, "location_summary": x["location"]["summary"], "needs_confirmation": x["location"]["needs_confirmation"]})
    n = len(out); pr = tot["p_tp"] / max(tot["p_tp"] + tot["p_fp"], 1); rc = tot["p_tp"] / max(tot["p_tp"] + tot["p_fn"], 1)
    print(f"{a.tagger.upper()} on {a.asr}: {n} clips")
    print(f"  area resolved correctly : {tot['area']}/{n} = {tot['area']/n:.1%}")
    print(f"  place names linked      : precision {pr:.1%}  recall {rc:.1%}  (gold {tot['p_tp']+tot['p_fn']} place mentions)")
    print(f"  time reference correct  : {tot['time']}/{tot['time_n']}")
    print(f"  keyword recall          : {tot['kw_hit']}/{tot['kw_n']} = {tot['kw_hit']/max(tot['kw_n'],1):.1%}")
    print(f"  victim count correct    : {tot['cnt']}/{tot['cnt_n']}")
    print("  clips with any place/area/keyword problem:")
    for o in out:
        if not o["area_ok"] or o["place_fn"] or o["place_fp"] or o["missing_keywords"] or o["time_ok"] is False or o["count_ok"] is False:
            print(f"   {o['id']}: area_ok={o['area_ok']} missing_places={o['missing_places']} wrong_places={o['wrong_places']} "
                  f"missing_kw={o['missing_keywords']} time_ok={o['time_ok']} count_ok={o['count_ok']}")
    name = a.out or f"asr_pipeline_{a.tagger}_{Path(a.asr).stem}"
    json.dump(out, open(HERE / f"{name}.json", "w"), indent=2, default=str)
    with open(HERE / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"  saved {name}.json/.csv")


if __name__ == "__main__":
    main()
