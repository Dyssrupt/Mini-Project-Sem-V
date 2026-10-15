"""
Validate the v3 dataset. Run:  python3 validate_dataset.py [path/to/output_v3]
Checks: entity offsets (clean + ASR-noisy), word-boundary alignment, spaCy token alignment (if spaCy is
installed), pronoun clashes, formatting glitches, duplicates, scenario leakage across splits, label balance,
and shortcut tests (closing line only / location only must be near chance for urgency).
"""
import json, re, sys, collections
from pathlib import Path
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "output_v3"
rows = [json.loads(l) for l in open(D / "emergency_calls_v3_full.jsonl", encoding="utf-8")]
fail = 0
def report(name, bad, ex=None):
    global fail
    print(("PASS " if not bad else "FAIL ") + f"{name}" + (f"  ({bad})" if bad else ""))
    if bad: fail += 1; print("     e.g.", ex)

# 1. offsets
for key, ek in (("transcript_clean", "entities"), ("transcript_asr_noisy", "entities_asr_noisy")):
    bad = ex = 0; ex = None
    for r in rows:
        t = r[key]; prev = -1
        for e in r[ek]:
            if t[e["start"]:e["end"]] != e["text"] or e["start"] < prev: bad += 1; ex = ex or (r["id"], e)
            prev = e["end"]
    report(f"entity offsets exact + ordered [{key}]", bad, ex)
# 2. word-boundary
bad = 0; ex = None
for r in rows:
    t = r["transcript_clean"]
    for e in r["entities"]:
        a = t[e["start"] - 1] if e["start"] else " "; z = t[e["end"]] if e["end"] < len(t) else " "
        if a.isalnum() or z.isalnum(): bad += 1; ex = ex or (r["id"], e)
report("entities start/end on word boundaries", bad, ex)
# 3. spaCy alignment (optional)
try:
    import spacy; nlp = spacy.blank("en"); bad = 0; ex = None
    for r in rows[:3000]:
        doc = nlp.make_doc(r["transcript_clean"])
        for e in r["entities"]:
            if doc.char_span(e["start"], e["end"]) is None: bad += 1; ex = ex or (r["id"], e)
    report("spaCy token alignment (first 3000 rows)", bad, ex)
except ImportError:
    print("SKIP spaCy token alignment (spaCy not installed here -- run this script locally)")
# 4. text glitches
pats = {"male+female pronoun in one call": r"\b(he|his|him)\b.*\b(she|her)\b|\b(she|her)\b.*\b(he|his|him)\b",
        "double space": r"  ", "space before punctuation": r" [,.!?]", "'sorry, I mean'": r"sorry, I mean",
        "'We are ... ago'": r"We are [^.]*\b(ago|just now)\b", "Room N/House number + building name": r"(House number|Room) \d+, [A-Z]\w+ (Apartment|Residency|Heights|Tower|Niwas)"}
for name, p in pats.items():
    hits = [r["transcript_clean"] for r in rows if re.search(p, r["transcript_clean"], re.I if "pronoun" in name else 0)]
    # a call may legitimately mention two people (e.g. "my husband ... my daughter"), so pronoun check is advisory
    if "pronoun" in name:
        print(f"INFO {name}: {len(hits)} calls (advisory -- review a sample; may be two different people)"); 
        if hits: print("     e.g.", hits[0])
    else: report(name, len(hits), hits[:1])
# 5. duplicates & leakage
c = collections.Counter(r["transcript_clean"] for r in rows)
report("no duplicate transcripts", sum(v > 1 for v in c.values()))
sc = collections.defaultdict(set)
for r in rows: sc[r["scenario_id"]].add(r["split"])
report("each scenario lives in exactly one split", sum(len(v) > 1 for v in sc.values()))
print("INFO distinct scenarios per split:", {s: len({r['scenario_id'] for r in rows if r['split'] == s}) for s in ("train", "val", "test")})
sc_score = collections.defaultdict(set)
for r in rows: sc_score[r["scenario_id"]].add(r["urgency_score"])
report("urgency_score fixed per scenario", sum(len(v) > 1 for v in sc_score.values()))
# 6. balance
for sp in ("train", "val", "test"):
    sub = [r for r in rows if r["split"] == sp]
    print(f"INFO {sp}: n={len(sub)}  type={dict(collections.Counter(r['emergency_type'] for r in sub))}  urgency={dict(collections.Counter(r['urgency_label'] for r in sub))}")
# 7. shortcut tests
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    split = lambda s: [r for r in rows if r["split"] == s]
    tr, va, te = split("train"), split("val"), split("test")
    def last_sentence(t): return re.split(r"(?<=[.!?]) ", t.strip())[-1]
    def loc_text(r): return r["transcript_clean"][r["location_phrase"]["start"]:r["location_phrase"]["end"]]
    def run(name, f, y, show=True):
        m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=2000))
        m.fit([f(r) for r in tr], [r[y] for r in tr])
        accs = [(m.predict([f(r) for r in s]) == [r[y] for r in s]).mean() for s in (va, te)]
        print(f"  {name:<34} val={accs[0]:.1%}  test={accs[1]:.1%}"); return accs
    print("\nBaselines (TF-IDF + LogReg, scenario-held-out)  [chance: type 25%, urgency 25%]")
    run("type  <- full transcript", lambda r: r["transcript_clean"], "emergency_type")
    run("urgency <- full transcript", lambda r: r["transcript_clean"], "urgency_label")
    print("Shortcut checks (urgency should be near/below ~40% for these):")
    run("urgency <- closing sentence only", lambda r: last_sentence(r["transcript_clean"]), "urgency_label")
    run("urgency <- location phrase only", loc_text, "urgency_label")
    run("type    <- location phrase only", loc_text, "emergency_type")
except ImportError:
    print("SKIP baselines (scikit-learn not installed)")
print("\nRESULT:", "ALL CHECKS PASSED" if not fail else f"{fail} CHECK(S) FAILED"); sys.exit(1 if fail else 0)
