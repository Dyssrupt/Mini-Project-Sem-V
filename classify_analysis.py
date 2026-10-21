"""
classify_analysis.py -- three diagnostics for the urgency classifier (sklearn only; ~2 min).
 1. 5-fold CV GROUPED BY SCENARIO over all 320 scenarios (steadier than one 48-scenario split)
 2. Learning curve over the number of TRAINING SCENARIOS (is data, not model size, the limit?)
 3. Triage operating point: shift predictions toward higher urgency, trading over- for under-triage
"""
import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
import classify_eval as C, ner_eval as E

HERE = Path(__file__).parent
data = E.load(HERE / "output_v3"); allrows = data["train"] + data["val"] + data["test"]
noisy = lambda rows: [r["transcript_asr_noisy"] for r in rows]


def fit(rows, key="urgency_label"):
    ex = [r for r in rows if r["transcript_asr_noisy"] != r["transcript_clean"]]
    X = [r["transcript_clean"] for r in rows] + noisy(ex); y = [r[key] for r in rows] + [r[key] for r in ex]
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2), LogisticRegression(C=1, max_iter=3000)).fit(X, y)


out = {}
# 1 -------------------------------------------------------------------------------------------------
groups = np.array([r["scenario_id"] for r in allrows])
print(f"1) 5-fold CV grouped by scenario, all {len(set(groups))} scenarios, tested on ASR-noisy text")
pred = {k: [None] * len(allrows) for k in ("urgency_label", "emergency_type")}
for tr, te in GroupKFold(5).split(allrows, groups=groups):
    for key in pred:
        m = fit([allrows[i] for i in tr], key); p = m.predict(noisy([allrows[i] for i in te]))
        for i, v in zip(te, p): pred[key][i] = v
for key, labels, ordn in (("emergency_type", C.TYPES, False), ("urgency_label", C.LEVELS, True)):
    r = C.evaluate([x[key] for x in allrows], pred[key], labels, list(groups), ordn); out["cv_" + key] = r
    print(C.line(key, r))
r = out["cv_urgency_label"]; print("   urgency per-level recall:", {l: r["per_class"][l]["recall"] for l in C.LEVELS})
# 2 -------------------------------------------------------------------------------------------------
print("\n2) Learning curve: urgency accuracy vs number of TRAINING scenarios (evaluated on val+test = 96 scenarios, 5 random subsets each)")
ev = data["val"] + data["test"]; scen = sorted({r["scenario_id"] for r in data["train"]}); combos = {}
for s in scen: combos.setdefault(s.rsplit("/", 1)[0], []).append(s)
rng = np.random.default_rng(0); curve = []
for frac in (0.25, 0.5, 0.75, 1.0):
    accs = []
    for _ in range(5 if frac < 1 else 1):
        keep = set()
        for c, ss in combos.items(): keep.update(rng.choice(ss, max(1, round(frac * len(ss))), replace=False))
        m = fit([r for r in data["train"] if r["scenario_id"] in keep]); accs.append((m.predict(noisy(ev)) == [r["urgency_label"] for r in ev]).mean())
    n = sum(max(1, round(frac * len(ss))) for ss in combos.values()); curve.append({"scenarios": n, "acc_mean": round(float(np.mean(accs)), 4), "acc_sd": round(float(np.std(accs)), 4)})
    print(f"   {n:3d} training scenarios -> {np.mean(accs):.1%}  (sd {np.std(accs):.1%})")
out["learning_curve"] = curve
# 3 -------------------------------------------------------------------------------------------------
print("\n3) Triage operating point: score = log P(level) + b*level_index   (b=0 is the normal classifier). Choose b on VALIDATION, report test.")
m = fit(data["train"]); lv = list(m.classes_); order = [lv.index(l) for l in C.LEVELS]


def at(rows, b):
    P = np.log(m.predict_proba(noisy(rows))[:, order] + 1e-9) + b * np.arange(4)
    return C.evaluate([r["urgency_label"] for r in rows], [C.LEVELS[i] for i in P.argmax(1)], C.LEVELS, [r["scenario_id"] for r in rows], True)


print(f"   {'b':>5} | {'val acc':>8} {'val UNDER':>10} {'val over':>9} | {'test acc':>8} {'test UNDER':>11} {'test over':>10}")
tab = []
for b in (0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    v, t = at(data["val"], b), at(data["test"], b); tab.append((b, v, t))
    print(f"   {b:5.2f} | {v['accuracy']:8.1%} {v['under_triage']:10.1%} {v['over_triage']:9.1%} | {t['accuracy']:8.1%} {t['under_triage']:11.1%} {t['over_triage']:10.1%}")
pick = next((x for x in tab if x[1]["under_triage"] <= 0.10), tab[-1]); b = pick[0]
print(f"\n   chosen on validation (first b with val under-triage <= 10%): b={b}")
print("   TEST at chosen b:", C.line("urgency test", pick[2]))
out["triage_shift"] = {"chosen_b": b, "val": {k: pick[1][k] for k in ("accuracy", "under_triage", "over_triage")}, "test": {k: pick[2][k] for k in ("accuracy", "under_triage", "over_triage", "under_triage_ci95")}}
json.dump(out, open(HERE / "classify_results_analysis.json", "w"), indent=2); print("\nsaved classify_results_analysis.json")

# 4 -------------------------------------------------------------------------------------------------
import re, collections
print("\n4) Safety floor from NER keywords: if the call contains a keyword that (in TRAINING data) marks high/critical calls with >=90% precision")
print("   across >=3 distinct scenarios, raise the prediction to at least HIGH. Forms are learned from the training fold only.")


def floor_forms(rows, min_scen=3, prec=0.9):
    cnt = collections.defaultdict(lambda: [0, 0, set()])
    for r in rows:
        hi = r["urgency_label"] in ("high", "critical")
        for f in {e["text"].lower() for e in r["entities"] if e["label"] == "EMERGENCY_KEYWORD"}:
            c = cnt[f]; c[0] += 1; c[1] += hi; c[2].add(r["scenario_id"])
    return sorted(f for f, (n, h, s) in cnt.items() if len(s) >= min_scen and h / n >= prec)


def apply_floor(preds, rows, forms):
    rx = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(f) for f in forms) + r")(?![\w-])", re.I) if forms else None
    return [("high" if C.LEVELS.index(p) < 2 else p) if (rx and rx.search(r["transcript_asr_noisy"])) else p for p, r in zip(preds, rows)]


forms = floor_forms(data["train"]); print(f"   {len(forms)} floor keywords from the training split, e.g.: {forms[:14]}")
res = {}
for name, rows in (("val", data["val"]), ("test", data["test"])):
    base = list(m.predict(noisy(rows))); fl = apply_floor(base, rows, forms)
    for tag, p in (("without floor", base), ("with floor", fl)):
        r = C.evaluate([x["urgency_label"] for x in rows], p, C.LEVELS, [x["scenario_id"] for x in rows], True); res[(name, tag)] = r
        print(f"   {name:<5}{C.line(tag, r)}")
cvp = {"base": [None] * len(allrows), "floor": [None] * len(allrows)}
for tr, te in GroupKFold(5).split(allrows, groups=groups):
    trr = [allrows[i] for i in tr]; mm = fit(trr); ter = [allrows[i] for i in te]; p = list(mm.predict(noisy(ter))); f = apply_floor(p, ter, floor_forms(trr))
    for i, a, b in zip(te, p, f): cvp["base"][i], cvp["floor"][i] = a, b
for tag in ("base", "floor"):
    r = C.evaluate([x["urgency_label"] for x in allrows], cvp[tag], C.LEVELS, list(groups), True); out["cv_floor_" + tag] = r
    print(f"   CV-320{C.line('floor' if tag == 'floor' else 'no floor', r)}  UNDER 95% CI {r['under_triage_ci95']}")
json.dump(out, open(HERE / "classify_results_analysis.json", "w"), indent=2, default=float); print("saved classify_results_analysis.json")
