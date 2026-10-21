"""classify_baseline.py -- Week 8 baseline (the synopsis' "SVM / traditional classifier as baseline"). Needs scikit-learn only.
    python3 classify_baseline.py
Train on train (clean + ASR-noisy copies), choose C on validation, report validation and test on clean AND noisy text.
"""
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
import classify_eval as C, ner_eval as E

HERE = Path(__file__).parent
data = E.load(HERE / "output_v3")


def texts(rows, noisy): return [r["transcript_asr_noisy" if noisy else "transcript_clean"] for r in rows]


train = data["train"]; X = texts(train, False) + texts([r for r in train if r["transcript_asr_noisy"] != r["transcript_clean"]], True)
extra = [r for r in train if r["transcript_asr_noisy"] != r["transcript_clean"]]
results = {}
for task, key, labels, ordinal in (("type", "emergency_type", C.TYPES, False), ("urgency", "urgency_label", C.LEVELS, True)):
    y = [r[key] for r in train] + [r[key] for r in extra]
    best = None
    for name, mk in (("tfidf+LogReg", lambda c: LogisticRegression(C=c, max_iter=3000)), ("tfidf+LinearSVC", lambda c: LinearSVC(C=c))):
        for c in (0.3, 1, 3, 10, 30):
            m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2), mk(c)).fit(X, y)
            v = sum((m.predict(texts(data["val"], nz)) == [r[key] for r in data["val"]]).mean() for nz in (False, True)) / 2
            if best is None or v > best[0]: best = (v, name, c, m)
    v, name, c, m = best
    print(f"\n=== {task.upper()}   best on validation: {name} C={c}   (chance 25%)")
    results[task] = {"model": name, "C": c}
    for sp in ("val", "test"):
        for nz in (False, True):
            rows = data[sp]; pred = m.predict(texts(rows, nz)); tag = f"{sp} {'asr-noisy' if nz else 'clean'}"
            r = C.evaluate([x[key] for x in rows], list(pred), labels, [x["scenario_id"] for x in rows], ordinal)
            results[task][tag] = r; print(C.line(tag, r))
    print(f"  confusion (test, asr-noisy):"); C.print_confusion(results[task]["test asr-noisy"])
json.dump(results, open(HERE / "classify_results_baseline.json", "w"), indent=2)
print("\nsaved classify_results_baseline.json")
