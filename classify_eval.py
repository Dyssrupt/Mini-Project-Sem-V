"""classify_eval.py -- shared metrics for the type / urgency classifiers (Week 8-9).
Beyond accuracy it reports what matters for dispatch:
  under_triage = share of truly HIGH/CRITICAL calls predicted LOW/MEDIUM   (the dangerous mistake)
  over_triage  = share of truly LOW/MEDIUM calls predicted HIGH/CRITICAL   (wastes resources)
  within_one   = predicted level is at most one step from the true level
and a 95% CI from a bootstrap over SCENARIOS (calls built from the same scenario are not independent,
so a plain per-call CI would be far too narrow: the test set has only 48 distinct scenarios).
"""
import numpy as np

TYPES = ["fire", "medical", "accident", "crime"]
LEVELS = ["low", "medium", "high", "critical"]


def _ci(num, den, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed); k = len(num); vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, k, k); d = den[pick].sum()
        if d: vals.append(num[pick].sum() / d)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)] if vals else None


def evaluate(y_true, y_pred, labels, scen, ordinal=False):
    t = np.array([labels.index(x) for x in y_true]); p = np.array([labels.index(x) for x in y_pred]); scen = np.array(scen)
    k = len(labels); cm = np.zeros((k, k), int)
    for a, b in zip(t, p): cm[a, b] += 1
    per, f1s = {}, []
    for i, l in enumerate(labels):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0; per[l] = {"precision": round(float(pr), 4), "recall": round(float(rc), 4), "f1": round(float(f), 4), "n": int(tp + fn)}; f1s.append(f)
    ids = sorted(set(scen.tolist())); groups = [np.where(scen == s)[0] for s in ids]
    ok = (t == p).astype(float)
    out = {"n": int(len(t)), "n_scenarios": len(ids), "accuracy": round(float((t == p).mean()), 4),
           "accuracy_ci95": _ci(np.array([ok[g].sum() for g in groups]), np.array([len(g) for g in groups])),
           "macro_f1": round(float(np.mean(f1s)), 4), "per_class": per, "labels": labels, "confusion": cm.tolist()}
    if ordinal:
        hi, lo = t >= 2, t <= 1
        out["within_one"] = round(float((abs(t - p) <= 1).mean()), 4); out["mae_levels"] = round(float(abs(t - p).mean()), 3)
        out["under_triage"] = round(float((p[hi] <= 1).mean()), 4)
        out["under_triage_ci95"] = _ci(np.array([((p <= 1) & hi)[g].sum() for g in groups]), np.array([hi[g].sum() for g in groups]))
        out["over_triage"] = round(float((p[lo] >= 2).mean()), 4)
    return out


def line(name, r):
    s = f"  {name:<26} acc={r['accuracy']:.1%} [{r['accuracy_ci95'][0]:.1%}-{r['accuracy_ci95'][1]:.1%}]  macro-F1={r['macro_f1']:.3f}"
    if "under_triage" in r: s += f"  within-1={r['within_one']:.1%}  UNDER-triage={r['under_triage']:.1%}  over-triage={r['over_triage']:.1%}"
    return s


def print_confusion(r):
    print("     true \\ pred  " + "".join(f"{l:>10}" for l in r["labels"]))
    for l, row in zip(r["labels"], r["confusion"]): print(f"     {l:<13}" + "".join(f"{v:>10}" for v in row))
