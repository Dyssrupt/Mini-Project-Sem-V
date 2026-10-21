"""pipeline.py -- audio -> transcript -> entities -> type + urgency -> structured incident.
Uses the project's own modules (sangli_nlp, ner_eval, classify_*), so keep backend/ next to them.
"""
import math, os, pickle, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import sangli_nlp as S, ner_eval as E, classify_eval as C

DISPATCH = {"fire": ["Fire brigade"], "medical": ["Ambulance"], "accident": ["Ambulance", "Police"], "crime": ["Police"]}


def derive(d):
    """Operator overrides win over the model; priority flag = high/critical (activity diagram: 'Flag as Priority Dispatch')."""
    t = d.get("type_override") or d.get("type"); u = d.get("urgency_override") or d.get("urgency")
    d.update(final_type=t, final_urgency=u, priority=u in ("high", "critical"), dispatch=DISPATCH.get(t, []))
    return d


class TfidfClassifier:
    """TF-IDF + logistic regression (the Week-8 baseline: ties DistilBERT on urgency, instant to load, no torch needed).
    Final model is fitted on train+val; the test split is never used."""

    def __init__(self, data_dir):
        cache = ROOT / "models" / "tfidf_cls.pkl"
        if cache.exists():
            self.m = pickle.load(open(cache, "rb")); return
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        data = E.load(data_dir); rows = data["train"] + data["val"]; ex = [r for r in rows if r["transcript_asr_noisy"] != r["transcript_clean"]]
        X = [r["transcript_clean"] for r in rows] + [r["transcript_asr_noisy"] for r in ex]; self.m = {}
        for task, key, c in (("type", "emergency_type", 3), ("urgency", "urgency_label", 0.3)):
            y = [r[key] for r in rows] + [r[key] for r in ex]
            self.m[task] = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2), LogisticRegression(C=c, max_iter=3000)).fit(X, y)
        cache.parent.mkdir(exist_ok=True); pickle.dump(self.m, open(cache, "wb"))

    def predict(self, text):
        out = []
        for task, labels in (("type", C.TYPES), ("urgency", C.LEVELS)):
            p = self.m[task].predict_proba([text])[0]; cls = list(self.m[task].classes_)
            out.append({l: float(p[cls.index(l)]) for l in labels})
        return out


class BertClassifier:
    """Fine-tuned DistilBERT from classify_bert.py (needs torch + transformers; slower on CPU). CLASSIFIER=bert"""

    def __init__(self, model_name="distilbert-base-uncased", model_dir=ROOT / "models" / "bert_cls"):
        import torch, classify_bert as B
        from transformers import AutoTokenizer
        self.torch, self.B = torch, B; self.tok = AutoTokenizer.from_pretrained(str(model_dir)); self.model = B.TwoHead(model_name)
        self.model.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location="cpu")); self.model.eval()

    def predict(self, text):
        with self.torch.no_grad():
            b = self.B.enc(self.tok, [text], 128); lt, lu = self.model(b["input_ids"], b["attention_mask"])
        pt, pu = lt.softmax(-1)[0].tolist(), lu.softmax(-1)[0].tolist()
        return dict(zip(C.TYPES, pt)), dict(zip(C.LEVELS, pu))


class Pipeline:
    def __init__(self, data_dir=ROOT / "output_v3", classifier="tfidf", whisper_model="small"):
        data = E.load(data_dir); gaz = S.Gazetteer()
        self.extractor = S.IncidentExtractor(S.RuleTagger.from_training_rows(gaz, data["train"]), gaz)   # final NER = rule tagger
        self.clf = BertClassifier() if classifier == "bert" else TfidfClassifier(data_dir)
        self.whisper_model, self._whisper, self._prompt = whisper_model, None, None
        self.test_rows = data["test"]

    def transcribe(self, path):
        import transcribe as T
        if self._whisper is None:
            from faster_whisper import WhisperModel
            self._whisper = WhisperModel(self.whisper_model, device="cpu", compute_type="int8"); self._prompt = T.build_prompt(T.load_place_names())
        segs, _ = self._whisper.transcribe(T.load_audio(path), beam_size=5, language="en", initial_prompt=self._prompt, vad_filter=True)
        return " ".join(s.text.strip() for s in segs)

    def analyse(self, text, cautious=0.0):
        """cautious>0 shifts the urgency decision toward higher levels (measured: under-triage 22% -> 8% at b=0.75, over-triage 19% -> 53%)."""
        ner = self.extractor.extract(text); tp, up = self.clf.predict(text)
        scores = [math.log(up[l] + 1e-9) + cautious * i for i, l in enumerate(C.LEVELS)]
        urgency = C.LEVELS[max(range(4), key=scores.__getitem__)]; typ = max(tp, key=tp.get)
        loc = ner["location"]
        return derive({
            "transcript": text, "type": typ, "type_probs": tp, "urgency": urgency, "urgency_probs": up,
            "uncertain": max(up.values()) < 0.5, "location": loc, "keywords": ner["keywords"],
            "victim_count": ner["victim_count"], "time": ner["time"],
            "entities": [{k: e[k] for k in ("text", "label", "start", "end") if k in e} | ({"canonical": e["canonical"]} if e.get("canonical") else {}) for e in ner["entities"]],
        })
