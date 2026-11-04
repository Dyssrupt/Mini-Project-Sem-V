"""pipeline.py -- audio -> transcript -> entities -> type + urgency -> structured incident.
Uses the project's own modules (sangli_nlp, ner_eval, classify_*), so keep backend/ next to them.
"""
import math, os, pickle, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import re
import sangli_nlp as S, ner_eval as E, classify_eval as C

DISPATCH = {
    "fire": ["Fire brigade"],
    "medical": ["Ambulance"],
    "accident": ["Ambulance", "Police"],
    "crime": ["Police"],
    "non_emergency": []
}

NON_EMERGENCY_PATTERNS = [
    # 1. Greetings, test calls, system tests
    (re.compile(r"^(hi|hello|hey|yo|namaste|test|testing|test call|testing 1 2 3|ok|okay|bye|good (morning|afternoon|evening)|anyone there|is this working|who is this)(\s+(test|testing|there|call|please|line|\d+|all))*[\s.?!]*$", re.I),
     "Call contains only a greeting or test message (no emergency details)"),
    (re.compile(r"\b(test call|testing (the|this)?\s*(line|system|phone|call|mic)?|just testing)\b", re.I),
     "Test call or system check"),

    # 2. Negation of impact, hit, injury, bleeding, harm ("didn't hit anything", "did not get hurt", "wasn't injured")
    (re.compile(r"\b(didn'?t|did not|wasn'?t|was not|not)\s+(get\s+|receive\s+|seem\s+to\s+)?(hit|hurt|injur(e|ed)?|harm(ed)?|break|damag(e|ed)?|wound(ed)?|bleed(ing)?)\b", re.I),
     "Caller confirms no injury, impact, or damage occurred"),

    # 3. Explicit statement of nobody hurt / nothing bad happened
    (re.compile(r"\b(nobody|no one|no-one)\s+(is|was|got)?\s+(hurt|injur(ed)?|harm(ed)?|in danger|dying)\b", re.I),
     "Caller indicates nobody was hurt or in danger"),
    (re.compile(r"\b(nothing|not (much|anything))\s+(serious|bad|happened|is broken|hurts?)\b", re.I),
     "Caller indicates nothing serious happened"),
    (re.compile(r"\bno\s+(injur(y|ies)|danger|problem|damage|casualt(y|ies)|harm)\b", re.I),
     "Caller reports no injuries or danger"),

    # 4. Being fine / okay / safe (including "completely fine", "doing okay", "all safe")
    (re.compile(r"\b(i think|seems|looks like)?\s*(he|she|they|we|everyone|everybody|my (child|kid|son|daughter|baby|father|mother|brother|sister|husband|wife)|someone)\s*('s|'re|\s+(is|are|looks?|seems?|got))\s+(completely|totally|absolutely|doing|just|now)?\s*(fine|okay|ok|alright|safe|unhurt|well|good|safe and sound)\b", re.I),
     "Caller indicates person is fine, unhurt, or safe"),

    # 5. False alarms / accidental calls / wrong numbers
    (re.compile(r"\b(false alarm|accidental(ly)? (call|dial|dialed|called)|pocket dial(ed)?|called by mistake|dialed by mistake|wrong number)\b", re.I),
     "False alarm or accidental call"),

    # 6. Explicitly not needing assistance / already resolved
    (re.compile(r"\b(don'?t|do not)\s+need\s+(help|assistance|any (help|assistance)|an? ambulance|the police|the fire brigade|doctor|anyone)\b", re.I),
     "Caller stated assistance is not needed"),
    (re.compile(r"\bno\s+need\s+(to send|for)\s+(help|assistance|anyone|an? ambulance|police|fire brigade)\b", re.I),
     "Caller stated no need to send emergency services"),
    (re.compile(r"\b(already (put out|resolved|handled|fixed)|under control|never mind|cancel (it|the call|the request))\b", re.I),
     "Situation already resolved or under control")
]

CRITICAL_EMERGENCY_WORDS = {
    "not breathing", "stopped breathing", "unconscious", "no pulse", "cardiac arrest",
    "heart attack", "collapsed", "stabbed", "shooting", "shot", "gun", "hostage", "explosion",
    "exploded", "bleeding heavily", "severe bleeding", "trapped", "dying", "choking"
}


def detect_non_emergency(text: str, keywords: list) -> tuple[bool, str | None]:
    t = text.strip().lower()
    if not t:
        return True, "Empty call transcript"

    words = re.findall(r"\w+", t)
    has_critical_words = any(cw in t for cw in CRITICAL_EMERGENCY_WORDS)

    for pat, desc in NON_EMERGENCY_PATTERNS:
        m = pat.search(t)
        if m and not has_critical_words:
            matched_phrase = m.group(0).strip()
            return True, f"{desc}: \"{matched_phrase}\""

    if len(words) <= 3 and not keywords:
        return True, "Too short / no emergency details provided"

    if not keywords and len(words) < 20:
        if re.search(r"\b(what('?s| is) the (time|date|weather)|lost my (keys|wallet|bag)|street light not working|power cut)\b", t):
            return True, "Non-emergency inquiry or utility complaint"

    return False, None


def derive(d):
    """Operator overrides win over the model; priority flag = high/critical (activity diagram: 'Flag as Priority Dispatch')."""
    is_non_emg = d.get("non_emergency") or d.get("type_override") == "non_emergency"
    t = d.get("type_override") or ("non_emergency" if is_non_emg else d.get("type"))
    u = d.get("urgency_override") or ("none" if is_non_emg else d.get("urgency"))

    is_priority = (u in ("high", "critical")) and (t != "non_emergency")
    dispatch = [] if t == "non_emergency" else DISPATCH.get(t, [])

    d.update(
        final_type=t,
        final_urgency=u,
        priority=is_priority,
        dispatch=dispatch,
        non_emergency=bool(is_non_emg),
        non_emergency_reason=d.get("non_emergency_reason")
    )
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
    def __init__(self, data_dir=ROOT / "output_v3", classifier="bert", tagger="combo", whisper_model="small"):
        data = E.load(data_dir); gaz = S.Gazetteer()
        rule = S.RuleTagger.from_training_rows(gaz, data["train"])
        ner_model_dir = ROOT / "models" / "sangli_ner"
        if tagger == "combo" and ner_model_dir.exists():
            sp = S.SpacyTagger(ner_model_dir)
            active_tagger = S.ComboTagger(sp, rule)
        elif tagger == "spacy" and ner_model_dir.exists():
            active_tagger = S.SpacyTagger(ner_model_dir)
        elif tagger == "hybrid" and ner_model_dir.exists():
            sp = S.SpacyTagger(ner_model_dir)
            active_tagger = S.HybridTagger(sp, rule)
        else:
            active_tagger = rule
        self.extractor = S.IncidentExtractor(active_tagger, gaz)
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

    def analyse(self, text, cautious=0.75):
        """cautious>0 shifts the urgency decision toward higher levels (measured: under-triage 22% -> 8% at b=0.75, over-triage 19% -> 53%)."""
        ner = self.extractor.extract(text); tp, up = self.clf.predict(text)
        scores = [math.log(up[l] + 1e-9) + cautious * i for i, l in enumerate(C.LEVELS)]
        urgency = C.LEVELS[max(range(4), key=scores.__getitem__)]; typ = max(tp, key=tp.get)
        loc = ner["location"]

        is_non_emg, non_emg_reason = detect_non_emergency(text, ner["keywords"])
        if is_non_emg:
            typ = "non_emergency"
            urgency = "none"
            tp = {"fire": 0.0, "medical": 0.0, "accident": 0.0, "crime": 0.0}
            up = {"low": 0.0, "medium": 0.0, "high": 0.0, "critical": 0.0}

        return derive({
            "transcript": text, "type": typ, "type_probs": tp, "urgency": urgency, "urgency_probs": up,
            "uncertain": max(up.values()) < 0.5, "location": loc, "keywords": ner["keywords"],
            "victim_count": ner["victim_count"], "time": ner["time"],
            "non_emergency": is_non_emg, "non_emergency_reason": non_emg_reason,
            "entities": [{k: e[k] for k in ("text", "label", "start", "end") if k in e} | ({"canonical": e["canonical"]} if e.get("canonical") else {}) for e in ner["entities"]],
        })
