"""
sangli_nlp.py -- Week 7: entity post-processing + structured incident extraction
==================================================================================
Pieces (all pure Python except SpacyTagger, which needs spaCy):
  Gazetteer        fuzzy + phonetic linking of an ASR-mangled place name to a canonical Sangli name
  parse_time       "about ten minutes ago" -> minutes
  parse_count      "several people" -> 3 (approximate)
  RuleTagger       lexicon + regex + gazetteer tagger (also the rule-based baseline for the report)
  SpacyTagger      trained spaCy NER (see ner_train.py)
  HybridTagger     spaCy spans + rule spans for what the model missed
  IncidentExtractor  text -> structured incident (location resolved to an area, keywords, count, time)
"""
import difflib, json, re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
KIND_LABEL = {"area": "AREA", "subarea": "AREA", "road": "ROAD", "chowk": "LANDMARK",
              "landmark": "LANDMARK", "generic": "LANDMARK", "city": "CITY"}
LABEL_KINDS = {"AREA": ("area", "subarea"), "ROAD": ("road",), "LANDMARK": ("chowk", "landmark", "generic"), "CITY": ("city",)}

# ----------------------------------------------------------------------------- phonetics
def compact(s): return re.sub(r"[^a-z0-9]", "", s.lower())

_PH = [("bh", "b"), ("gh", "g"), ("kh", "k"), ("dh", "d"), ("th", "t"), ("ph", "f"), ("sh", "s"), ("ck", "k"),
       ("ee", "i"), ("oo", "u"), ("aa", "a"), ("ou", "u"), ("w", "v"), ("y", "i"), ("z", "j"), ("q", "k")]


def phon(s):
    """Indian-English-friendly phonetic key: kh/bh/gh/th/dh..., v~w, c~k, ee~i, doubled letters collapsed."""
    s = compact(s)
    s = s.replace("ch", "C")
    for a, b in _PH: s = s.replace(a, b)
    s = s.replace("c", "k").replace("C", "c")
    return re.sub(r"(.)\1+", r"\1", s)


def skeleton(s): return re.sub(r"[aeiou]", "", phon(s))


# ----------------------------------------------------------------------------- gazetteer
class Gazetteer:
    def __init__(self, path=HERE / "sangli_gazetteer.json", threshold=0.82):
        g = json.load(open(path, encoding="utf-8")); self.threshold = threshold
        ents = {}

        def add(name, kind, areas=()):
            e = ents.setdefault((name, kind), {"name": name, "kind": kind, "label": KIND_LABEL[kind], "areas": set()})
            e["areas"].update(areas)
        add("Sangli", "city")
        for area, d in g["areas"].items():
            add(area, "area", [area])
            for k, kind in (("roads", "road"), ("chowks", "chowk"), ("landmarks", "landmark"), ("subareas", "subarea")):
                for n in d[k]: add(n, kind, [area])
        for h in g["highways"]: add(h, "road")
        for n in g["generic_spots"]: add(n, "generic")
        self.entries = list(ents.values())
        for e in self.entries:
            base = re.sub(r"^the ", "", e["name"].lower())
            e.update(c=compact(base), p=phon(base), s=skeleton(base))
        self.by_phon = defaultdict(list)
        for e in self.entries: self.by_phon[e["p"]].append(e)
        self.area_names = {e["name"] for e in self.entries if e["kind"] == "area"}
        self.first_words = {re.sub(r"^the ", "", e["name"].lower()).split()[0] for e in self.entries if e["kind"] != "generic"}

    def match(self, text, label=None):
        """Return (entry|None, score). Exact/phonetic hit = 1.0; otherwise fuzzy ratio on the phonetic key."""
        t = re.sub(r"^the ", "", text.lower().strip()); p = phon(t); c = compact(t)
        if len(c) < 3: return None, 0.0
        pool = self.entries
        if label in LABEL_KINDS: pool = [e for e in self.entries if e["kind"] in LABEL_KINDS[label]] or self.entries
        hits = [e for e in self.by_phon.get(p, []) if e in pool]
        if hits: return hits[0], 1.0
        best, bs = None, 0.0
        sm = difflib.SequenceMatcher(autojunk=False); sm.set_seq2(p)
        for e in pool:
            if abs(len(e["p"]) - len(p)) > 4: continue
            sm.set_seq1(e["p"])
            if sm.real_quick_ratio() < self.threshold - 0.05 or sm.quick_ratio() < self.threshold - 0.05: continue
            r = sm.ratio()
            if len(e["s"]) >= 4 and len(skeleton(t)) >= 4:           # consonant-skeleton rescue ("Theta" ~ "Theatre")
                r = max(r, 0.93 * difflib.SequenceMatcher(None, e["s"], skeleton(t)).ratio())
            if r > bs: best, bs = e, r
        return (best, bs) if bs >= self.threshold else (None, bs)


# ----------------------------------------------------------------------------- normalisers
NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
       "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30}
_NUMRE = "|".join(sorted(NUM, key=len, reverse=True)) + r"|\d{1,3}"       # words OR digits (Whisper writes "2 minutes")


def _num(w): return int(w) if w.isdigit() else NUM[w]


def parse_time(text):
    """TIME_REF text -> {'text','minutes_ago','ongoing','approximate'}"""
    t = text.lower().strip()
    out = {"text": text, "minutes_ago": None, "ongoing": False, "approximate": False}
    if "just now" in t or "seconds" in t: out["minutes_ago"] = 0; return out
    if "quarter of an hour" in t: out.update(minutes_ago=15, approximate=True); return out
    m = re.search(rf"(a couple of|a few|{_NUMRE})\s+minutes?", t)
    if m:
        w = m.group(1); n = 2 if w == "a couple of" else 3 if w == "a few" else _num(w)
        out.update(minutes_ago=n, approximate=w in ("a couple of", "a few") or bool(re.search(r"about|maybe|around", t)))
        return out
    if re.search(r"a minute", t): out["minutes_ago"] = 1; return out
    out["ongoing"] = True
    if "last hour" in t: out["minutes_ago"] = 60
    else:
        m = re.search(rf"({_NUMRE})\s+hours?", t)
        if m: out["minutes_ago"] = 60 * _num(m.group(1))
    out["approximate"] = True
    return out


def parse_count(text):
    """VICTIM_COUNT text -> {'text','count','approximate'}"""
    t = text.lower()
    for w, n, ap in (("a couple of", 2, True), ("a few", 3, True), ("several", 3, True), ("many", 5, True), ("both", 2, False)):
        if t.startswith(w): return {"text": text, "count": n, "approximate": ap}
    m = re.match(rf"({_NUMRE})\b", t)
    if m: return {"text": text, "count": _num(m.group(1)), "approximate": False}
    return {"text": text, "count": None, "approximate": True}


# ----------------------------------------------------------------------------- rule tagger
_NUMW = r"(?:a couple of|a few|a|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|\d{1,3})"
TIME_RE = re.compile(rf"(?:(?:about|maybe|around|just)\s+)?{_NUMW}\s+(?:seconds?|minutes?)\s+(?:ago|back)|just now|"
                     rf"(?:around\s+)?quarter of an hour ago|since (?:morning|last night|yesterday evening)|"
                     rf"for (?:the last hour|about (?:two|three|\d) hours)", re.I)
PEOPLE = r"(?:people|passengers|workers|labourers|riders|bikers|men|women|children|families|boys|walkers|students)"
COUNT_RE = re.compile(rf"\b(?:(?:two|three|four|five|six|seven|\d{{1,2}}|several|many|a few|a couple of|both)\s+(?:armed\s+|young\s+)?{PEOPLE}|(?:two|three|four) of them)\b", re.I)
BUILDING_RE = re.compile(r"Flat \d+, [A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*){0,2}|House number \d+")
LOC_PREP_RE = re.compile(r"(?:near|at|opposite|behind|outside|beside|by|to|from|in front of|next to|under)\s+$|(?:^|[.,]\s*)$", re.I)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+")
STOP_START = set("please sir yes hello oh it we the a an i there this my our near at on in opposite behind next front under outside also come send kindly thank someone nobody people uh um okay listen wait help hurry".split())

# Universal emergency symptom & dispatch phrases to complement synthetic training lexicon
COMMON_EMERGENCY_RE = [
    # Breathing and respiratory distress variations
    re.compile(r'\b((?:not\s+(?:able\s+to\s+)?|unable\s+to\s+|can\'?t\s+|cannot\s+|hard\s+to\s+|trouble\s+|difficulty\s+(?:in\s+)?|struggling\s+to\s+|stopped\s+|shortness\s+of\s+)breath(?:e|ing)?(?:\s+properly)?|breathlessness|gasping(?:\s+for\s+breath)?|choking|suffocating|wheezing)\b', re.I),
    # Consciousness and responsiveness
    re.compile(r'\b(unconscious|unresponsive|not\s+(?:responding|waking\s+up|moving)|passed\s+out|blacked\s+out|fainted|loss\s+of\s+consciousness|collapsed)\b', re.I),
    # Cardiac and chest emergencies
    re.compile(r'\b(chest\s+pain|heart\s+attack|cardiac\s+arrest|no\s+pulse|heart\s+stopped)\b', re.I),
    # Severe bleeding, trauma, and fractures
    re.compile(r'\b(bleeding(?:\s+heavily)?|losing\s+blood|pool\s+of\s+blood|head\s+injury|fracture|broken\s+(?:bone|leg|arm|hip)|internal\s+bleeding)\b', re.I),
    # Universal calls for emergency assistance
    re.compile(r'\b(emergency|medical\s+emergency|send\s+(?:help|an?\s+ambulance|the\s+police|the\s+fire\s+brigade)|need\s+(?:urgent\s+)?help)\b', re.I),
]


class RuleTagger:
    """Closed-vocabulary tagger: keyword lexicon (from TRAIN) + regexes + fuzzy gazetteer windows."""

    def __init__(self, gaz, keyword_forms=()):
        self.gaz = gaz
        forms = sorted({f.lower() for f in keyword_forms}, key=lambda f: (-len(f), f))
        self.kw_re = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(f) for f in forms) + r")(?![\w-])", re.I) if forms else None
        gen = sorted((e["name"] for e in gaz.entries if e["kind"] == "generic"), key=len, reverse=True)
        self.gen_re = re.compile("|".join(re.escape(g) for g in gen), re.I)

    @classmethod
    def from_training_rows(cls, gaz, train_rows, min_rate=0.9):
        """Keyword lexicon = forms the annotators label >= min_rate of the time in TRAIN (drops context-dependent words)."""
        lab, tot = Counter(), Counter()
        forms = {e["text"].lower() for r in train_rows for e in r["entities"] if e["label"] == "EMERGENCY_KEYWORD"}
        for f in forms:
            rx = re.compile(r"(?<![\w-])" + re.escape(f) + r"(?![\w-])", re.I)
            for r in train_rows:
                spans = [(e["start"], e["end"]) for e in r["entities"] if e["label"] == "EMERGENCY_KEYWORD"]
                for m in rx.finditer(r["transcript_clean"]):
                    tot[f] += 1; lab[f] += any(a <= m.start() and m.end() <= b for a, b in spans)
        return cls(gaz, [f for f in forms if lab[f] / max(tot[f], 1) >= min_rate])

    def tag(self, text):
        spans = []                                           # (start, end, label)

        def free(a, b): return all(b <= x or a >= y for x, y, _ in spans)

        def put(a, b, label):
            if free(a, b): spans.append((a, b, label))
        for m in BUILDING_RE.finditer(text): put(m.start(), m.end(), "BUILDING")
        for m in self.gen_re.finditer(text):
            if LOC_PREP_RE.search(text[max(0, m.start() - 16):m.start()]) or m.start() == 0: put(m.start(), m.end(), "LANDMARK")
        # gazetteer windows (proper-noun-like starts only)
        toks = [(m.start(), m.end(), m.group(0)) for m in TOKEN_RE.finditer(text)]
        cands = []
        for i, (a, _, w) in enumerate(toks):
            if not (w[0].isupper() or w.isdigit() or w.lower() in self.gaz.first_words) or w.lower() in STOP_START: continue
            for n in range(5, 0, -1):
                if i + n > len(toks): continue
                if any(text[toks[j][1]:toks[j + 1][0]] != " " for j in range(i, i + n - 1)): continue   # no punctuation inside a name
                b = toks[i + n - 1][1]; ent, sc = self.gaz.match(text[a:b])
                if ent and (sc >= 1.0 or w[0].isupper() or w.isdigit()): cands.append((sc, n, a, b, ent["label"]))
        for sc, n, a, b, label in sorted(cands, key=lambda c: (-c[0] * (1 + 0.02 * c[1]), -c[1])):
            put(a, b, label)
        for m in TIME_RE.finditer(text): put(m.start(), m.end(), "TIME_REF")
        for m in COUNT_RE.finditer(text): put(m.start(), m.end(), "VICTIM_COUNT")
        for rx in COMMON_EMERGENCY_RE:
            for m in rx.finditer(text): put(m.start(), m.end(), "EMERGENCY_KEYWORD")
        if self.kw_re:
            for m in self.kw_re.finditer(text): put(m.start(), m.end(), "EMERGENCY_KEYWORD")
        return sorted(spans)


class SpacyTagger:
    def __init__(self, model_dir):
        import spacy
        self.nlp = spacy.load(model_dir)

    def tag(self, text):
        return sorted((e.start_char, e.end_char, e.label_) for e in self.nlp(text).ents)


class HybridTagger:
    """spaCy spans first; rule spans only where they don't overlap a model span."""

    # VICTIM_COUNT is context-dependent (victims vs bystanders) -> left to the model only
    def __init__(self, model_tagger, rule_tagger, rule_labels=("AREA", "ROAD", "LANDMARK", "BUILDING", "CITY", "TIME_REF", "EMERGENCY_KEYWORD")):
        self.m, self.r, self.labels = model_tagger, rule_tagger, set(rule_labels)

    def tag(self, text):
        spans = list(self.m.tag(text))
        for a, b, l in self.r.tag(text):
            if l in self.labels and all(b <= x or a >= y for x, y, _ in spans): spans.append((a, b, l))
        return sorted(spans)


class ComboTagger:
    """Rules for everything a closed vocabulary covers (places, building, time, known keywords: they are precise and
    never hallucinate); the trained model for VICTIM_COUNT and model-generalized EMERGENCY_KEYWORD spans."""

    def __init__(self, model_tagger, rule_tagger, model_labels=("VICTIM_COUNT", "EMERGENCY_KEYWORD")):
        self.m, self.r, self.ml = model_tagger, rule_tagger, set(model_labels)

    def tag(self, text):
        spans = list(self.r.tag(text))
        def free(a, b): return all(b <= x or a >= y for x, y, _ in spans)
        for a, b, l in self.m.tag(text):
            if l in self.ml and free(a, b): spans.append((a, b, l))
        return sorted(spans)


# ----------------------------------------------------------------------------- incident extractor
class IncidentExtractor:
    W_AREA, W_SUB = 3.0, 2.0

    def __init__(self, tagger, gaz): self.tagger, self.gaz = tagger, gaz

    def extract(self, text):
        ents, votes = [], Counter()
        for a, b, label in self.tagger.tag(text):
            e = {"text": text[a:b], "label": label, "start": a, "end": b}
            if label in LABEL_KINDS:
                hit, sc = self.gaz.match(e["text"], label)
                if hit:
                    e.update(canonical=hit["name"], kind=hit["kind"], link_score=round(sc, 2), areas=sorted(hit["areas"]))
                    w = self.W_AREA if hit["kind"] == "area" else self.W_SUB if hit["kind"] == "subarea" else 1.0
                    for ar in hit["areas"]: votes[ar] += w / len(hit["areas"])
                else:
                    e.update(canonical=None, link_score=round(sc, 2))
            ents.append(e)
        explicit = {e["canonical"] for e in ents if e.get("kind") == "area"}
        area, conf, review = None, 0.0, True
        if votes:
            area, top = votes.most_common(1)[0]; conf = round(top / sum(votes.values()), 2)
            review = conf < 0.6 or len(explicit) > 1
        pick = lambda k: next((e["canonical"] for e in ents if e.get("kind") in k and e.get("canonical")), None)
        building = next((e["text"] for e in ents if e["label"] == "BUILDING"), None)
        time = next((parse_time(e["text"]) for e in ents if e["label"] == "TIME_REF"), None)
        count = next((parse_count(e["text"]) for e in ents if e["label"] == "VICTIM_COUNT"), None)
        parts = [building, pick(("subarea",)), pick(("road",)), pick(("chowk", "landmark", "generic")), area]
        return {"entities": ents,
                "location": {"area": area, "area_confidence": conf, "road": pick(("road",)),
                             "landmark": pick(("chowk", "landmark", "generic")), "subarea": pick(("subarea",)),
                             "building": building, "needs_confirmation": review,
                             "summary": ", ".join(dict.fromkeys(p for p in parts if p)) or None,
                             "unlinked_place_text": [e["text"] for e in ents if e["label"] in LABEL_KINDS and not e.get("canonical")]},
                "keywords": list(dict.fromkeys(e["text"].lower() for e in ents if e["label"] == "EMERGENCY_KEYWORD")),
                "victim_count": count, "time": time}
