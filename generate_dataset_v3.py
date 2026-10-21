"""
Synthetic Emergency Call Dataset v3 -- Sangli, Maharashtra
===========================================================
Fixes over v2 (found by auditing v2):
  * Scenarios are hand-written, self-contained and tagged with WHERE they can happen
    (road / home / public / open / river / railway) -> no more road crashes at "Flat 304".
  * 20 phrasings per (type x severity) = 320 scenarios (v2: 8). Hold-out is by scenario:
    0-13 train, 14-16 val, 17-19 test  -> val/test each contain 48 distinct unseen scenarios (v2: 16).
  * No random "sorry, I mean," glue, no victim-contradicting extras, no pronoun clashes,
    no "We are at X, about two minutes ago", no "House number 5, Shree Apartment".
  * urgency_score is fixed per scenario (v2 drew it at random inside the band, so it was noise).
  * ASR-noisy twin only does phonetic place-name confusions + dropped fillers (no letter swaps).
Everything is synthetic. Place names come from sangli_gazetteer.json (web-compiled, not surveyed).

Usage:  python3 generate_dataset_v3.py        -> writes ./output_v3/
"""
import csv, json, random, re
from pathlib import Path
from scenario_bank_v3 import SCENARIOS, OPENERS, NEUTRAL_EXTRAS, KIND_EXTRAS
from scenario_bank_v3b import SCENARIOS_B
for _t, _d in SCENARIOS_B.items():                       # scenarios 20-39 per (type, severity); 0-19 are the original bank
    for _s, _l in _d.items(): SCENARIOS[_t][_s] = SCENARIOS[_t][_s] + _l

HERE = Path(__file__).parent
OUT_DIR = HERE / "output_v3"; OUT_DIR.mkdir(exist_ok=True)
GAZ = json.load(open(HERE / "sangli_gazetteer.json", encoding="utf-8"))
AREAS, HIGHWAYS = GAZ["areas"], GAZ["highways"]

BASE_SEED = 20261015
SPLIT_COUNTS = {"train": 1700, "val": 150, "test": 150}          # per emergency type -> 8000 total
SPLIT_OF_IDX = {**{i: "train" for i in range(14)}, 14: "val", 15: "val", 16: "val", 17: "test", 18: "test", 19: "test"}

HIGHWAY_AREAS = ["Ankali", "Haripur", "Budhgaon", "Kupwad", "Bamnoli"]
GEN_ROAD = ["the main bus stop", "the petrol pump"]
GEN_PUBLIC = ["the main bus stop", "the petrol pump", "the vegetable market", "the cinema hall", "the temple"]
GEN_OPEN = ["the water tank", "the school ground", "the temple", "the sugarcane field"]
WATER = [("Haripur", "Irwin Bridge"), ("Ankali", "Ankali Bridge"), ("Ganapati Peth", "Krishna Ghat"),
         ("Haripur", "Krishna river bank")]
RAIL = [("Market Yard", "Sangli Railway Station"), ("Vishrambag", "Vishrambag Railway Station"), ("Miraj", "Miraj Junction")]
WATER_NAMES = {lm for _, lm in WATER}; RAIL_NAMES = {lm for _, lm in RAIL}
DEFAULT_PREPS = ["near", "opposite", "behind", "next to", "in front of", "at"]
LM_PREPS = {"Krishna river bank": ["on", "near"], "Irwin Bridge": ["on", "near", "at"],
            "Ankali Bridge": ["on", "near", "at"], "Vishrambag flyover": ["near", "under"],
            "Krishna Ghat": ["at", "near"], "Sangli Railway Station": ["near", "at", "outside"],
            "Vishrambag Railway Station": ["near", "at", "outside"], "Miraj Junction": ["near", "at", "outside"]}
BUILDING_NAMES = ["Shree Apartment", "Sai Residency", "Datta Prasad Apartment", "Ganesh Heights", "Om Sai Complex",
                  "Laxmi Niwas", "Krishna Residency", "Shivam Apartment", "Vasant Tower", "Ashirwad Apartment",
                  "Swami Samarth Apartment", "Mangal Residency", "Rajlaxmi Apartment", "Sahyadri Heights", "Gokul Park"]
LOCATION_LABELS = {"LANDMARK", "ROAD", "AREA", "BUILDING", "CITY"}

# phonetic confusions Whisper realistically makes (noisy twin only)
CONFUSIONS = {
    "Vishrambag": ["Vishram Bagh", "Vishrambaug"], "Khanbhag": ["Kanbhag", "Khan Bagh"],
    "Gaon Bhag": ["Gaonbhag", "Gaon Bag"], "Ganapati Peth": ["Ganpati Peth", "Ganpathi Peth"],
    "Miraj": ["Meeraj", "Miraz"], "Kupwad": ["Kupvad", "Kup wad"], "Madhavnagar": ["Madhav Nagar"],
    "Brahmanpuri": ["Brahman Puri"], "Walchand": ["Wallchand"], "Sangli": ["Sangly", "Sanglee"],
    "Tilak": ["Tilac"], "Shivneri": ["Shivner"], "Garpir": ["Garpeer"], "Kandil": ["Candle"],
    "Siddhivinayak": ["Siddhi Vinayak"], "Neminathnagar": ["Neminath Nagar"], "Ankali": ["Ankli"],
    "Haripur": ["Haripoor"], "Budhgaon": ["Budhgaon", "Budh Gaon"], "Bamnoli": ["Bamnoli", "Bamnolly"],
    "Kupwad": ["Kupvad"], "Ganpati": ["Ganapati"], "Panchamukhi": ["Panchmukhi"], "Irwin": ["Erwin"],
    "Sangameshwar": ["Sangmeshwar"], "Vitthal": ["Vithal"], "Pushparaj": ["Pushpraj"], "Harbhat": ["Harbhat"],
}

SEV = {"mild": ("low", (1, 2, 3)), "moderate": ("medium", (4, 5, 6)),
       "severe": ("high", (7, 8)), "critical": ("critical", (9, 10))}
SEVERITY_WEIGHTS = [("mild", 0.15), ("moderate", 0.35), ("severe", 0.35), ("critical", 0.15)]
LEVELS = ["low", "medium", "high", "critical"]
EMOTION_TAGS = {
    "low": ["[calm but worried]", "[a little shaky]", "[speaking quickly but controlled]"],
    "medium": ["[distressed]", "[voice trembling]", "[anxious, talking fast]"],
    "high": ["[panicked]", "[out of breath]", "[shouting over noise]", "[sobbing]"],
    "critical": ["[hysterical, crying]", "[screaming]", "[gasping, barely able to speak]", "[sobbing uncontrollably]"],
}
CLOSERS = {
    "low": ["Can someone come and check, please.", "I just wanted to report it.",
            "Please send someone when you can.", "Please sir, someone should look into it."],
    "medium": ["Please send someone quickly.", "I don't know what else to do, please hurry.",
               "Please come fast sir.", "Kindly send help as soon as possible."],
    "high": ["Please, please hurry, I don't know how much time we have.", "Someone needs to get here right now!",
             "Please sir, send help immediately!", "Hurry up, it's getting worse every second!"],
    "critical": ["Oh god please hurry, PLEASE!", "I don't- I don't know if- please just come NOW.",
                 "Please, please help, please sir!", "Come quickly, there's no time, no time!"],
}
NEUTRAL_CLOSERS = ["Please send someone.", "Please come as soon as possible.", "Please sir, send help.",
                   "Kindly send help.", "Please hurry.", "Please come quickly sir.",
                   "Send someone fast, please.", "Please do something.", "Thank you, please come."]
FILLERS_CALM = ["um, ", "uh, ", "sir, ", "listen, ", "okay, "]
FILLERS_PANIC = FILLERS_CALM + ["oh god, ", "please, ", "I- I don't know, ", "wait, wait, "]
REL_TIMES = ["just now", "a few seconds ago", "about two minutes ago", "maybe five minutes back",
             "about ten minutes ago", "a couple of minutes ago", "just a minute ago",
             "around quarter of an hour ago", "a few minutes back"]
DUR_TIMES = ["since morning", "since last night", "for the last hour", "for about two hours", "since yesterday evening"]
SECONDARY = {  # primary -> [(secondary type, sentence)] ; only on moderate/severe, kind-compatible
    "fire": [("medical", "Also, someone has inhaled [[KW:smoke]] and is [[KW:unconscious]].", {"H", "P"})],
    "accident": [("medical", "Also, a bystander is [[KW:bleeding]] from the head.", {"R"}),
                 ("fire", "Also, fuel is [[KW:leaking]] on the road and I can see sparks.", {"R"})],
    "crime": [("medical", "Also, someone is [[KW:bleeding]] and not responding.", {"H", "P", "R", "A"})],
}
LABEL_MAP = {"KW": "EMERGENCY_KEYWORD", "VC": "VICTIM_COUNT"}
MARK = re.compile(r"\[\[(\w+):(.*?)\]\]")
KW_RE = None   # compiled after _kw_forms is defined (below)


# ---------------------------------------------------------------- annotation guideline (v3.1)
# EMERGENCY_KEYWORD: EVERY occurrence of a keyword form is labelled (longest form wins), not only the one the
# scenario author marked. Forms = all [[KW:..]] in the bank/openers/secondary clauses. Words whose meaning depends
# on context are NOT auto-labelled (they keep only the scenario author's label).
# VICTIM_COUNT: number of people involved in the incident as the caller states it (victims, trapped people or
# attackers: "three people", "both riders", "two men"). Bystander phrases ("A few people have gathered") are not labelled.
CONTEXT_DEPENDENT = {"red", "fit", "force", "breathing", "broken"}


def _kw_forms():
    forms = set()
    texts = [e for d in SCENARIOS.values() for l in d.values() for e in l] + \
            [o for v in OPENERS.values() for o in v] + [t for v in SECONDARY.values() for _, t, _ in v]
    for t in texts:
        forms.update(m.group(2).lower() for m in MARK.finditer(t) if m.group(1) == "KW")
    return sorted(forms - CONTEXT_DEPENDENT, key=lambda f: (-len(f), f))


def harmonize(text, ents):
    """Add-only consistency pass: label unlabelled occurrences of known keyword forms."""
    taken = [(e["start"], e["end"]) for e in ents]; added = []
    for m in KW_RE.finditer(text):
        a, b = m.start(), m.end()
        if all(b <= x or a >= y for x, y in taken):
            taken.append((a, b)); added.append({"text": m.group(0), "label": "EMERGENCY_KEYWORD", "start": a, "end": b})
    return sorted(ents + added, key=lambda e: e["start"])


def cap_first(s): return s[:1].upper() + s[1:]


KW_RE = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(f) for f in _kw_forms()) + r")(?![\w-])", re.I)


def parse(entry):
    head, text = entry.split("|", 1)
    kind, _, flag = head.partition(":")
    return kind, flag, text


def corrupt(text, nrng):
    for k, variants in CONFUSIONS.items():
        if k in text:
            return text.replace(k, nrng.choice(variants), 1)
    return text


class Builder:
    def __init__(self, noise, nrng):
        self.parts, self.ents, self.pos, self.noise, self.nrng = [], [], 0, noise, nrng

    def t(self, s): self.parts.append(s); self.pos += len(s)

    def e(self, s, label):
        st = self.pos; self.t(s); self.ents.append({"text": s, "label": label, "start": st, "end": self.pos})

    def filler(self, s):          # Whisper tends to drop um/uh; returns whether it was emitted
        if self.noise and self.nrng.random() < 0.7: return False
        self.t(s); return True

    def emit(self, tpl, cap=False):
        pos, first = 0, True
        for m in MARK.finditer(tpl):
            plain = tpl[pos:m.start()]
            if plain: self.t(cap_first(plain) if (cap and first) else plain)
            ent = cap_first(m.group(2)) if (cap and first and not plain) else m.group(2)   # sentence may start with an entity
            self.e(ent, LABEL_MAP[m.group(1)]); first, pos = False, m.end()
        rest = tpl[pos:]
        if rest: self.t(cap_first(rest) if (cap and first) else rest)

    def emit_loc(self, segs, cap=False):
        st = self.pos
        for i, (text, label) in enumerate(segs):
            if label is None:
                self.t(cap_first(text) if (cap and i == 0) else text)
            else:
                if self.noise and label in LOCATION_LABELS and self.nrng.random() < 0.35:
                    text = corrupt(text, self.nrng)
                self.e(text, label)
        return {"start": st, "end": self.pos}

    def text(self): return "".join(self.parts)


# ------------------------------------------------------------------ locations
ANCHOR_WEIGHTS = {
    "H": {"building": 5, "subarea": 3, "landmark": 2, "chowk": 1, "area_only": 1},
    "R": {"road": 4, "chowk": 3, "highway": 1, "landmark": 1, "generic_road": 1},
    "P": {"landmark": 3, "chowk": 3, "generic_public": 3, "road": 1},
    "O": {"generic_open": 3, "landmark": 1, "area_only": 2, "highway": 1},
    "A": {"building": 3, "subarea": 2, "landmark": 3, "chowk": 3, "road": 3, "generic_public": 2, "area_only": 1},
}
GENERIC = {"generic_road": GEN_ROAD, "generic_public": GEN_PUBLIC, "generic_open": GEN_OPEN}


def lms(area, kind):
    out = [l for l in AREAS[area]["landmarks"] if l not in WATER_NAMES]
    if kind == "H": out = [l for l in out if l not in RAIL_NAMES]
    return out


def has(area, anchor, kind):
    g = AREAS[area]
    return {"landmark": bool(lms(area, kind)), "chowk": bool(g["chowks"]), "road": bool(g["roads"]),
            "subarea": bool(g["subareas"]), "building": True, "area_only": True,
            "highway": True, "generic_road": True, "generic_public": True, "generic_open": True}[anchor]


def make_location(rng, kind):
    if kind in ("W", "RW"):
        area, lm = rng.choice(WATER if kind == "W" else RAIL)
        prep = rng.choice(LM_PREPS[lm]); the = "the " if (rng.random() < 0.3 and lm.startswith(("Irwin", "Ankali"))) else ""
        segs = [(f"{prep} {the}", None), (lm, "LANDMARK"), (", ", None), (area, "AREA")]
    else:
        names = list(AREAS)
        for _ in range(50):
            anchors = ANCHOR_WEIGHTS[kind]
            anchor = rng.choices(list(anchors), weights=list(anchors.values()))[0]
            if anchor == "highway": area = rng.choice(HIGHWAY_AREAS)
            else:
                area = rng.choices(names, weights=[AREAS[a]["weight"] for a in names])[0]
            if has(area, anchor, kind): break
        g = AREAS[area]; segs = []
        if anchor == "landmark":
            lm = rng.choice(lms(area, kind)); prep = rng.choice(LM_PREPS.get(lm, DEFAULT_PREPS))
            segs += [(prep + " " + ("the " if rng.random() < 0.3 and lm not in LM_PREPS else ""), None), (lm, "LANDMARK")]
        elif anchor == "chowk":
            segs += [(rng.choice(["near", "at", "opposite"]) + " ", None), (rng.choice(g["chowks"]), "LANDMARK")]
            if g["roads"] and rng.random() < 0.4: segs += [(" on ", None), (rng.choice(g["roads"]), "ROAD")]
        elif anchor == "road":
            segs += [("on ", None), (rng.choice(g["roads"]), "ROAD")]
            if g["chowks"] and rng.random() < 0.3: segs += [(", near ", None), (rng.choice(g["chowks"]), "LANDMARK")]
        elif anchor == "subarea":
            segs += [("in ", None), (rng.choice(g["subareas"]), "AREA")]
        elif anchor == "building":
            if rng.random() < 0.6:
                b = f"Flat {rng.randint(1, 5)}0{rng.randint(1, 6)}, {rng.choice(BUILDING_NAMES)}"
            else:
                b = f"House number {rng.randint(1, 250)}"
            segs += [("at ", None), (b, "BUILDING")]
            if g["subareas"] and rng.random() < 0.6: segs += [(", ", None), (rng.choice(g["subareas"]), "AREA")]
        elif anchor == "highway":
            segs += [("on the ", None), (rng.choice(HIGHWAYS), "ROAD"), (" near ", None)]
        elif anchor in GENERIC:
            segs += [(rng.choice(["near", "next to", "at", "opposite"]) + " ", None), (rng.choice(GENERIC[anchor]), "LANDMARK")]
        if anchor == "area_only": segs += [("in ", None), (area, "AREA")]
        elif anchor == "highway": segs += [(area, "AREA")]
        else: segs += [(", ", None), (area, "AREA")]
    if rng.random() < 0.3: segs += [(", ", None), ("Sangli", "CITY")]
    return segs, area


def pick_closer(rng, level):
    """Closers must not reveal urgency on their own: 50% neutral, 25% own level, 25% adjacent level."""
    r = rng.random()
    if r < 0.5: return rng.choice(NEUTRAL_CLOSERS)
    if r < 0.75: return rng.choice(CLOSERS[level])
    i = LEVELS.index(level); adj = [LEVELS[j] for j in (i - 1, i + 1) if 0 <= j < 4]
    return rng.choice(CLOSERS[rng.choice(adj)])


# ------------------------------------------------------------------ row builder
def build(seed, etype, severity, scen_idx, noise):
    rng = random.Random(seed); nrng = random.Random(seed * 7919 + 13); b = Builder(noise, nrng)
    label, scores = SEV[severity]; score = scores[scen_idx % len(scores)]
    kind, flag, scen = parse(SCENARIOS[etype][severity][scen_idx])
    opener = rng.choice(OPENERS[etype]) if rng.random() < 0.75 else None
    pool = NEUTRAL_EXTRAS + KIND_EXTRAS.get(kind, [])
    extra = rng.choice(pool) if rng.random() < 0.4 else None
    secondary = None
    if severity in ("moderate", "severe") and rng.random() < 0.08:
        opts = [s for s in SECONDARY.get(etype, []) if kind in s[2]]
        if opts: secondary = rng.choice(opts)
    segs, area = make_location(rng, kind)
    use_time = flag != "N" and rng.random() < 0.8
    time_ref = rng.choice(DUR_TIMES if flag == "D" else REL_TIMES)
    time_frame = "It's been like this " if flag == "D" else rng.choice(["It happened ", "It started ", "This happened "])
    loc_frame = rng.choice(["We are ", "It's "] + ([] if kind == "H" else ["It's happening "]))
    closer = pick_closer(rng, label)
    filler = rng.choice(FILLERS_PANIC if label in ("high", "critical") else FILLERS_CALM)
    use_filler = rng.random() < 0.5
    struct = rng.choices(["classic", "loc_first", "yes_sir", "scen_first", "plain", "terse"],
                         weights=[4, 2, 2, 3, 2, 2 if label in ("high", "critical") else 0])[0]

    def say(op, fill):
        """Opener (optional) + scenario sentence."""
        if op:
            b.emit(op, cap=True)
            if fill: b.t("... "); b.filler(filler); b.emit(scen)
            else: b.t(". "); b.emit(scen, cap=True)
        else:
            em = b.filler(filler) if fill else False
            b.emit(scen, cap=not em)
        b.t(". ")

    def tail(with_extra=True):
        if secondary: b.emit(secondary[1], cap=True); b.t(" ")
        if with_extra and extra: b.emit(extra); b.t(" ")

    def locsent():
        b.t(loc_frame); span = b.emit_loc(segs); b.t(". "); return span

    def timesent():
        if use_time: b.t(time_frame); b.e(time_ref, "TIME_REF"); b.t(". ")

    if struct in ("classic", "plain"):
        say(opener, use_filler and struct == "classic"); tail(); loc = locsent(); timesent()
    elif struct == "scen_first":
        say(None, use_filler); tail(); loc = locsent(); timesent()
    elif struct == "loc_first":
        loc = b.emit_loc(segs, cap=True); b.t(". ")
        say(opener, False); tail(); timesent()
    elif struct == "yes_sir":
        b.t("Yes sir, "); loc = b.emit_loc(segs); b.t(". "); say(None, False); tail(); timesent()
    else:  # terse: panic fragment, no time, no extra
        say(None, False); tail(False); loc = locsent()
    b.t(closer)
    text = b.text()
    if text and text[0].islower():                    # call that opens with a filler or an entity ("several people ...")
        text = text[0].upper() + text[1:]
        for e in b.ents:
            if e["start"] == 0: e["text"] = cap_first(e["text"])
    return {"text": text, "entities": harmonize(text, b.ents), "location_phrase": loc, "area": area,
            "secondary_type": secondary[0] if secondary else None, "urgency_label": label,
            "urgency_score": score, "tag": rng.choice(EMOTION_TAGS[label]), "structure": struct, "place_kind": kind}


EXT_SPLIT_OF_IDX = {**{i: "train" for i in range(20, 34)}, 34: "val", 35: "val", 36: "val", 37: "test", 38: "test", 39: "test"}
EXT_SPLIT_COUNTS = {"train": 1000, "val": 100, "test": 100}


def generate(split_of_idx=SPLIT_OF_IDX, counts=SPLIT_COUNTS, seed_off=0, counter0=0, id_off=0):
    """Defaults reproduce the original 8,000 rows exactly; the extension call uses new scenarios, seeds and ids."""
    master = random.Random(BASE_SEED + seed_off); rows, seen, counter = [], set(), counter0
    train_idx = [i for i, s in split_of_idx.items() if s == "train"]
    val_idx = [i for i, s in split_of_idx.items() if s == "val"]
    test_idx = [i for i, s in split_of_idx.items() if s == "test"]
    for etype in SCENARIOS:
        splits = sum(([s] * n for s, n in counts.items()), []); master.shuffle(splits)
        for i, split in enumerate(splits, 1):
            severity = master.choices([s for s, _ in SEVERITY_WEIGHTS], weights=[w for _, w in SEVERITY_WEIGHTS])[0]
            scen_idx = master.choice({"train": train_idx, "val": val_idx, "test": test_idx}[split])
            while True:
                counter += 1; seed = BASE_SEED + counter
                clean = build(seed, etype, severity, scen_idx, False)
                if clean["text"] not in seen: seen.add(clean["text"]); break
            noisy = build(seed, etype, severity, scen_idx, True)
            rows.append({
                "id": f"{etype}_{i + id_off:04d}", "city": "Sangli", "split": split, "emergency_type": etype,
                "secondary_type": clean["secondary_type"], "severity_band": severity,
                "urgency_label": clean["urgency_label"], "urgency_score": clean["urgency_score"],
                "scenario_id": f"{etype}/{severity}/{scen_idx}", "place_kind": clean["place_kind"],
                "primary_area": clean["area"], "transcript_clean": clean["text"],
                "transcript_with_direction": f"{clean['tag']} {clean['text']}",
                "entities": clean["entities"], "location_phrase": clean["location_phrase"],
                "transcript_asr_noisy": noisy["text"], "entities_asr_noisy": noisy["entities"]})
    master.shuffle(rows); return rows


def save(rows):
    with open(OUT_DIR / "emergency_calls_v3_full.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for sp in ("train", "val", "test"):
        with open(OUT_DIR / f"{sp}_v3.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                if r["split"] == sp: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "emergency_calls_v3_overview.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "split", "emergency_type", "urgency_label", "urgency_score", "primary_area", "transcript_clean"])
        for r in rows: w.writerow([r["id"], r["split"], r["emergency_type"], r["urgency_label"], r["urgency_score"], r["primary_area"], r["transcript_clean"]])


if __name__ == "__main__":
    data = generate() + generate(EXT_SPLIT_OF_IDX, EXT_SPLIT_COUNTS, 7, 1_000_000, 2000); save(data); print(f"rows: {len(data)} -> {OUT_DIR}")
