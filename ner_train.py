"""
ner_train.py -- train the spaCy NER model (Week 7).   Needs:  pip install spacy
    python3 ner_train.py                       # trains on train_v3 (clean + ASR-noisy copies), early-stops on val
    python3 ner_train.py --max-epochs 20 --patience 4
Then evaluate:
    python3 ner_eval.py --tagger spacy  --model models/sangli_ner
    python3 ner_eval.py --tagger hybrid --model models/sangli_ner
Model = blank English pipeline + a from-scratch NER head (CNN tok2vec). CPU is fine (~1 min/epoch on one core).
Why train on the ASR-noisy copy too: at run time the NER reads Whisper output, not clean scripts.
"""
import argparse, json, random, time
from pathlib import Path
import spacy
from spacy.training import Example
from spacy.util import minibatch, fix_random_seed
import ner_eval as E

HERE = Path(__file__).parent


def make_examples(nlp, rows, noisy, stats):
    out = []
    for r in rows:
        text, ents = (r["transcript_asr_noisy"], r["entities_asr_noisy"]) if noisy else (r["transcript_clean"], r["entities"])
        doc = nlp.make_doc(text); spans = []
        for e in ents:
            if doc.char_span(e["start"], e["end"]) is None: stats["dropped"] += 1; continue   # not token-aligned -> skip span
            spans.append((e["start"], e["end"], e["label"]))
        stats["kept"] += len(spans)
        out.append(Example.from_dict(doc, {"entities": spans}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "output_v3")); ap.add_argument("--model-dir", default=str(HERE / "models" / "sangli_ner"))
    ap.add_argument("--max-epochs", type=int, default=15); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--dropout", type=float, default=0.2); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    fix_random_seed(a.seed); random.seed(a.seed)
    data = E.load(a.data); nlp = spacy.blank("en"); ner = nlp.add_pipe("ner")
    for l in E.LABELS: ner.add_label(l)
    st = {"kept": 0, "dropped": 0}
    train = make_examples(nlp, data["train"], False, st) + \
            make_examples(nlp, [r for r in data["train"] if r["transcript_asr_noisy"] != r["transcript_clean"]], True, st)
    val_clean, val_noisy = make_examples(nlp, data["val"], False, st), make_examples(nlp, data["val"], True, st)
    print(f"train docs {len(train)} | val {len(val_clean)} clean + {len(val_noisy)} noisy | entity spans kept {st['kept']}, "
          f"dropped (not token-aligned) {st['dropped']}", flush=True)
    opt = nlp.initialize(lambda: train[:500])
    best, bad, hist = -1.0, 0, []
    Path(a.model_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, a.max_epochs + 1):
        t0 = time.time(); random.shuffle(train); losses = {}
        for batch in minibatch(train, size=a.batch):
            nlp.update(batch, sgd=opt, drop=a.dropout, losses=losses)
        fc, fn = float(nlp.evaluate(val_clean)["ents_f"]), float(nlp.evaluate(val_noisy)["ents_f"]); f = (fc + fn) / 2
        hist.append({"epoch": epoch, "loss": round(float(losses["ner"]), 1), "val_f1_clean": round(fc, 4), "val_f1_noisy": round(fn, 4)})
        flag = ""
        if f > best + 1e-4: best, bad = f, 0; nlp.to_disk(a.model_dir); flag = "  <- saved"
        else: bad += 1
        print(f"epoch {epoch:2d}  loss={losses['ner']:9.1f}  val F1 clean={fc:.4f} noisy={fn:.4f}  [{time.time()-t0:.0f}s]{flag}", flush=True)
        if bad >= a.patience: print("early stop (no val improvement)"); break
    json.dump({"best_val_f1": round(best, 4), "history": hist, "alignment": st}, open(HERE / "ner_train_log.json", "w"), indent=2)
    print(f"\nbest val F1 {best:.4f}; model saved to {a.model_dir}\nNext: python3 ner_eval.py --tagger spacy / hybrid")


if __name__ == "__main__":
    main()
