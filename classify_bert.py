"""
classify_bert.py -- fine-tune a small transformer for emergency TYPE + URGENCY (Week 8), CPU-friendly.
One shared encoder, two heads (type: 4 classes, urgency: 4 levels), mean pooling.

Setup:   pip install torch transformers           (CPU build is fine)
Quick check that everything runs (a few minutes):
         python3 classify_bert.py --limit 800 --epochs 1
First real run (no noisy copies = half the data = ~2x faster):
         python3 classify_bert.py --no-noisy --epochs 3
Full run, and the much faster 6-layer alternative:
         python3 classify_bert.py
         python3 classify_bert.py --model nreimers/MiniLM-L6-H384-uncased --lr 1e-4
The first run downloads the model from huggingface.co (worked for Whisper on your machine, so it should work here).
Compare the printed table with classify_baseline.py -- the key columns are accuracy and UNDER-triage.
"""
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import classify_eval as C, ner_eval as E

HERE = Path(__file__).parent


class TwoHead(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name); h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1); self.type_head = nn.Linear(h, len(C.TYPES)); self.urg_head = nn.Linear(h, len(C.LEVELS))

    def forward(self, ids, mask):
        out = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float(); pooled = self.drop((out * m).sum(1) / m.sum(1).clamp(min=1))
        return self.type_head(pooled), self.urg_head(pooled)


def enc(tok, texts, max_len): return tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")


@torch.no_grad()
def predict(model, tok, texts, max_len, bs=64):
    model.eval(); order = sorted(range(len(texts)), key=lambda i: len(texts[i])); pt, pu = [0] * len(texts), [0] * len(texts)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]; b = enc(tok, [texts[j] for j in idx], max_len)
        lt, lu = model(b["input_ids"], b["attention_mask"])
        for j, a, u in zip(idx, lt.argmax(-1).tolist(), lu.argmax(-1).tolist()): pt[j], pu[j] = a, u
    return [C.TYPES[i] for i in pt], [C.LEVELS[i] for i in pu]


def score_split(model, tok, rows, noisy, max_len):
    texts = [r["transcript_asr_noisy" if noisy else "transcript_clean"] for r in rows]; pt, pu = predict(model, tok, texts, max_len)
    sc = [r["scenario_id"] for r in rows]
    return (C.evaluate([r["emergency_type"] for r in rows], pt, C.TYPES, sc),
            C.evaluate([r["urgency_label"] for r in rows], pu, C.LEVELS, sc, True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased"); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=128); ap.add_argument("--no-noisy", action="store_true", help="train on clean text only (2x faster)")
    ap.add_argument("--limit", type=int, default=0, help="use only N training examples (smoke test)"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0); ap.add_argument("--data", default=str(HERE / "output_v3"))
    ap.add_argument("--out", default=str(HERE / "models" / "bert_cls"))
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads or os.cpu_count())
    data = E.load(a.data); train = data["train"]
    ex = [(r["transcript_clean"], r) for r in train]
    if not a.no_noisy: ex += [(r["transcript_asr_noisy"], r) for r in train if r["transcript_asr_noisy"] != r["transcript_clean"]]
    random.shuffle(ex)
    if a.limit: ex = ex[:a.limit]
    X = [t for t, _ in ex]; yt = [C.TYPES.index(r["emergency_type"]) for _, r in ex]; yu = [C.LEVELS.index(r["urgency_label"]) for _, r in ex]
    tok = AutoTokenizer.from_pretrained(a.model); model = TwoHead(a.model); ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(X) + a.batch - 1) // a.batch); sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    print(f"{a.model} | train examples {len(X)} | {steps} steps | threads {torch.get_num_threads()}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True); best, step, t0 = -1.0, 0, time.time()
    for epoch in range(1, a.epochs + 1):
        model.train(); order = list(range(len(X))); random.shuffle(order); run = 0.0
        for i in range(0, len(order), a.batch):
            idx = order[i:i + a.batch]; b = enc(tok, [X[j] for j in idx], a.max_len)
            lt, lu = model(b["input_ids"], b["attention_mask"])
            loss = ce(lt, torch.tensor([yt[j] for j in idx])) + ce(lu, torch.tensor([yu[j] for j in idx]))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
            step += 1; run += loss.item()
            if step % 25 == 0:
                el = time.time() - t0; print(f"  step {step}/{steps}  loss {run / 25:.3f}  elapsed {el / 60:.1f} min  eta {el / step * (steps - step) / 60:.1f} min", flush=True); run = 0.0
        vt = [score_split(model, tok, data["val"], nz, a.max_len) for nz in (False, True)]
        v = float(np.mean([u["macro_f1"] for _, u in vt])); tacc = float(np.mean([t["accuracy"] for t, _ in vt]))
        flag = ""
        if v > best: best = v; torch.save(model.state_dict(), Path(a.out) / "model.pt"); tok.save_pretrained(a.out); flag = "  <- saved"
        print(f"epoch {epoch}: val type acc {tacc:.1%} | urgency macro-F1 {v:.3f} | UNDER-triage {np.mean([u['under_triage'] for _, u in vt]):.1%}{flag}", flush=True)
    model.load_state_dict(torch.load(Path(a.out) / "model.pt")); res = {"args": vars(a)}
    print("\n=== final (best epoch by validation urgency macro-F1)")
    for sp in ("val", "test"):
        for nz in (False, True):
            t, u = score_split(model, tok, data[sp], nz, a.max_len); tag = f"{sp} {'asr-noisy' if nz else 'clean'}"; res[tag] = {"type": t, "urgency": u}
            print(C.line("TYPE    " + tag, t)); print(C.line("URGENCY " + tag, u))
    print("\nurgency confusion (test, asr-noisy):"); C.print_confusion(res["test asr-noisy"]["urgency"])
    json.dump(res, open(HERE / f"classify_results_{Path(a.model).name}.json", "w"), indent=2); print("saved results json")


if __name__ == "__main__":
    main()
