# Operator dashboard (Weeks 10-11)

Put `backend/` and `frontend/` inside your project folder (next to `sangli_nlp.py`, `output_v3/`, `transcribe.py`, ...).

## Run (two terminals, from the project folder)
    source .venv/bin/activate
    pip install fastapi "uvicorn[standard]" python-multipart
    uvicorn backend.main:app --port 8000

    cd frontend
    npm install
    npm run dev          # open http://localhost:5173

Production-style (one port): `cd frontend && npm run build`, then just run uvicorn and open http://localhost:8000

## Try it
1. Click "Replay demo calls": 8 held-out test calls stream in, one every 4 s.
2. Upload one of your recorded clips (audio_out/*.mp3): Whisper (prompt + VAD) -> NER -> classifier.
3. Paste any transcript into the box.

## Active Defaults & Options (environment variables)
Defaults are configured to use the highest-accuracy stack:
    CLASSIFIER=bert     (default: bert) fine-tuned DistilBERT (99.8% type accuracy)
    CAUTIOUS=0.75       (default: 0.75) cuts missed high/critical calls to ~8%
    TAGGER=combo        (default: combo) hybrid rules + spaCy victim disambiguation (100% precision)
    WHISPER_MODEL=small (default: small)

To switch to the lightweight baseline on low-spec hardware:
    CLASSIFIER=tfidf CAUTIOUS=0 TAGGER=rule uvicorn backend.main:app --port 8000

Incidents are stored in incidents.db (SQLite).
