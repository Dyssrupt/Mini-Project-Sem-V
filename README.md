# Operator dashboard (Weeks 10-11)

Put `backend/` and `frontend/` inside your project folder (next to `sangli_nlp.py`, `output_v3/`, `transcribe.py`, ...).

## Run (two terminals, from the project folder)
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

## Options (environment variables)
    CLASSIFIER=bert     use the fine-tuned DistilBERT (needs models/bert_cls from classify_bert.py + torch)
    CAUTIOUS=0.75       raise urgency when unsure (cuts missed high/critical calls, many more false alarms)
    WHISPER_MODEL=small
First start takes ~20 s (trains and caches the TF-IDF classifier in models/tfidf_cls.pkl). Incidents are stored in incidents.db (SQLite).
