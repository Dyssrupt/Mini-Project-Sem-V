"""
extract_call.py -- one call in, structured incident out (what the dashboard will receive).
    python3 extract_call.py --text "Please send an ambulance. My father has chest pain. We are near Station Chowk, Khanbhag."
    python3 extract_call.py --audio audio_out/fire_1127.m4a            # Whisper (prompt+VAD) -> NER -> incident
    python3 extract_call.py --audio call.wav --tagger hybrid           # uses models/sangli_ner if you trained it
"""
import argparse, json
from pathlib import Path
import ner_eval as E, sangli_nlp as S

HERE = Path(__file__).parent


def transcribe(path, model="small"):
    import transcribe as T
    from faster_whisper import WhisperModel
    names = T.load_place_names()
    m = WhisperModel(model, device="cpu", compute_type="int8")
    segs, _ = m.transcribe(T.load_audio(path), beam_size=5, language="en", initial_prompt=T.build_prompt(names), vad_filter=True)
    return " ".join(s.text.strip() for s in segs)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True); src.add_argument("--text"); src.add_argument("--audio")
    ap.add_argument("--tagger", default="rule", choices=["rule", "spacy", "hybrid", "combo"]); ap.add_argument("--model", default=str(HERE / "models" / "sangli_ner"))
    ap.add_argument("--whisper", default="small"); ap.add_argument("--data", default=str(HERE / "output_v3"))
    a = ap.parse_args()
    gaz, tagger = E.build_tagger(a.tagger, a.model, E.load(a.data)); ext = S.IncidentExtractor(tagger, gaz)
    text = a.text if a.text else transcribe(a.audio, a.whisper)
    inc = ext.extract(text); loc = inc["location"]
    print("TRANSCRIPT:", text, "\n")
    print("LOCATION  :", loc["summary"], f"(area confidence {loc['area_confidence']})", "  ** CONFIRM WITH CALLER **" if loc["needs_confirmation"] else "")
    if loc["unlinked_place_text"]: print("UNMATCHED :", loc["unlinked_place_text"], "(not in gazetteer)")
    print("KEYWORDS  :", ", ".join(inc["keywords"]) or "-")
    print("PEOPLE    :", inc["victim_count"]["text"] + (f" (~{inc['victim_count']['count']})" if inc["victim_count"] and inc["victim_count"]["count"] else "") if inc["victim_count"] else "-")
    t = inc["time"]; print("WHEN      :", (t["text"] + (f"  [{t['minutes_ago']} min ago]" if t["minutes_ago"] is not None else "")) if t else "-")
    print("\nJSON:\n" + json.dumps({k: v for k, v in inc.items() if k != "entities"}, indent=2))


if __name__ == "__main__":
    main()
