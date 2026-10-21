import { useEffect, useMemo, useRef, useState } from "react";

const URG_COLOR = { critical: "#c62828", high: "#ef6c00", medium: "#f9a825", low: "#2e7d32" };
const URG_RANK = { critical: 0, high: 1, medium: 2, low: 3 };
const TYPE_ICON = { fire: "🔥", medical: "🚑", accident: "🚗", crime: "🚓" };
const TYPES = ["fire", "medical", "accident", "crime"];
const LEVELS = ["low", "medium", "high", "critical"];
const ENT_COLOR = { AREA: "#bbdefb", ROAD: "#bbdefb", LANDMARK: "#bbdefb", BUILDING: "#bbdefb", CITY: "#e3f2fd",
  TIME_REF: "#e1bee7", VICTIM_COUNT: "#ffe0b2", EMERGENCY_KEYWORD: "#ffcdd2" };

const api = (path, opts) =>
  fetch("/api" + path, opts).then((r) => {
    if (!r.ok) throw new Error(r.status + " " + r.statusText);
    return r.json();
  });
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const patch = (id, body) => api("/incidents/" + id, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function timeAgo(iso, now) {
  const s = Math.max(0, Math.round((now - new Date(iso).getTime()) / 1000));
  if (s < 60) return s + "s ago";
  if (s < 3600) return Math.floor(s / 60) + "m ago";
  return Math.floor(s / 3600) + "h ago";
}

// live incident list: REST for the initial load, WebSocket for updates (auto-reconnect)
function useIncidents() {
  const [items, setItems] = useState({});
  const [online, setOnline] = useState(false);
  useEffect(() => {
    let ws, timer, closed = false;
    const load = () =>
      api("/incidents").then((list) => setItems(Object.fromEntries(list.map((i) => [i.id, i])))).catch(() => {});
    const connect = () => {
      ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
      ws.onopen = () => { setOnline(true); load(); };
      ws.onmessage = (m) => {
        const msg = JSON.parse(m.data);
        if (msg.event === "incident") setItems((prev) => ({ ...prev, [msg.data.id]: msg.data }));
      };
      ws.onclose = () => { setOnline(false); if (!closed) timer = setTimeout(connect, 2000); };
    };
    connect();
    return () => { closed = true; clearTimeout(timer); if (ws) ws.close(); };
  }, []);
  return [items, online];
}

function Highlighted({ text, entities }) {
  const parts = [];
  let pos = 0;
  [...(entities || [])].sort((a, b) => a.start - b.start).forEach((e, i) => {
    if (e.start < pos) return;
    if (e.start > pos) parts.push(text.slice(pos, e.start));
    const tip = e.label + (e.canonical ? " → " + e.canonical : "");
    parts.push(<mark key={i} title={tip} style={{ background: ENT_COLOR[e.label] || "#eee" }}>{text.slice(e.start, e.end)}</mark>);
    pos = e.end;
  });
  parts.push(text.slice(pos));
  return <p className="transcript">{parts}</p>;
}

function Bars({ probs, order, colors }) {
  return (
    <div className="bars">
      {order.map((k) => (
        <div className="bar-row" key={k}>
          <span className="bar-label">{k}</span>
          <div className="bar-track"><div className="bar-fill" style={{ width: Math.round((probs[k] || 0) * 100) + "%", background: colors ? colors[k] : "#5c6bc0" }} /></div>
          <span className="bar-pct">{Math.round((probs[k] || 0) * 100)}%</span>
        </div>
      ))}
    </div>
  );
}

function Card({ inc, selected, now, onClick }) {
  const u = inc.final_urgency;
  const busy = inc.stage && inc.stage !== "done" && inc.stage !== "error";
  return (
    <div className={"card" + (selected ? " selected" : "") + (inc.status === "resolved" ? " resolved" : "")}
         style={{ borderLeftColor: u ? URG_COLOR[u] : "#9e9e9e" }} onClick={onClick}>
      <div className="card-top">
        <span className="card-title">{inc.final_type ? TYPE_ICON[inc.final_type] + " " + inc.final_type : "…"} <small>#{inc.id}</small></span>
        {u && <span className="badge" style={{ background: URG_COLOR[u] }}>{u}</span>}
      </div>
      <div className="card-loc">{busy ? "⏳ " + inc.stage + "…" : inc.stage === "error" ? "⚠ processing failed" : (inc.location && inc.location.summary) || "location unknown"}</div>
      <div className="card-meta">
        <span>{timeAgo(inc.created_at, now)} · {inc.status}</span>
        {inc.location && inc.location.needs_confirmation && <span className="warn">⚠ confirm location</span>}
      </div>
    </div>
  );
}

function Detail({ inc }) {
  const [notes, setNotes] = useState(inc.notes || "");
  useEffect(() => setNotes(inc.notes || ""), [inc.id]);
  if (!inc.type) {
    return (
      <div className="detail">
        <h2>Incident #{inc.id}</h2>
        <p className="muted">{inc.stage === "error" ? "Error: " + inc.error : "Processing (" + inc.stage + ")…"}</p>
        {inc.transcript && <p className="transcript">{inc.transcript}</p>}
      </div>
    );
  }
  const loc = inc.location || {};
  const act = (body) => patch(inc.id, body).catch((e) => alert(e.message));
  return (
    <div className="detail">
      {inc.priority && inc.status !== "resolved" && <div className="priority">⚠ PRIORITY DISPATCH — {inc.final_urgency.toUpperCase()}</div>}
      <h2>{TYPE_ICON[inc.final_type]} {inc.final_type} <small>#{inc.id} · {inc.source}</small></h2>

      <h3>Transcript</h3>
      <Highlighted text={inc.transcript} entities={inc.entities} />
      <div className="legend">
        {["EMERGENCY_KEYWORD", "LANDMARK", "TIME_REF", "VICTIM_COUNT"].map((l) => <span key={l} style={{ background: ENT_COLOR[l] }}>{l.replace("_", " ").toLowerCase()}</span>)}
      </div>

      <div className="grid">
        <section>
          <h3>Location</h3>
          <p className="big">{loc.summary || "not found in transcript"}</p>
          {loc.area && <p className="muted">Area: {loc.area} (confidence {Math.round((loc.area_confidence || 0) * 100)}%)</p>}
          {loc.needs_confirmation && <p className="warn">⚠ Confirm location with the caller</p>}
          {loc.unlinked_place_text && loc.unlinked_place_text.length > 0 && <p className="warn">Not in gazetteer: {loc.unlinked_place_text.join(", ")}</p>}
        </section>
        <section>
          <h3>Details</h3>
          <p><b>People:</b> {inc.victim_count ? inc.victim_count.text + (inc.victim_count.count ? " (~" + inc.victim_count.count + ")" : "") : "—"}</p>
          <p><b>When:</b> {inc.time ? inc.time.text + (inc.time.minutes_ago != null ? " (" + inc.time.minutes_ago + " min)" : "") : "—"}</p>
          <p><b>Keywords:</b> {(inc.keywords || []).map((k) => <span className="chip" key={k}>{k}</span>)}</p>
          <p className="warn-text">Keywords can be negated ("no flames") — read the transcript.</p>
        </section>
        <section>
          <h3>AI urgency {inc.uncertain && <span className="warn">(uncertain)</span>}</h3>
          <Bars probs={inc.urgency_probs} order={["critical", "high", "medium", "low"]} colors={URG_COLOR} />
        </section>
        <section>
          <h3>AI type</h3>
          <Bars probs={inc.type_probs} order={TYPES} />
        </section>
      </div>

      <h3>Dispatch</h3>
      <p>Suggested: {(inc.dispatch || []).map((d) => <span className="chip dispatch" key={d}>{d}</span>)}</p>
      <div className="row">
        <button className="primary" disabled={inc.status !== "new"} onClick={() => act({ status: "dispatched" })}>{inc.status === "new" ? "Dispatch" : inc.status}</button>
        <button disabled={inc.status === "resolved"} onClick={() => act({ status: "resolved" })}>Mark resolved</button>
        <label>Type <select value={inc.type_override || ""} onChange={(e) => act({ type_override: e.target.value })}>
          <option value="">AI ({inc.type})</option>{TYPES.map((t) => <option key={t}>{t}</option>)}</select></label>
        <label>Urgency <select value={inc.urgency_override || ""} onChange={(e) => act({ urgency_override: e.target.value })}>
          <option value="">AI ({inc.urgency})</option>{LEVELS.map((t) => <option key={t}>{t}</option>)}</select></label>
      </div>
      <h3>Operator notes</h3>
      <textarea rows={3} value={notes} onChange={(e) => setNotes(e.target.value)} onBlur={() => notes !== (inc.notes || "") && act({ notes })} placeholder="Notes are saved when you click away" />
      {inc.demo_truth && <p className="muted">Demo ground truth: {inc.demo_truth.type} / {inc.demo_truth.urgency} / {inc.demo_truth.area}</p>}
      <p className="disclaimer">AI suggestions are advisory — the operator decides. Measured on synthetic test calls: type ≈ 98%; urgency ≈ 57–67% exact, ≈ 96% within one level; about 1 in 4 high/critical calls is rated too low. Place names and times are reliable; keyword spans miss unfamiliar words.</p>
    </div>
  );
}

export default function App() {
  const [items, online] = useIncidents();
  const [selected, setSelected] = useState(null);
  const [now, setNow] = useState(Date.now());
  const [text, setText] = useState("");
  const [busy, setBusy] = useState("");
  const fileRef = useRef(null);
  useEffect(() => { const t = setInterval(() => setNow(Date.now()), 15000); return () => clearInterval(t); }, []);

  const list = useMemo(() => Object.values(items).sort((a, b) => {
    const ra = a.status === "resolved" ? 1 : 0, rb = b.status === "resolved" ? 1 : 0;
    if (ra !== rb) return ra - rb;
    const ua = URG_RANK[a.final_urgency] ?? 2, ub = URG_RANK[b.final_urgency] ?? 2;
    return ua !== ub ? ua - ub : b.id - a.id;
  }), [items]);
  const current = selected && items[selected];

  const run = (label, fn) => { setBusy(label); fn().catch((e) => alert(label + " failed: " + e.message)).finally(() => setBusy("")); };
  const sendText = () => { if (text.trim()) run("Submit", () => post("/calls/text", { text }).then((i) => { setText(""); setSelected(i.id); })); };
  const sendAudio = (e) => {
    const f = e.target.files[0]; if (!f) return;
    const fd = new FormData(); fd.append("file", f);
    run("Upload", () => api("/calls/audio", { method: "POST", body: fd }).then((i) => setSelected(i.id)));
    e.target.value = "";
  };

  return (
    <div className="app">
      <header>
        <div><b>AI Emergency Call Understanding System</b> <span className="muted">· operator dashboard · Sangli</span></div>
        <div className="row">
          <span className={"dot " + (online ? "on" : "off")} /> {online ? "live" : "reconnecting…"}
          <button onClick={() => run("Replay", () => post("/demo/replay?n=8&gap=4", {}))}>▶ Replay demo calls</button>
          <button onClick={() => fileRef.current.click()}>⬆ Upload call audio</button>
          <input ref={fileRef} type="file" accept="audio/*" hidden onChange={sendAudio} />
        </div>
      </header>
      <main>
        <aside>
          <div className="submit">
            <textarea rows={3} value={text} onChange={(e) => setText(e.target.value)} placeholder="…or paste a call transcript" />
            <button onClick={sendText} disabled={!text.trim() || busy}>Analyse transcript</button>
          </div>
          <div className="queue">
            {list.length === 0 && <p className="muted pad">No incidents yet. Replay demo calls or upload audio.</p>}
            {list.map((i) => <Card key={i.id} inc={i} now={now} selected={i.id === selected} onClick={() => setSelected(i.id)} />)}
          </div>
        </aside>
        <section className="pane">{current ? <Detail inc={current} key={current.id} /> : <p className="muted pad">Select an incident.</p>}</section>
      </main>
    </div>
  );
}
