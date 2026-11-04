import { useEffect, useMemo, useRef, useState } from "react";

const URG_COLOR = { critical: "#c62828", high: "#ef6c00", medium: "#f9a825", low: "#2e7d32", none: "#78909c" };
const URG_RANK = { critical: 0, high: 1, medium: 2, low: 3, none: 4 };
const TYPE_ICON = { fire: "🔥", medical: "🚑", accident: "🚗", crime: "🚓", non_emergency: "🚫" };
const TYPES = ["fire", "medical", "accident", "crime", "non_emergency"];
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
const del = (id) => api("/incidents/" + id, { method: "DELETE" });

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
        if (msg.event === "incident_deleted") {
          setItems((prev) => {
            const next = { ...prev };
            delete next[msg.data.id];
            return next;
          });
        }
        if (msg.event === "all_incidents_deleted") {
          setItems({});
        }
      };
      ws.onclose = () => { setOnline(false); if (!closed) timer = setTimeout(connect, 2000); };
    };
    connect();
    return () => { closed = true; clearTimeout(timer); if (ws) ws.close(); };
  }, []);
  return [items, online, setItems];
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

function Card({ inc, selected, now, onClick, onDelete }) {
  const isNonEmg = inc.non_emergency || inc.final_type === "non_emergency";
  const u = isNonEmg ? "none" : inc.final_urgency;
  const busy = inc.stage && inc.stage !== "done" && inc.stage !== "error";
  return (
    <div className={"card" + (selected ? " selected" : "") + (inc.status === "resolved" ? " resolved" : "")}
         style={{ borderLeftColor: isNonEmg ? "#78909c" : (u ? URG_COLOR[u] : "#9e9e9e") }} onClick={onClick}>
      <div className="card-top">
        <span className="card-title">
          {inc.final_type ? (TYPE_ICON[inc.final_type] || "❓") + " " + (isNonEmg ? "Non-Emergency" : inc.final_type) : "…"} <small>#{inc.id}</small>
        </span>
        <div className="row" style={{ gap: "4px" }}>
          {isNonEmg ? (
            <span className="badge" style={{ background: "#78909c" }}>NON-EMERGENCY</span>
          ) : (
            u && <span className="badge" style={{ background: URG_COLOR[u] }}>{u}</span>
          )}
          <button className="card-del-btn" title="Delete incident" onClick={(e) => onDelete(inc.id, e)}>✕</button>
        </div>
      </div>
      <div className="card-loc">
        {busy ? "⏳ " + inc.stage + "…" : inc.stage === "error" ? "⚠ processing failed" : isNonEmg ? "🚫 " + (inc.non_emergency_reason || "No emergency detected") : (inc.location && inc.location.summary) || "location unknown"}
      </div>
      <div className="card-meta">
        <span>{timeAgo(inc.created_at, now)} · {inc.status}</span>
        {inc.location && inc.location.needs_confirmation && !isNonEmg && <span className="warn">⚠ confirm location</span>}
      </div>
    </div>
  );
}

function Detail({ inc, onDelete }) {
  const [notes, setNotes] = useState(inc.notes || "");
  useEffect(() => setNotes(inc.notes || ""), [inc.id]);
  if (!inc.type) {
    return (
      <div className="detail">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2>Incident #{inc.id}</h2>
          <button className="danger" onClick={() => onDelete(inc.id)}>🗑 Delete Incident</button>
        </div>
        <p className="muted">{inc.stage === "error" ? "Error: " + inc.error : "Processing (" + inc.stage + ")…"}</p>
        {inc.transcript && <p className="transcript">{inc.transcript}</p>}
      </div>
    );
  }
  const isNonEmg = inc.non_emergency || inc.final_type === "non_emergency";
  const loc = inc.location || {};
  const act = (body) => patch(inc.id, body).catch((e) => alert(e.message));
  return (
    <div className="detail">
      {isNonEmg ? (
        <div className="non-emergency-alert">
          <span className="alert-icon">🚫</span>
          <div>
            <strong>Potential Non-Emergency / False Call Detected</strong>
            <p>{inc.non_emergency_reason || "Caller indicates assistance is not needed, false alarm, or trivial inquiry."}</p>
          </div>
        </div>
      ) : (
        inc.priority && inc.status !== "resolved" && <div className="priority">⚠ PRIORITY DISPATCH — {inc.final_urgency.toUpperCase()}</div>
      )}

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "8px" }}>
        <h2>{TYPE_ICON[inc.final_type] || "❓"} {isNonEmg ? "Non-Emergency Call" : inc.final_type} <small>#{inc.id} · {inc.source}</small></h2>
        <button className="danger" onClick={() => onDelete(inc.id)}>🗑 Delete Incident</button>
      </div>

      <h3>Transcript</h3>
      <Highlighted text={inc.transcript} entities={inc.entities} />
      <div className="legend">
        {["EMERGENCY_KEYWORD", "LANDMARK", "TIME_REF", "VICTIM_COUNT"].map((l) => <span key={l} style={{ background: ENT_COLOR[l] }}>{l.replace("_", " ").toLowerCase()}</span>)}
      </div>

      <div className="grid">
        <section>
          <h3>Location</h3>
          <p className="big">{loc.summary || (isNonEmg ? "Not applicable (non-emergency)" : "not found in transcript")}</p>
          {loc.area && <p className="muted">Area: {loc.area} (confidence {Math.round((loc.area_confidence || 0) * 100)}%)</p>}
          {loc.needs_confirmation && !isNonEmg && <p className="warn">⚠ Confirm location with the caller</p>}
          {loc.unlinked_place_text && loc.unlinked_place_text.length > 0 && <p className="warn">Not in gazetteer: {loc.unlinked_place_text.join(", ")}</p>}
        </section>
        <section>
          <h3>Details</h3>
          <p><b>People:</b> {inc.victim_count ? inc.victim_count.text + (inc.victim_count.count ? " (~" + inc.victim_count.count + ")" : "") : "—"}</p>
          <p><b>When:</b> {inc.time ? inc.time.text + (inc.time.minutes_ago != null ? " (" + inc.time.minutes_ago + " min)" : "") : "—"}</p>
          <p><b>Keywords:</b> {(inc.keywords || []).length > 0 ? inc.keywords.map((k) => <span className="chip" key={k}>{k}</span>) : <span className="muted">None detected</span>}</p>
          <p className="warn-text">Keywords can be negated ("no flames") — read the transcript.</p>
        </section>
        <section>
          <h3>AI urgency {inc.uncertain && !isNonEmg && <span className="warn">(uncertain)</span>}</h3>
          {isNonEmg ? (
            <div className="non-emg-box">
              <span className="muted">🚫 No urgency rating (Non-emergency / Fake call)</span>
            </div>
          ) : (
            <Bars probs={inc.urgency_probs} order={["critical", "high", "medium", "low"]} colors={URG_COLOR} />
          )}
        </section>
        <section>
          <h3>AI type</h3>
          {isNonEmg ? (
            <div className="non-emg-box">
              <span className="muted">🚫 Classified as Non-Emergency</span>
            </div>
          ) : (
            <Bars probs={inc.type_probs} order={["fire", "medical", "accident", "crime"]} />
          )}
        </section>
      </div>

      <h3>Dispatch</h3>
      <p>
        {isNonEmg ? (
          <span className="muted">🚫 No emergency dispatch required.</span>
        ) : (
          <>Suggested: {(inc.dispatch || []).map((d) => <span className="chip dispatch" key={d}>{d}</span>)}</>
        )}
      </p>
      <div className="row">
        <button className="primary" disabled={inc.status !== "new" || isNonEmg} onClick={() => act({ status: "dispatched" })}>{inc.status === "new" ? "Dispatch" : inc.status}</button>
        <button disabled={inc.status === "resolved"} onClick={() => act({ status: "resolved" })}>Mark resolved</button>
        <label>Type <select value={inc.type_override || ""} onChange={(e) => act({ type_override: e.target.value })}>
          <option value="">AI ({inc.type})</option>{TYPES.map((t) => <option key={t}>{t}</option>)}</select></label>
        <label>Urgency <select value={inc.urgency_override || ""} onChange={(e) => act({ urgency_override: e.target.value })}>
          <option value="">AI ({inc.urgency})</option>{LEVELS.map((t) => <option key={t}>{t}</option>)}</select></label>
      </div>
      <h3>Operator notes</h3>
      <textarea rows={3} value={notes} onChange={(e) => setNotes(e.target.value)} onBlur={() => notes !== (inc.notes || "") && act({ notes })} placeholder="Notes are saved when you click away" />
      {inc.demo_truth && <p className="muted">Demo ground truth: {inc.demo_truth.type} / {inc.demo_truth.urgency} / {inc.demo_truth.area}</p>}
      <p className="disclaimer">AI suggestions are advisory — the operator decides. Fine-tuned DistilBERT + Cautious Mode: type ≈ 99.8%; urgency within one level ≈ 96%, under-triage reduced to ~8%. Combo NER: 99.8% place-name linking, 100% victim disambiguation.</p>
    </div>
  );
}

export default function App() {
  const [items, online, setItems] = useIncidents();
  const [selected, setSelected] = useState(null);
  const [now, setNow] = useState(Date.now());
  const [text, setText] = useState("");
  const [busy, setBusy] = useState("");
  const fileRef = useRef(null);
  useEffect(() => { const t = setInterval(() => setNow(Date.now()), 15000); return () => clearInterval(t); }, []);

  const list = useMemo(() => Object.values(items).sort((a, b) => {
    const ra = a.status === "resolved" ? 1 : 0, rb = b.status === "resolved" ? 1 : 0;
    if (ra !== rb) return ra - rb;
    const aNon = a.non_emergency || a.final_type === "non_emergency" ? 1 : 0;
    const bNon = b.non_emergency || b.final_type === "non_emergency" ? 1 : 0;
    if (aNon !== bNon) return aNon - bNon;
    const ua = URG_RANK[a.final_urgency] ?? 2, ub = URG_RANK[b.final_urgency] ?? 2;
    return ua !== ub ? ua - ub : b.id - a.id;
  }), [items]);
  const current = selected && items[selected];

  const deleteIncident = (id, e) => {
    if (e) e.stopPropagation();
    if (window.confirm(`Delete Incident #${id}? This will remove it from the dashboard.`)) {
      del(id)
        .then(() => {
          setSelected((curr) => (curr === id ? null : curr));
          setItems((prev) => {
            const next = { ...prev };
            delete next[id];
            return next;
          });
        })
        .catch((err) => alert("Failed to delete incident: " + err.message));
    }
  };

  const deleteAllIncidents = () => {
    if (list.length === 0) return;
    if (window.confirm(`Delete ALL ${list.length} incidents? This will permanently clear the entire dashboard.`)) {
      api("/incidents", { method: "DELETE" })
        .then(() => {
          setSelected(null);
          setItems({});
        })
        .catch((err) => {
          post("/incidents/clear", {})
            .then(() => {
              setSelected(null);
              setItems({});
            })
            .catch(() => alert("Failed to clear incidents: " + err.message + "\n\nNote: Please restart your backend server (Ctrl+C then run uvicorn again) so it picks up the new delete routes."));
        });
    }
  };

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
        <div><b>AI Emergency Call Understanding System</b> <span className="muted">· Operator Dashboard · Sangli</span></div>
        <div className="row">
          <span className={"dot " + (online ? "on" : "off")} /> {online ? "Live" : "Reconnecting…"}
          <button onClick={() => run("Replay", () => post("/demo/replay?n=8&gap=4", {}))}>▶ Replay demo calls</button>
          <button onClick={() => fileRef.current.click()}>⬆ Upload call audio</button>
          <input ref={fileRef} type="file" accept="audio/*" hidden onChange={sendAudio} />
          {list.length > 0 && (
            <button className="danger" onClick={deleteAllIncidents} title="Delete all incidents from the dashboard">🗑 Delete all</button>
          )}
        </div>
      </header>
      <main>
        <aside>
          <div className="submit">
            <textarea rows={3} value={text} onChange={(e) => setText(e.target.value)} placeholder="…or paste a call transcript" />
            <button onClick={sendText} disabled={!text.trim() || busy}>Analyse transcript</button>
          </div>
          <div className="queue-header">
            <span>Incidents ({list.length})</span>
            {list.length > 0 && (
              <button className="danger queue-del-all" onClick={deleteAllIncidents} title="Delete all incidents">🗑 Delete all</button>
            )}
          </div>
          <div className="queue">
            {list.length === 0 && <p className="muted pad">No incidents yet. Replay demo calls or upload audio.</p>}
            {list.map((i) => (
              <Card
                key={i.id}
                inc={i}
                now={now}
                selected={i.id === selected}
                onClick={() => setSelected(i.id)}
                onDelete={deleteIncident}
              />
            ))}
          </div>
        </aside>
        <section className="pane">
          {current ? (
            <Detail inc={current} key={current.id} onDelete={deleteIncident} />
          ) : (
            <p className="muted pad">Select an incident.</p>
          )}
        </section>
      </main>
    </div>
  );
}
