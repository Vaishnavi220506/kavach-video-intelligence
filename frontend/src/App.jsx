import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "";
const ACTIVITY_TYPES = new Set(["ZONE_TRANSITION", "OBJECT_ACTIVITY"]);
const ANOMALY_TYPES = new Set(["MOTION_ANOMALY"]);

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    throw new Error(payload?.detail || payload || `Request failed (${response.status})`);
  }
  return payload;
}

function formatTime(value = 0) {
  const seconds = Math.max(0, Math.round(Number(value) || 0));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remaining = seconds % 60;
  return hours
    ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
}

function prettyLabel(value = "") {
  return String(value).replaceAll("_", " ").toLowerCase().replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function riskClass(category = "LOW") {
  return `risk-${String(category).toLowerCase()}`;
}

function signalKind(event = {}) {
  if (event.signal_kind) return event.signal_kind;
  if (ANOMALY_TYPES.has(event.behaviour)) return "anomaly";
  if (ACTIVITY_TYPES.has(event.behaviour)) return "activity";
  return "safety";
}

function signalLabel(kind) {
  return kind === "anomaly" ? "Anomaly signal" : kind === "activity" ? "Activity" : "Safety signal";
}

function groupedTimeline(events, bucketSeconds = 0.5) {
  const groups = new Map();
  events.forEach((event) => {
    const bucket = Math.max(0, Math.round((Number(event.timestamp) || 0) / bucketSeconds) * bucketSeconds);
    const key = `${bucket.toFixed(1)}-${signalKind(event)}`;
    const current = groups.get(key);
    if (current) current.count += 1;
    else groups.set(key, { key, event, count: 1 });
  });
  return [...groups.values()].sort((a, b) => Number(a.event.timestamp) - Number(b.event.timestamp));
}

function Icon({ name = "dot", size = 18 }) {
  const paths = {
    grid: "M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z",
    video: "M4 5.5A2.5 2.5 0 0 1 6.5 3h7A2.5 2.5 0 0 1 16 5.5v13a2.5 2.5 0 0 1-2.5 2.5h-7A2.5 2.5 0 0 1 4 18.5zM19 8l3-2v12l-3-2z",
    events: "M5 4h14v16H5zM8 8h8M8 12h8M8 16h5",
    chat: "M4 5.5A3.5 3.5 0 0 1 7.5 2h9A3.5 3.5 0 0 1 20 5.5v6a3.5 3.5 0 0 1-3.5 3.5H11l-5 4v-4.7A3.5 3.5 0 0 1 4 11.5z",
    chart: "M4 19V5M4 19h17M8 16v-5M13 16V7M18 16v-9",
    upload: "M12 16V4m0 0L7 9m5-5 5 5M5 20h14",
    play: "M8 5v14l11-7z",
    pause: "M7 5h3v14H7zM14 5h3v14h-3z",
    back: "M12 5 5 12l7 7M5 12h14",
    forward: "m12 5 7 7-7 7M19 12H5",
    search: "m20 20-4.5-4.5M10.5 18a7.5 7.5 0 1 1 0-15 7.5 7.5 0 0 1 0 15z",
    filter: "M4 6h16M7 12h10M10 18h4",
    external: "M14 4h6v6M20 4l-9 9M18 13v5a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h5",
    check: "m5 12 4 4L19 6",
    refresh: "M20 11a8 8 0 0 0-14.9-3M4 5v4h4M4 13a8 8 0 0 0 14.9 3M20 19v-4h-4",
    alert: "M12 4 3 20h18zM12 9v5m0 3v.01",
    dot: "M12 12m-4 0a4 4 0 1 0 8 0a4 4 0 1 0-8 0",
  };
  return (
    <svg aria-hidden="true" className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d={paths[name] || paths.dot} />
    </svg>
  );
}

function RiskBadge({ category = "LOW", score }) {
  return <span className={`badge ${riskClass(category)}`}><span className="badge-dot" />{category}{score !== undefined ? ` · ${Number(score).toFixed(0)}` : ""}</span>;
}

function EmptyState({ title, body, action }) {
  return <div className="empty-state"><div className="empty-icon"><Icon name="video" size={22} /></div><h3>{title}</h3><p>{body}</p>{action}</div>;
}

function StatCard({ label, value, note, accent = "blue" }) {
  return <div className={`stat-card stat-${accent}`}><span className="stat-label">{label}</span><strong>{value}</strong>{note && <span className="stat-note">{note}</span>}</div>;
}

function EventRow({ event, selected, onSelect, onReplay }) {
  const risk = event.risk || {};
  const kind = signalKind(event);
  return (
    <div className={`event-row ${selected ? "selected" : ""}`}>
      <button className="event-main" onClick={() => onSelect(event)} aria-pressed={selected}>
        <span className="event-time">{event.timestamp_display || formatTime(event.timestamp)}</span>
        <span className="event-copy"><strong>{prettyLabel(event.behaviour)}</strong><small>{(event.entities || []).join(" · ") || "No linked entities"}</small><em className={`signal-tag signal-${kind}`}>{signalLabel(kind)}</em></span>
        <RiskBadge category={risk.category} score={risk.score} />
      </button>
      <button className="icon-button replay-button" onClick={() => onReplay(event)} aria-label={`Play replay for ${event.event_id}`} title="Play evidence replay"><Icon name="play" size={15} /></button>
    </div>
  );
}

function EvidencePanel({ event, onReplay }) {
  if (!event) return <div className="panel-placeholder"><Icon name="events" size={24} /><p>Select an event to inspect its evidence.</p></div>;
  const risk = event.risk || {};
  const evidence = event.evidence_summary || event.evidence || {};
  const breakdown = risk.breakdown?.components || {};
  const kind = signalKind(event);
  const prevention = event.prevention || evidence.prevention || [];
  const snapshots = event.evidence_artifacts || [];
  return (
    <div className="evidence-panel">
      <div className="detail-heading"><div><span className="eyebrow">Selected {kind === "safety" ? "incident" : "signal"}</span><h3>{prettyLabel(event.behaviour)}</h3><span className={`signal-tag signal-${kind}`}>{signalLabel(kind)}</span></div><RiskBadge category={risk.category} score={risk.score} /></div>
      <div className="detail-meta"><span>{event.event_id}</span><span>{event.timestamp_display || formatTime(event.timestamp)}</span><span>{event.review_status || "NEW"}</span></div>
      <div className="evidence-section"><span className="eyebrow">Why it was flagged</span><div className="evidence-grid">{Object.entries(evidence).slice(0, 8).map(([key, value]) => <div className="evidence-item" key={key}><span>{prettyLabel(key)}</span><strong>{typeof value === "boolean" ? (value ? "Yes" : "No") : typeof value === "number" ? value.toFixed(2) : String(value)}</strong></div>)}</div></div>
      <div className="evidence-section"><span className="eyebrow">Risk breakdown</span><div className="breakdown-list">{Object.entries(breakdown).map(([key, value]) => <div className="breakdown-row" key={key}><span>{prettyLabel(key)}</span><div className="bar-track"><span style={{ width: `${Math.min(100, Number(value) || 0)}%` }} /></div><strong>{Number(value).toFixed(0)}</strong></div>)}</div></div>
      {snapshots.length > 0 && <div className="evidence-section"><span className="eyebrow">Evidence snapshot</span><a className="snapshot-link" href={`${API_BASE}${snapshots[0].url}`} target="_blank" rel="noreferrer"><img src={`${API_BASE}${snapshots[0].url}`} alt={`Evidence snapshot for ${event.event_id}`} /><span><Icon name="check" size={14} /> {snapshots[0].verified ? "Integrity verified" : "Integrity check failed"}</span></a></div>}
      {prevention.length > 0 && <div className="evidence-section"><span className="eyebrow">Transparent prevention guidance</span><div className="prevention-card">{prevention.slice(0, 2).map((item) => <div key={item.rule_id || item.title}><strong>{item.title}</strong><p>{item.action}</p><small>{prettyLabel(item.root_cause)} cause · {item.rule_id}</small></div>)}</div></div>}
      <div className="detail-actions"><button className="primary-button" onClick={() => onReplay(event)}><Icon name="play" size={16} /> Play evidence clip</button><span className="helper-text">The clip covers up to 3 seconds before and after the signal.</span></div>
    </div>
  );
}

function VideoPlayer({ data, clip, onClearClip, eventMarkers = [], focusTimestamp = null }) {
  const videoRef = useRef(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(Number(data?.duration || 0));
  const source = clip?.url || (data?.processed_url ? `${API_BASE}${data.processed_url}?v=${data.processing_fps || data.duration || "source"}` : "");
  const seek = (delta) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = Math.max(0, Math.min(videoRef.current.duration || duration, videoRef.current.currentTime + delta));
  };
  const selectTime = (value) => { if (videoRef.current) videoRef.current.currentTime = Number(value); };
  useEffect(() => { setCurrentTime(0); setDuration(Number(data?.duration || 0)); }, [data?.video_id, clip?.url]);
  useEffect(() => {
    const target = clip?.eventOffset ?? focusTimestamp;
    if (target === null || target === undefined || !videoRef.current) return undefined;
    const player = videoRef.current;
    const applyFocus = () => {
      const maximum = Number.isFinite(player.duration) && player.duration > 0 ? player.duration : duration;
      const nextTime = Math.max(0, Math.min(Number(target) || 0, maximum || Number(target) || 0));
      player.currentTime = nextTime;
      setCurrentTime(nextTime);
    };
    if (player.readyState >= 1) {
      applyFocus();
      return undefined;
    }
    player.addEventListener("loadedmetadata", applyFocus, { once: true });
    return () => player.removeEventListener("loadedmetadata", applyFocus);
  }, [clip?.eventOffset, clip?.url, duration, focusTimestamp]);
  return (
    <div className="player-shell">
      <div className="player-topline"><span className="live-dot" />{clip ? <><span>Evidence replay</span><button className="text-button" onClick={onClearClip}>Return to processed video</button></> : <span>Processed analysis video</span>}<span className="player-time">{formatTime(currentTime)} / {formatTime(duration)}</span></div>
      {source ? <>
        <div className="video-frame"><video ref={videoRef} src={source} controls playsInline preload="metadata" onLoadedMetadata={(event) => setDuration(event.currentTarget.duration || duration)} onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)} aria-label={clip ? "KAVACH evidence replay" : "KAVACH processed analysis video"} /></div>
        <div className="transport"><button className="secondary-button" onClick={() => seek(-10)} aria-label="Skip backward 10 seconds"><Icon name="back" size={16} />10 sec</button><input aria-label="Video position" type="range" min="0" max={duration || 1} step="0.1" value={Math.min(currentTime, duration || 1)} onChange={(event) => selectTime(event.target.value)} /><button className="secondary-button" onClick={() => seek(10)} aria-label="Skip forward 10 seconds">10 sec<Icon name="forward" size={16} /></button></div>
        <div className="timeline-markers" aria-label="Evidence markers"><span className="timeline-label">Evidence markers</span><div className="marker-track">{groupedTimeline(eventMarkers).map(({ event, count, key }) => <button key={key} className={`marker ${riskClass(event.risk?.category)}`} style={{ left: `${Math.min(100, Math.max(0, (Number(event.timestamp) / Math.max(1, duration)) * 100))}%` }} title={`${count} signal${count === 1 ? "" : "s"} near ${event.timestamp_display}`} onClick={() => selectTime(event.timestamp)} aria-label={`Jump to ${event.event_id} at ${event.timestamp_display}; ${count} signal${count === 1 ? "" : "s"} in this interval`} />)}</div></div>
      </> : <EmptyState title="No processed video yet" body="Upload a local video or add a direct video URL, then run analysis to create the evidence player." />}
    </div>
  );
}

function UploadCard({ onUpload, onUrl, busy }) {
  const [url, setUrl] = useState("");
  const [mode, setMode] = useState("upload");
  return <div className="source-card"><div className="source-card-heading"><div><span className="eyebrow">Video source</span><h2>Bring in footage</h2></div><span className="source-status"><span className="status-dot" />Local processing</span></div><div className="source-tabs"><button className={mode === "upload" ? "active" : ""} onClick={() => setMode("upload")}><Icon name="upload" size={16} />Upload file</button><button className={mode === "url" ? "active" : ""} onClick={() => setMode("url")}><Icon name="external" size={16} />Direct video URL</button></div>{mode === "upload" ? <label className="dropzone"><Icon name="upload" size={22} /><strong>Choose a video file</strong><span>MP4, MOV, MKV, AVI or WEBM · up to 1 GB</span><input type="file" accept="video/*,.mkv" onChange={(event) => event.target.files?.[0] && onUpload(event.target.files[0])} disabled={busy} /></label> : <div className="url-form"><label htmlFor="video-url">Direct video URL</label><div className="input-row"><input id="video-url" value={url} onChange={(event) => setUrl(event.target.value)} placeholder="https://example.com/warehouse.mp4" /><button className="primary-button" onClick={() => onUrl(url)} disabled={busy || !url.trim()}>{busy ? "Loading..." : "Load URL"}</button></div><span className="helper-text">Only direct HTTP(S) video files are supported; website pages are not downloaded.</span></div>}</div>;
}

function SetupCard({ data, job, onAnalyse, busy, models, modelChoice, onModelChange }) {
  if (!data) return null;
  const running = job?.status === "queued" || job?.status === "running";
  return <div className="setup-card"><div><span className="eyebrow">Ready to analyse</span><h3>{data.name}</h3><p>{data.width}×{data.height} · {data.fps?.toFixed?.(2) || data.fps} FPS · {formatTime(data.duration)}</p><label className="model-select" htmlFor="perception-model"><span>Perception checkpoint</span><select id="perception-model" value={modelChoice} onChange={(event) => onModelChange(event.target.value)} disabled={busy || running}>{(models || []).map((model) => <option key={model.key} value={model.key} disabled={!model.available}>{model.label}{model.available ? "" : " · unavailable"}</option>)}</select><small>{(models || []).find((model) => model.key === modelChoice)?.scope || "Model scope is reported with the analysis."}</small></label></div><button className="primary-button" onClick={onAnalyse} disabled={busy || running}>{running ? `${Math.round((job.progress || 0) * 100)}% processing` : data.processed ? "Re-run analysis" : "Analyse video"}<Icon name="forward" size={16} /></button>{running && <div className="job-progress"><span style={{ width: `${Math.max(2, (job.progress || 0) * 100)}%` }} /><small>{job.message || "Working through frames..."}</small></div>}</div>;
}

function Overview({ data, job, onUpload, onUrl, onAnalyse, onSelectEvent, onReplay, busy, clip, onClearClip, models, modelChoice, onModelChange }) {
  const events = data?.events || [];
  const stats = data?.statistics || {};
  const safetyEvents = events.filter((event) => signalKind(event) === "safety");
  const activityEvents = events.filter((event) => signalKind(event) === "activity");
  const anomalyEvents = events.filter((event) => signalKind(event) === "anomaly");
  const rankEvents = (items) => [...items].sort((a, b) => Number(b.risk?.score || 0) - Number(a.risk?.score || 0) || Number(a.timestamp || 0) - Number(b.timestamp || 0));
  // Keep the overview representative: a long run of one signal type should
  // not hide activity transitions or anomaly evidence from the reviewer.
  const priorityEvents = [
    ...rankEvents(anomalyEvents).slice(0, 2),
    ...rankEvents(safetyEvents).slice(0, 2),
    ...rankEvents(activityEvents).slice(0, 2),
  ];
  const objectCount = data?.unique_objects ? Object.values(data.unique_objects).reduce((total, value) => total + Number(value || 0), 0) : 0;
  return <>
    {!data && <UploadCard onUpload={onUpload} onUrl={onUrl} busy={busy} />}
    {data && !data.processed && <><UploadCard onUpload={onUpload} onUrl={onUrl} busy={busy} /><SetupCard data={data} job={job} onAnalyse={onAnalyse} busy={busy} models={models} modelChoice={modelChoice} onModelChange={onModelChange} /></>}
    {data && data.processed && <SetupCard data={data} job={job} onAnalyse={onAnalyse} busy={busy} models={models} modelChoice={modelChoice} onModelChange={onModelChange} />}
    <div className="page-title"><div><span className="eyebrow">Operations overview</span><h1>{data ? data.name : "Start with a video source"}</h1><p>{data ? "A readable view of what the pipeline found, why it matters, and where to review it." : "The workspace stays empty until you provide footage."}</p></div>{data?.processed && <div className="confidence-note"><span className="status-dot" />Evidence-backed · local CV</div>}</div>
     <div className="stats-grid"><StatCard label="Source duration" value={formatTime(data?.duration)} note={data ? `${data.total_frames || 0} frames` : "Waiting for source"} /><StatCard label="Processing speed" value={data?.processing_fps ? `${data.processing_fps.toFixed(1)} fps` : "—"} note="wall-clock throughput" accent="amber" /><StatCard label="Safety incidents" value={safetyEvents.length} note={stats.average_risk_score ? `avg risk ${Number(stats.average_risk_score).toFixed(0)}/100` : "No incidents yet"} accent="red" /><StatCard label="Activity signals" value={activityEvents.length} note={`${anomalyEvents.length} anomaly signals`} accent="blue" /><StatCard label="Tracked objects" value={objectCount || "—"} note="unique class/ID pairs" accent="green" /></div>
     <div className="workspace-grid"><section className="panel player-panel"><div className="panel-heading"><div><span className="eyebrow">Evidence player</span><h2>Inspect the moment</h2></div><span className="muted">Select an alert to jump to its source time</span></div><VideoPlayer data={data} clip={clip} onClearClip={onClearClip} eventMarkers={events} focusTimestamp={data?.focus_timestamp ?? null} /></section><section className="panel review-panel"><div className="panel-heading"><div><span className="eyebrow">Priority review</span><h2>What needs attention</h2></div><button className="link-button" onClick={() => onSelectEvent("all")}>View all</button></div>{events.length ? <div className="review-list">{priorityEvents.slice(0, 6).map((event) => <EventRow key={event.event_id} event={event} onSelect={onSelectEvent} onReplay={onReplay} />)}</div> : <EmptyState title="No signals stored" body="Run analysis to populate a review queue from structured video evidence." />}</section></div>
    {data?.processed && <div className="lower-grid"><section className="panel"><div className="panel-heading"><div><span className="eyebrow">At a glance</span><h2>Event timeline</h2></div><button className="link-button" onClick={() => onSelectEvent("all")}>Open events</button></div><Timeline events={events} duration={data.duration} onSelect={onSelectEvent} /></section><section className="panel"><div className="panel-heading"><div><span className="eyebrow">Operational readout</span><h2>What the system knows</h2></div></div><div className="readout-list"><div><Icon name="check" size={17} /><span>Detection, tracking, temporal memory, geometry and risk are computed before chat.</span></div><div><Icon name="check" size={17} /><span>Risk score and detection confidence stay separate in every incident.</span></div><div><Icon name="alert" size={17} /><span>Possible behaviours remain cautious image-space evidence, not physical proof.</span></div></div></section></div>}
  </>;
}

function Timeline({ events, duration, onSelect }) {
  if (!events.length) return <div className="mini-empty">No event markers for this video yet.</div>;
  const grouped = groupedTimeline(events);
  return <div className="timeline-card"><div className="timeline-scale"><span>00:00</span><span>{formatTime(duration / 2)}</span><span>{formatTime(duration)}</span></div><div className="timeline-axis">{grouped.map(({ event, count, key }) => <button key={key} className={`timeline-event ${riskClass(event.risk?.category)}`} style={{ left: `${Math.min(98, Math.max(2, (Number(event.timestamp) / Math.max(1, duration)) * 100))}%` }} onClick={() => onSelect(event)} aria-label={`${count} signal${count === 1 ? "" : "s"} near ${event.timestamp_display}`}><span /></button>)}</div><div className="timeline-legend"><span><i className="legend-dot risk-high" />High priority</span><span><i className="legend-dot risk-medium" />Medium</span><span><i className="legend-dot risk-low" />Low</span><span className="muted">{events.length} records · {grouped.length} time clusters</span></div></div>;
}

function EventsView({ data, onReplay, onSelectEvent, selectedEvent: selectedEventProp, onReview }) {
  const [behaviour, setBehaviour] = useState("ALL");
  const [risk, setRisk] = useState("ALL");
  const [kind, setKind] = useState("ALL");
  const [entity, setEntity] = useState("");
  const [reviewStatus, setReviewStatus] = useState("ALL");
  const events = data?.events || [];
  const behaviours = [...new Set(events.map((event) => event.behaviour))];
  const risks = [...new Set(events.map((event) => event.risk?.category).filter(Boolean))];
  const filtered = events.filter((event) => (kind === "ALL" || signalKind(event) === kind) && (behaviour === "ALL" || event.behaviour === behaviour) && (risk === "ALL" || event.risk?.category === risk) && (reviewStatus === "ALL" || event.review_status === reviewStatus) && (!entity.trim() || (event.entities || []).some((value) => value.toLowerCase().includes(entity.toLowerCase()))));
  // When a filter changes, keep the detail panel inside the filtered result set.
  const selectedEvent = selectedEventProp && filtered.some((event) => event.event_id === selectedEventProp.event_id) ? selectedEventProp : filtered[0];
  return <><div className="page-title"><div><span className="eyebrow">Structured evidence store</span><h1>Events & signals</h1><p>Safety incidents, activity transitions and anomaly signals share one timestamped, explainable view.</p></div><div className="count-pill">{filtered.length} shown · {events.length} total</div></div>{data ? <div className="events-layout"><section className="panel event-table-panel"><div className="filter-bar"><label>Signal type<select value={kind} onChange={(event) => setKind(event.target.value)}><option value="ALL">All signal types</option><option value="safety">Safety signals</option><option value="anomaly">Anomaly signals</option><option value="activity">Activity signals</option></select></label><label>Behaviour<select value={behaviour} onChange={(event) => setBehaviour(event.target.value)}><option value="ALL">All behaviours</option>{behaviours.map((value) => <option key={value} value={value}>{prettyLabel(value)}</option>)}</select></label><label>Risk<select value={risk} onChange={(event) => setRisk(event.target.value)}><option value="ALL">All risk levels</option>{risks.map((value) => <option key={value} value={value}>{value}</option>)}</select></label><label>Review<select value={reviewStatus} onChange={(event) => setReviewStatus(event.target.value)}><option value="ALL">All review states</option><option value="NEW">Needs review</option><option value="REVIEWED">Reviewed</option><option value="FALSE_POSITIVE">False positive</option></select></label><label className="search-field">Entity<input value={entity} onChange={(event) => setEntity(event.target.value)} placeholder="person_3 or carton_7" /></label></div><div className="event-list">{filtered.length ? filtered.map((event) => <EventRow key={event.event_id} event={event} selected={selectedEvent?.event_id === event.event_id} onSelect={onSelectEvent} onReplay={onReplay} />) : <EmptyState title="No matching signals" body="Try clearing one of the filters." />}</div></section><section className="panel sticky-detail"><EvidencePanel event={selectedEvent || filtered[0]} onReplay={onReplay} />{(selectedEvent || filtered[0]) && <div className="review-controls"><span>Review status</span>{["NEW", "REVIEWED", "FALSE_POSITIVE"].map((status) => <button key={status} className={(selectedEvent || filtered[0]).review_status === status ? "active" : ""} onClick={() => onReview((selectedEvent || filtered[0]), status)}>{prettyLabel(status)}</button>)}</div>}</section></div> : <EmptyState title="Analyse a video first" body="The evidence store will become available after a video has been processed." />}</>;
}

function AskView({ data, chat, question, setQuestion, onAsk, onReplay, busy }) {
  const suggestions = ["Summarize this video", "Show motion anomalies", "Show activity signals", "What happened around 00:10?", "Show every high risk event", "Why was the highest risk event flagged?"];
  return <><div className="page-title"><div><span className="eyebrow">Grounded local assistant</span><h1>Ask KAVACH</h1><p>Ask in plain language. Python retrieves the stored evidence first; Ollama only explains the supplied records.</p></div><div className="confidence-note"><span className="status-dot" />No raw video sent to the LLM</div></div><div className="ask-layout"><section className="panel chat-panel"><div className="chat-history">{!chat.length && <div className="chat-welcome"><div className="assistant-mark"><Icon name="chat" size={22} /></div><h2>What would you like to review?</h2><p>Ask about timestamps, behaviours, entities, risk, counts or the overall operational picture.</p><div className="suggestions">{suggestions.map((item) => <button key={item} onClick={() => setQuestion(item)}>{item}<Icon name="forward" size={14} /></button>)}</div></div>}{chat.map((message, index) => <div className={`message ${message.role}`} key={`${message.role}-${index}`}><div className="message-label">{message.role === "user" ? "You" : "KAVACH"}</div><div className="message-body">{message.content}</div>{message.role === "assistant" && message.references?.length > 0 && <div className="reference-list"><span className="eyebrow">Verified replay references</span>{message.references.map((reference) => <button className="reference-card" key={reference.event_id} onClick={() => onReplay(reference)}><span className="reference-time">{reference.timestamp_display}</span><span><strong>{prettyLabel(reference.behaviour)}</strong><small>{reference.event_id} · {(reference.entities || []).join(" · ")}</small></span><Icon name="play" size={15} /></button>)}</div>}</div>)}</div><form className="chat-form" onSubmit={(event) => { event.preventDefault(); onAsk(); }}><label className="sr-only" htmlFor="question">Ask KAVACH a question</label><input id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about this video..." disabled={busy || !data} /><button className="primary-button" type="submit" disabled={busy || !question.trim() || !data}>{busy ? "Thinking..." : "Ask"}<Icon name="forward" size={16} /></button></form></section><aside className="panel assistant-note"><div className="assistant-note-icon"><Icon name="check" size={18} /></div><h3>Grounded by design</h3><p>The assistant cannot invent a timestamp, object, count, risk value, damage outcome or incident. Every replay reference below comes from SQLite.</p><div className="rule-list"><span>01 <b>Retrieve</b> stored events</span><span>02 <b>Compute</b> deterministic answers</span><span>03 <b>Explain</b> with local Ollama</span></div></aside></div></>;
}

function BarChart({ values, empty = "No data yet" }) {
  const entries = Object.entries(values || {});
  if (!entries.length) return <div className="mini-empty">{empty}</div>;
  const max = Math.max(...entries.map(([, value]) => Number(value) || 0), 1);
  return <div className="bar-chart">{entries.map(([label, value]) => <div className="chart-row" key={label}><div className="chart-label">{prettyLabel(label)}</div><div className="chart-track"><span style={{ width: `${(Number(value) / max) * 100}%` }} /></div><strong>{value}</strong></div>)}</div>;
}

function EvidenceGraph({ graph, events = [], onSelectEvent }) {
  const allNodes = graph?.nodes || [];
  const entities = allNodes.filter((node) => node.kind === "entity");
  const incidents = allNodes.filter((node) => node.kind === "incident");
  if (!incidents.length) return <div className="mini-empty">Run analysis to build an evidence graph.</div>;
  const entityById = Object.fromEntries(entities.map((node) => [node.id, node]));
  const eventById = Object.fromEntries(events.map((event) => [event.event_id, event]));
  const riskRank = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3, UNKNOWN: 4 };
  const visibleIncidents = [...incidents]
    .sort((a, b) => (riskRank[a.risk] ?? 4) - (riskRank[b.risk] ?? 4) || Number(a.timestamp || 0) - Number(b.timestamp || 0))
    .slice(0, 10);
  const edgeMap = new Map();
  (graph.edges || []).forEach((edge) => {
    if (!edgeMap.has(edge.target)) edgeMap.set(edge.target, []);
    const entity = entityById[edge.source];
    if (entity && !edgeMap.get(edge.target).some((item) => item.id === entity.id)) edgeMap.get(edge.target).push(entity);
  });
  return <div className="evidence-flow" role="list" aria-label="Entity to event evidence relationships"><div className="flow-key"><span className="flow-key-entity">Entity</span><span className="flow-key-arrow">→</span><span className="flow-key-event">Stored event or signal</span></div>{visibleIncidents.map((node) => { const eventId = String(node.id || "").replace(/^event:/, ""); const linked = edgeMap.get(node.id) || []; return <div className="flow-row" role="listitem" key={node.id}><span className="flow-time">{formatTime(node.timestamp)}</span><div className="flow-entities">{linked.length ? linked.map((entity) => <span className="entity-chip" key={entity.id}>{prettyLabel(entity.label || "Entity")}</span>) : <span className="entity-chip muted-chip">No linked entity</span>}</div><span className="flow-arrow">→</span><button type="button" className={`flow-event ${riskClass(node.risk)}`} aria-label={`Inspect ${eventId}`} onClick={() => eventById[eventId] && onSelectEvent?.(eventById[eventId])}><div className="flow-event-top"><strong>{prettyLabel(node.label || "Event")}</strong><RiskBadge category={node.risk || "UNKNOWN"} /></div><small>{eventId}</small></button></div>; })}<p className="helper-text">Showing the 10 highest-priority stored events. Each row reads left-to-right: tracked entity → explainable event. Select a row to open its evidence and replay.</p></div>;
}

function AnalyticsView({ data, onSelectEvent }) {
  if (!data) return <><div className="page-title"><div><span className="eyebrow">Pattern view</span><h1>Analytics</h1><p>Charts are built from the structured event store, not model guesses.</p></div></div><EmptyState title="No analytics yet" body="Analyse a video to populate distributions, timeline markers and the evidence graph." /></>;
  const stats = data.statistics || {};
  return <><div className="page-title"><div><span className="eyebrow">Pattern view</span><h1>Analytics</h1><p>Use this page to understand repeat frequency, risk distribution and where incidents cluster in source time.</p></div><div className="count-pill">{stats.total_events || 0} stored incidents</div></div><div className="analytics-grid"><section className="panel chart-panel"><div className="panel-heading"><div><span className="eyebrow">Frequency</span><h2>Behaviours observed</h2></div></div><BarChart values={stats.by_behaviour} /></section><section className="panel chart-panel"><div className="panel-heading"><div><span className="eyebrow">Risk distribution</span><h2>Risk categories</h2></div></div><BarChart values={stats.by_risk} /></section><section className="panel wide-chart"><div className="panel-heading"><div><span className="eyebrow">Source time</span><h2>Event timeline</h2></div><span className="muted">Click a point to inspect</span></div><Timeline events={data.events || []} duration={data.duration || data.events?.at(-1)?.timestamp || 1} onSelect={onSelectEvent} /></section><section className="panel wide-chart"><div className="panel-heading"><div><span className="eyebrow">Relationships</span><h2>Evidence graph</h2></div></div><EvidenceGraph graph={data.graph} events={data.events || []} onSelectEvent={onSelectEvent} /></section></div></>;
}

function App() {
  const [view, setView] = useState("overview");
  const [videos, setVideos] = useState([]);
  const [videoId, setVideoId] = useState("");
  const [data, setData] = useState(null);
  const [job, setJob] = useState(null);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [clip, setClip] = useState(null);
  const [chat, setChat] = useState([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [models, setModels] = useState([]);
  const [modelChoice, setModelChoice] = useState("bundled");

  const refreshVideos = async () => {
    const result = await api("/api/videos");
    setVideos(result.videos || []);
  };
  const refreshVideo = async (id = videoId) => { if (!id) return; const result = await api(`/api/videos/${id}`); setData(result); setSelectedEvent((current) => current && result.events?.some((event) => event.event_id === current.event_id) ? current : (result.events?.[0] || null)); };
  useEffect(() => { Promise.all([refreshVideos(), api("/api/models").then((result) => setModels(result.models || []))]).catch((reason) => setError(reason.message)); }, []);
  useEffect(() => { if (videoId) refreshVideo(videoId).catch((reason) => setError(reason.message)); }, [videoId]);
  useEffect(() => { if (!videoId || !["queued", "running"].includes(job?.status)) return undefined; const timer = setInterval(async () => { try { const next = await api(`/api/videos/${videoId}/job`); setJob(next); if (next.status === "complete") { await refreshVideo(videoId); await refreshVideos(); } if (next.status === "error") setError(next.message || "Analysis failed"); } catch (reason) { setError(reason.message); } }, 1200); return () => clearInterval(timer); }, [videoId, job?.status]);

  const handleUpload = async (file) => { setBusy(true); setError(""); try { const form = new FormData(); form.append("file", file); const result = await api("/api/videos/upload", { method: "POST", body: form }); setVideoId(result.video_id); setData(result); setSelectedEvent(null); setClip(null); setView("overview"); await refreshVideos(); } catch (reason) { setError(reason.message); } finally { setBusy(false); } };
  const handleUrl = async (url) => { setBusy(true); setError(""); try { const result = await api("/api/videos/url", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }) }); setVideoId(result.video_id); setData(result); setSelectedEvent(null); setClip(null); setView("overview"); await refreshVideos(); } catch (reason) { setError(reason.message); } finally { setBusy(false); } };
  const handleVideoChange = (id) => { setVideoId(id); setData(null); setJob(null); setSelectedEvent(null); setClip(null); setChat([]); setQuestion(""); setError(""); setView("overview"); };
  const handleAnalyse = async () => { if (!videoId) return; setBusy(true); setError(""); try { const next = await api(`/api/videos/${videoId}/analyse`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confidence: 0.25, model: modelChoice }) }); setJob(next); } catch (reason) { setError(reason.message); } finally { setBusy(false); } };
  const handleReplay = async (event) => { const eventId = event.event_id; if (!eventId) return; setBusy(true); setError(""); try { const result = await api(`/api/events/${eventId}/replay`, { method: "POST" }); const eventTimestamp = Number(result.event_timestamp) || Number(event.timestamp) || 0; const eventOffset = Math.max(0, eventTimestamp - Number(result.start_time || 0)); setSelectedEvent(event); setData((current) => current ? { ...current, focus_timestamp: eventTimestamp } : current); setClip({ url: `${API_BASE}${result.clip_url}?v=${Date.now()}`, eventId, eventOffset }); setView("overview"); } catch (reason) { setError(reason.message); } finally { setBusy(false); } };
  const handleAsk = async () => { if (!data || !question.trim()) return; const asked = question.trim(); setQuestion(""); setChat((messages) => [...messages, { role: "user", content: asked }]); setBusy(true); setError(""); try { const result = await api("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ video_id: videoId, question: asked, use_llm: true }) }); setChat((messages) => [...messages, { role: "assistant", content: result.answer, references: result.event_references || [], latency: result.latency_seconds, usedLLM: result.used_llm }]); } catch (reason) { setChat((messages) => [...messages, { role: "assistant", content: `I could not answer from the stored evidence: ${reason.message}` }]); } finally { setBusy(false); } };
  const handleReview = async (event, status) => { try { const result = await api(`/api/events/${event.event_id}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status }) }); setData((current) => current ? { ...current, events: current.events.map((item) => item.event_id === result.event_id ? result : item) } : current); setSelectedEvent(result); } catch (reason) { setError(reason.message); } };
  const selectEvent = (event) => { if (event === "all") { setView("events"); return; } const timestamp = Number(event?.timestamp) || 0; setSelectedEvent(event); setData((current) => current ? { ...current, focus_timestamp: timestamp } : current); setClip(null); setView("overview"); };
  const navItems = [{ id: "overview", label: "Overview", icon: "grid" }, { id: "events", label: "Events", icon: "events" }, { id: "ask", label: "Ask KAVACH", icon: "chat" }, { id: "analytics", label: "Analytics", icon: "chart" }];
  const viewTitle = navItems.find((item) => item.id === view)?.label || "Overview";
  const latest = videos.find((item) => item.video_id === videoId);
  return <div className="app-shell"><a className="skip-link" href="#main-content">Skip to main content</a><aside className="sidebar"><div className="brand"><div className="brand-mark">K</div><div><strong>KAVACH</strong><span>VIDEO INTELLIGENCE</span></div></div><div className="workspace-select"><span className="eyebrow">Workspace</span><button>{latest?.name || "No video selected"}<Icon name="forward" size={14} /></button></div><nav aria-label="Primary navigation">{navItems.map((item) => <button key={item.id} className={view === item.id ? "active" : ""} onClick={() => setView(item.id)}><Icon name={item.icon} size={18} /><span>{item.label}</span>{item.id === "events" && data?.events?.length ? <em>{data.events.length}</em> : null}</button>)}</nav><div className="sidebar-footer"><div className="system-state"><span className="status-dot" /><div><strong>Local pipeline</strong><small>CV + SQLite ready</small></div></div><span className="upstream-note">UPSTREAM: forklift-safety-ai<br />KAVACH contribution layer</span></div></aside><main id="main-content" className="main-content"><header className="topbar"><div><span className="eyebrow">Supervisor workspace / {viewTitle}</span><div className="breadcrumb">KAVACH <span>/</span> {viewTitle}</div></div><div className="topbar-actions"><label className="video-switcher"><span className="sr-only">Select video</span><select value={videoId} onChange={(event) => handleVideoChange(event.target.value)}><option value="">Select video</option>{videos.map((video) => <option key={video.video_id} value={video.video_id}>{video.name}</option>)}</select></label><button className="icon-button" aria-label="Refresh video data" title="Refresh" onClick={() => refreshVideo().catch((reason) => setError(reason.message))}><Icon name="check" size={17} /></button></div></header>{error && <div className="error-banner" role="alert"><Icon name="alert" size={17} /><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss error">×</button></div>}<div className="content-wrap">{view === "overview" && <Overview data={data} job={job} onUpload={handleUpload} onUrl={handleUrl} onAnalyse={handleAnalyse} onSelectEvent={selectEvent} onReplay={handleReplay} busy={busy} clip={clip} onClearClip={() => setClip(null)} models={models} modelChoice={modelChoice} onModelChange={setModelChoice} />}{view === "events" && <EventsView data={data} onReplay={handleReplay} onSelectEvent={selectEvent} selectedEvent={selectedEvent} onReview={handleReview} />}{view === "ask" && <AskView data={data} chat={chat} question={question} setQuestion={setQuestion} onAsk={handleAsk} onReplay={handleReplay} busy={busy} />}{view === "analytics" && <AnalyticsView data={data} onSelectEvent={selectEvent} />}</div></main></div>;
}

export default App;
