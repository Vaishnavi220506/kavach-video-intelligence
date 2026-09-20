import { useEffect, useMemo, useRef, useState } from "react";
import Icon from "../lib/Icon.jsx";
import Reconstruction from "../components/Reconstruction.jsx";
import { buildQueries } from "../lib/demoRetrieval.js";
import {
  RISK_ORDER,
  evidenceLabel,
  formatEvidenceValue,
  formatScore,
  formatTime,
  prettyLabel,
  riskClass,
  riskSegments,
  signalLabel,
} from "../lib/format.js";

const SECTIONS = [
  { id: "overview", label: "Overview", icon: "overview" },
  { id: "findings", label: "Findings", icon: "findings" },
  { id: "ask", label: "Ask", icon: "ask" },
  { id: "analysis", label: "Analysis", icon: "analysis" },
];

const REVIEW_STATES = ["NEW", "REVIEWED", "FALSE_POSITIVE"];

function RiskReading({ category, score, showScore = true }) {
  const filled = riskSegments(category);
  return (
    <span className={`reading ${riskClass(category)}`}>
      <span className="reading__scale" aria-hidden="true">
        {RISK_ORDER.map((_, index) => (
          <span
            key={index}
            className={`scale__seg ${index < filled ? "is-filled" : ""}`.trim()}
          />
        ))}
      </span>
      {showScore ? (
        <span className="reading__value measure">{formatScore(score)}</span>
      ) : null}
      <span className="sr-only">
        {category} risk, score {formatScore(score)}
      </span>
    </span>
  );
}

/* A capability the current source cannot provide explains itself where the
 * control would be, instead of failing when pressed. */
function Unavailable({ reason, children }) {
  return (
    <div className="unavailable">
      <p className="unavailable__reason">
        <Icon name="caution" size={15} />
        <span>{reason}</span>
      </p>
      {children}
    </div>
  );
}

function FindingRow({ event, selected, onSelect }) {
  return (
    <li className={`finding ${selected ? "is-selected" : ""}`.trim()}>
      <button type="button" onClick={() => onSelect(event)} aria-pressed={selected}>
        <span className="finding__time measure">{event.timestamp_display}</span>
        <span className="finding__body">
          <span className="finding__name">{prettyLabel(event.behaviour)}</span>
          <span className="finding__entities measure">
            {(event.entities || []).join(" · ") || "no linked entities"}
          </span>
        </span>
        {/* A findings list is safety findings by default, so labelling every
            row "safety finding" is noise that costs a line at phone width and
            out-shouts the severity scale. Only a row that deviates is
            marked. */}
        {event.signal_kind === "safety" ? (
          <span className="finding__class finding__class--safety sr-only">
            {signalLabel(event.signal_kind)}
          </span>
        ) : (
          <span className={`finding__class finding__class--${event.signal_kind}`}>
            {signalLabel(event.signal_kind)}
          </span>
        )}
        <RiskReading category={event.risk?.category} score={event.risk?.score} />
      </button>
    </li>
  );
}

function EvidencePanel({ event, source, onReview, busy }) {
  if (!event) {
    return (
      <div className="record record--empty">
        <p>Select a finding to read its record.</p>
      </div>
    );
  }

  const evidence = event.evidence_summary || {};
  const components = event.risk?.breakdown?.components || {};
  const weighted = event.risk?.breakdown?.weighted_contributions || {};
  const rows = Object.keys(components)
    .map((key) => ({
      key,
      raw: Number(components[key]) || 0,
      contribution: Number(weighted[key]) || 0,
    }))
    .sort((a, b) => b.contribution - a.contribution);
  const peak = rows[0]?.contribution || 1;

  return (
    <article className="record">
      <header className="record__head">
        <div>
          <h3>{prettyLabel(event.behaviour)}</h3>
          <p className="record__meta measure">
            {event.event_id} · {event.timestamp_display} ·{" "}
            {event.review_status || "NEW"}
          </p>
        </div>
        <RiskReading category={event.risk?.category} score={event.risk?.score} />
      </header>

      <section className="record__section">
        <h4>Entities</h4>
        <ul className="chips">
          {(event.entities || []).length ? (
            event.entities.map((entity) => (
              <li key={entity} className="chip measure">
                <Icon name={entity.startsWith("person") ? "entity" : "equipment"} size={13} />
                {entity}
              </li>
            ))
          ) : (
            <li className="chip chip--muted">none recorded</li>
          )}
        </ul>
      </section>

      <section className="record__section">
        <h4>Score components</h4>
        <table className="ledger">
          <thead>
            <tr>
              <th scope="col">Component</th>
              <th scope="col">Raw</th>
              <th scope="col">Contribution</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key}>
                <th scope="row">{evidenceLabel(row.key)}</th>
                <td className="measure">{formatScore(row.raw)}</td>
                <td className="ledger__bar-cell">
                  <span className="measure">{formatScore(row.contribution)}</span>
                  <span
                    className="ledger__bar"
                    style={{ width: `${Math.min(100, (row.contribution / peak) * 100)}%` }}
                    aria-hidden="true"
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="record__section">
        <h4>Measured evidence</h4>
        <dl className="readings">
          {Object.entries(evidence).map(([key, value]) => (
            <div key={key}>
              <dt>{evidenceLabel(key)}</dt>
              <dd className="measure">{formatEvidenceValue(value)}</dd>
            </div>
          ))}
        </dl>
      </section>

      {event.prevention?.length ? (
        <section className="record__section">
          <h4>Recommended action</h4>
          {event.prevention.map((item) => (
            <div key={item.rule_id} className="recommendation">
              <p>
                <strong>{item.title}</strong> {item.action}
              </p>
              <p className="measure recommendation__rule">
                {item.rule_id} · root cause: {item.root_cause}
              </p>
            </div>
          ))}
        </section>
      ) : null}

      <section className="record__section">
        <h4>Review</h4>
        <div className="review">
          {REVIEW_STATES.map((status) => (
            <button
              key={status}
              type="button"
              className={
                (event.review_status || "NEW") === status ? "review__set is-active" : "review__set"
              }
              onClick={() => onReview(event, status)}
              disabled={busy}
            >
              {status.replaceAll("_", " ").toLowerCase()}
            </button>
          ))}
        </div>
        {source.mode === "demo" ? (
          <p className="record__note">
            Held in this browser tab only. Persisting review state writes to the local
            SQLite database.
          </p>
        ) : null}
      </section>

      {!source.can("replay") ? (
        <p className="record__note">
          <Icon name="caution" size={14} /> {source.reason("replay")}
        </p>
      ) : null}
    </article>
  );
}

function Distribution({ title, entries, total, note }) {
  const peak = Math.max(1, ...entries.map(([, value]) => value));
  return (
    <section className="panel">
      <h3>{title}</h3>
      <table className="ledger">
        <thead>
          <tr>
            <th scope="col">Key</th>
            <th scope="col">Count</th>
            <th scope="col">Share</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([key, value]) => (
            <tr key={key}>
              <th scope="row">{prettyLabel(key)}</th>
              <td className="measure">{value}</td>
              <td className="ledger__bar-cell">
                <span className="measure">
                  {total ? `${((value / total) * 100).toFixed(0)}%` : "--"}
                </span>
                <span
                  className="ledger__bar"
                  style={{ width: `${(value / peak) * 100}%` }}
                  aria-hidden="true"
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {note ? <p className="panel__note">{note}</p> : null}
    </section>
  );
}

export default function Workspace({ source, video, tracks, onExit, onError }) {
  const [section, setSection] = useState("overview");
  const [selected, setSelected] = useState(null);
  const [behaviourFilter, setBehaviourFilter] = useState("all");
  const [riskFilter, setRiskFilter] = useState("all");
  const [classFilter, setClassFilter] = useState("all");
  const [answers, setAnswers] = useState([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [events, setEvents] = useState(video?.events || []);
  const answerRef = useRef(null);
  const recordRef = useRef(null);

  /* Below the two-column breakpoint the record panel sits under the whole
   * list, so selecting a finding would otherwise leave the reader looking at
   * the row they just tapped with the answer off-screen. */
  const selectAndReveal = (event) => {
    setSelected(event);
    if (window.matchMedia("(max-width: 1080px)").matches) {
      requestAnimationFrame(() => {
        recordRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }
  };

  useEffect(() => {
    setEvents(video?.events || []);
  }, [video]);

  const behaviours = useMemo(
    () => [...new Set(events.map((event) => event.behaviour))].sort(),
    [events]
  );

  const filtered = useMemo(
    () =>
      events.filter((event) => {
        if (behaviourFilter !== "all" && event.behaviour !== behaviourFilter) return false;
        if (riskFilter !== "all" && (event.risk?.category || "") !== riskFilter) return false;
        if (classFilter !== "all" && event.signal_kind !== classFilter) return false;
        return true;
      }),
    [events, behaviourFilter, riskFilter, classFilter]
  );

  const safety = useMemo(
    () => events.filter((event) => event.signal_kind === "safety"),
    [events]
  );

  const queries = useMemo(() => buildQueries(events), [events]);

  const handleReview = async (event, status) => {
    setBusy(true);
    try {
      const updated = await source.review(event.event_id, status);
      setEvents((current) =>
        current.map((item) => (item.event_id === event.event_id ? { ...item, ...updated } : item))
      );
      setSelected((current) =>
        current && current.event_id === event.event_id ? { ...current, ...updated } : current
      );
    } catch (error) {
      onError(error.message);
    } finally {
      setBusy(false);
    }
  };

  const runQuery = (query) => {
    const result = query.run();
    setAnswers((current) => [...current, { question: query.question, ...result }]);
    requestAnimationFrame(() => {
      answerRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    });
  };

  const askLive = async (asked) => {
    setBusy(true);
    setAnswers((current) => [...current, { question: asked, pending: true }]);
    try {
      const result = await source.ask(video.video_id, asked);
      setAnswers((current) => [
        ...current.slice(0, -1),
        {
          question: asked,
          answer: result.answer,
          rows: [],
          references: result.event_references || [],
          generated: result.used_llm,
          latency: result.latency_seconds,
        },
      ]);
    } catch (error) {
      setAnswers((current) => current.slice(0, -1));
      onError(error.message);
    } finally {
      setBusy(false);
    }
  };

  const selectFinding = (event) => {
    setSelected(event);
    if (section !== "findings") setSection("findings");
  };

  const statistics = video?.statistics || {};
  const byBehaviour = Object.entries(statistics.by_behaviour || {}).sort(
    (a, b) => b[1] - a[1]
  );
  const byRisk = RISK_ORDER.filter((category) =>
    Object.prototype.hasOwnProperty.call(statistics.by_risk || {}, category)
  ).map((category) => [category, statistics.by_risk[category]]);

  return (
    <div className="workspace">
      <header className="workspace__head">
        <div className="page workspace__head-inner">
          <button type="button" className="workspace__back" onClick={onExit}>
            <Icon name="back" size={15} />
            The record
          </button>

          <dl className="workspace__file">
            <div>
              <dt>Source</dt>
              <dd className="measure">{video?.name}</dd>
            </div>
            <div>
              <dt>Duration</dt>
              <dd className="measure">{formatTime(video?.duration || 0)}</dd>
            </div>
            <div>
              <dt>Records</dt>
              <dd className="measure">
                {events.length} · {safety.length} safety
              </dd>
            </div>
            <div>
              <dt>Mode</dt>
              <dd className="measure">
                {source.mode === "demo" ? "Committed dataset" : "Local pipeline"}
              </dd>
            </div>
          </dl>
        </div>

        <nav className="workspace__nav page" aria-label="Workspace sections">
          {SECTIONS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={section === item.id ? "is-current" : ""}
              aria-current={section === item.id ? "page" : undefined}
              onClick={() => setSection(item.id)}
            >
              <Icon name={item.icon} size={15} />
              {item.label}
              {item.id === "findings" ? (
                <span className="workspace__count measure">{safety.length}</span>
              ) : null}
            </button>
          ))}
        </nav>
      </header>

      <main id="main" className="page workspace__body">
        {section === "overview" ? (
          <div className="stack">
            {tracks ? (
              <Reconstruction
                tracks={tracks}
                events={events}
                selectedEvent={selected}
                onSelectEvent={selectFinding}
              />
            ) : null}

            <div className="split">
              <section className="panel">
                <h3>What this record contains</h3>
                <dl className="readings">
                  <div>
                    <dt>Frames processed</dt>
                    <dd className="measure">{video?.frames_processed ?? "--"}</dd>
                  </div>
                  <div>
                    <dt>Track observations</dt>
                    <dd className="measure">{video?.track_observations ?? "--"}</dd>
                  </div>
                  <div>
                    <dt>Tracked objects</dt>
                    <dd className="measure">
                      {Object.entries(video?.unique_objects || {})
                        .map(([name, count]) => `${count} ${name}`)
                        .join(", ") || "--"}
                    </dd>
                  </div>
                  <div>
                    <dt>Source geometry</dt>
                    <dd className="measure">
                      {video?.width}&#215;{video?.height} at {video?.fps} fps
                    </dd>
                  </div>
                </dl>
                <p className="panel__note">{video?.model_scope_note}</p>
              </section>

              <section className="panel">
                <h3>Analyse a video</h3>
                {source.can("upload") ? (
                  <p className="panel__note">
                    A local backend is connected. Upload a file or give a direct HTTP(S)
                    video URL to run the full pipeline.
                  </p>
                ) : (
                  <Unavailable reason={source.reason("upload")}>
                    <p className="panel__note">
                      The dataset shown here was produced by the same behaviour, risk and
                      prevention engines, from scripted trajectories rather than
                      detections. Run the project locally to analyse real footage.
                    </p>
                  </Unavailable>
                )}
              </section>
            </div>

            <section className="panel">
              <h3>Highest scoring findings</h3>
              <ul className="findings">
                {safety
                  .slice()
                  .sort((a, b) => (b.risk?.score || 0) - (a.risk?.score || 0))
                  .slice(0, 6)
                  .map((event) => (
                    <FindingRow
                      key={event.event_id}
                      event={event}
                      selected={selected?.event_id === event.event_id}
                      onSelect={selectFinding}
                    />
                  ))}
              </ul>
            </section>
          </div>
        ) : null}

        {section === "findings" ? (
          <div className="findings-layout">
            <div>
              <div className="filters">
                <label>
                  <span>Behaviour</span>
                  <select
                    value={behaviourFilter}
                    onChange={(event) => setBehaviourFilter(event.target.value)}
                  >
                    <option value="all">All behaviours</option>
                    {behaviours.map((behaviour) => (
                      <option key={behaviour} value={behaviour}>
                        {prettyLabel(behaviour)}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Risk</span>
                  <select
                    value={riskFilter}
                    onChange={(event) => setRiskFilter(event.target.value)}
                  >
                    <option value="all">Any risk</option>
                    {RISK_ORDER.map((category) => (
                      <option key={category} value={category}>
                        {category}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Class</span>
                  <select
                    value={classFilter}
                    onChange={(event) => setClassFilter(event.target.value)}
                  >
                    <option value="all">All classes</option>
                    <option value="safety">Safety findings</option>
                    <option value="activity">Activity</option>
                    <option value="anomaly">Anomaly</option>
                  </select>
                </label>
                <p className="filters__count measure">
                  {filtered.length} of {events.length}
                </p>
              </div>

              {filtered.length ? (
                <ul className="findings">
                  {filtered.map((event) => (
                    <FindingRow
                      key={event.event_id}
                      event={event}
                      selected={selected?.event_id === event.event_id}
                      onSelect={selectAndReveal}
                    />
                  ))}
                </ul>
              ) : (
                <p className="panel__note">
                  No records match this filter. Widen it to see the rest of the record.
                </p>
              )}
            </div>

            <div className="findings-layout__record" ref={recordRef}>
              <EvidencePanel
                event={selected}
                source={source}
                onReview={handleReview}
                busy={busy}
              />
            </div>
          </div>
        ) : null}

        {section === "ask" ? (
          <div className="ask">
            <section className="panel">
              <h3>Ask the record</h3>
              {source.can("ask") ? (
                <p className="panel__note">
                  Retrieval runs in SQL first; the local model phrases the retrieved rows
                  and reports insufficient evidence rather than guessing.
                </p>
              ) : (
                <Unavailable reason={source.reason("ask")}>
                  <p className="panel__note">
                    The questions below are answered by deterministic retrieval over the
                    shipped record, computed in your browser. No prose is generated, and
                    every figure comes from the rows themselves.
                  </p>
                </Unavailable>
              )}

              <ul className="queries">
                {queries.map((query) => (
                  <li key={query.id}>
                    <button type="button" onClick={() => runQuery(query)}>
                      {query.question}
                      <Icon name="next" size={14} />
                    </button>
                  </li>
                ))}
              </ul>

              {source.can("ask") ? (
                <form
                  className="ask__form"
                  onSubmit={(formEvent) => {
                    formEvent.preventDefault();
                    const asked = question.trim();
                    if (!asked) return;
                    setQuestion("");
                    askLive(asked);
                  }}
                >
                  <label className="sr-only" htmlFor="ask-input">
                    Ask a question about this record
                  </label>
                  <input
                    id="ask-input"
                    value={question}
                    onChange={(inputEvent) => setQuestion(inputEvent.target.value)}
                    placeholder="Why was this finding scored high?"
                  />
                  <button type="submit" className="stamped" disabled={busy}>
                    Ask
                  </button>
                </form>
              ) : null}
            </section>

            <section className="panel answers" ref={answerRef}>
              <h3>Answers</h3>
              {answers.length === 0 ? (
                <p className="panel__note">
                  Nothing asked yet. Pick a question and the answer appears here with the
                  rows it used.
                </p>
              ) : (
                <ol className="answer-list">
                  {answers.map((entry, index) => (
                    <li key={index} className="answer">
                      <p className="answer__question">{entry.question}</p>
                      {entry.pending ? (
                        <p className="answer__body">Retrieving…</p>
                      ) : (
                        <>
                          <p className="answer__body">{entry.answer}</p>
                          {entry.rows?.length ? (
                            <dl className="readings">
                              {entry.rows.map((row) => (
                                <div key={row.label}>
                                  <dt>{row.label}</dt>
                                  <dd className="measure">{row.value}</dd>
                                </div>
                              ))}
                            </dl>
                          ) : null}
                          {entry.references?.length ? (
                            <ul className="answer__refs">
                              {entry.references.slice(0, 6).map((reference) => (
                                <li key={reference.event_id}>
                                  <button type="button" onClick={() => selectFinding(reference)}>
                                    <span className="measure">
                                      {reference.timestamp_display}
                                    </span>
                                    {prettyLabel(reference.behaviour)}
                                    <Icon name="next" size={13} />
                                  </button>
                                </li>
                              ))}
                            </ul>
                          ) : null}
                          <p className="answer__origin measure">
                            {entry.generated
                              ? `generated from retrieved rows · ${formatScore(entry.latency)}s`
                              : "deterministic retrieval · no text generated"}
                          </p>
                        </>
                      )}
                    </li>
                  ))}
                </ol>
              )}
            </section>
          </div>
        ) : null}

        {section === "analysis" ? (
          <div className="stack">
            <div className="split">
              <Distribution
                title="Records by behaviour"
                entries={byBehaviour}
                total={events.length}
              />
              <Distribution
                title="Safety findings by risk"
                entries={byRisk}
                total={safety.length}
                note={video?.anomaly_scope_note}
              />
            </div>

            <section className="panel">
              <h3>Entities and the findings that name them</h3>
              <p className="panel__note">
                Built from relationships the incident records actually retained. Module 6
                frame-level scene graphs are in-memory and are not persisted, so this is
                not one of them.
              </p>
              <table className="ledger">
                <thead>
                  <tr>
                    <th scope="col">Entity</th>
                    <th scope="col">Findings</th>
                    <th scope="col">Highest risk</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(
                    events.reduce((accumulator, event) => {
                      (event.entities || []).forEach((entity) => {
                        accumulator[entity] = accumulator[entity] || [];
                        accumulator[entity].push(event);
                      });
                      return accumulator;
                    }, {})
                  )
                    .sort((a, b) => b[1].length - a[1].length)
                    .map(([entity, related]) => {
                      const top = related
                        .slice()
                        .sort((a, b) => (b.risk?.score || 0) - (a.risk?.score || 0))[0];
                      return (
                        <tr key={entity}>
                          <th scope="row" className="measure">
                            {entity}
                          </th>
                          <td className="measure">{related.length}</td>
                          <td>
                            <RiskReading
                              category={top?.risk?.category}
                              score={top?.risk?.score}
                            />
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </section>
          </div>
        ) : null}
      </main>
    </div>
  );
}
