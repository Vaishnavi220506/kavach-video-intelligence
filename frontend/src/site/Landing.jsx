import { useMemo } from "react";
import Icon from "../lib/Icon.jsx";
import Reconstruction from "../components/Reconstruction.jsx";
import {
  RISK_ORDER,
  evidenceLabel,
  formatEvidenceValue,
  formatScore,
  formatTime,
  prettyLabel,
  riskClass,
  riskSegments,
} from "../lib/format.js";

const REPO_URL = "https://github.com/Vaishnavi220506/kavach-video-intelligence";

/* The pipeline as the record describes it. Order matters: this is the claim
 * the whole site rests on, that vision writes the record before language ever
 * sees it. */
const STAGES = [
  {
    stage: "Perception",
    module: "detector.py · tracker.py",
    body: "YOLO11n proposes objects frame by frame. ByteTrack gives them short-term identities so an object has a history rather than a series of unrelated boxes.",
  },
  {
    stage: "Memory and geometry",
    module: "object_memory.py · scene_graph.py",
    body: "Each track accumulates a bounded state history. Geometry derives the relations that matter — supported by, near, moving with — and keeps the measurements that produced them.",
  },
  {
    stage: "Behaviour rules",
    module: "registry.py",
    body: "Eleven operational rules read that history against committed thresholds. A rule fires on evidence it can name, and every threshold it used travels with the event.",
  },
  {
    stage: "Risk and incident",
    module: "engine.py · manager.py",
    body: "Scoring decomposes into named components with explicit weights. Repeated firings fold into one incident rather than flooding the queue.",
  },
  {
    stage: "The record",
    module: "database.py",
    body: "Rows in SQLite with JSON evidence. This is the artifact. Everything above it is how the artifact got written.",
  },
  {
    stage: "Reading it back",
    module: "assistant.py · query_router.py",
    body: "Counts and filters are computed by SQL first. The language model receives only retrieved rows, and reports insufficient evidence rather than inventing an answer.",
    downstream: true,
  },
];

const MEASURED = [
  {
    claim: "Every designed behaviour case reproduces",
    value: "22 / 22",
    detail:
      "One positive and one negative case for each of the 11 rules, precision, recall and F1 of 1.00 per rule.",
    limit:
      "A rule sanity check on structured trajectories. It says the rules do what they were specified to do; it says nothing about real warehouse accuracy.",
    source: "reports/controlled_evaluation.json",
  },
  {
    claim: "Two-class detector, measured honestly",
    value: "mAP50 0.636",
    detail:
      "Precision 0.886, recall 0.578, mAP50-95 0.308 for a person and carton checkpoint at 100 epochs.",
    limit:
      "Trained on 15 manually reviewed frames, 10 train and 5 validation. A smoke test of the training loop, not a general accuracy claim, which is why it is not the default model.",
    source: "reports/mvp_v2_detection_metrics.json",
  },
  {
    claim: "End-to-end throughput on one CPU",
    value: "13.5 fps",
    detail:
      "Over the first 30 frames of a 1920×1080, 59.94 fps source. Detection alone measured 19.8 fps across five frames after a warmup.",
    limit:
      "A bounded measurement on one machine. It does not establish real-time performance or deployment-scale requirements. Re-run it on the target hardware.",
    source: "reports/pipeline_benchmark.json",
  },
  {
    claim: "One grounded explanation, locally",
    value: "8.8 s",
    detail: "A single summary request answered by llama3.2:3b through Ollama on CPU.",
    limit:
      "Retrieval is deterministic and fast; the wait is the language model. The explanation is downstream of the record and never analyses raw video.",
    source: "reports/pipeline_benchmark_with_llm.json",
  },
];

const LIMITS = [
  "The default YOLO11n model carries COCO semantics. A model's configured class names do not establish warehouse-class accuracy.",
  "No valid local eight-class warehouse validation dataset is included, so package, pallet, trolley, pallet-truck, forklift and truck support is not claimed.",
  "Track identities can switch under occlusion, missed detections, similar objects or camera motion.",
  "Behaviour thresholds are image-space approximations. Pixel distances are not metres without camera-specific calibration.",
  "A monocular camera does not establish height, impact, damage, injury or intent. “Possible drop” is cautious temporal evidence, not a claim of physical impact.",
  "Default zones are demonstration polygons and need configuring against a real camera.",
  "Live streams, authentication, multi-site operation and deployment-scale monitoring are out of scope.",
  "Human review remains necessary. An event is evidence for a supervisor's decision, never the decision.",
];

function RiskScale({ category }) {
  const filled = riskSegments(category);
  return (
    <span className={`scale ${riskClass(category)}`} aria-hidden="true">
      {RISK_ORDER.map((_, index) => (
        <span
          key={index}
          className={`scale__seg ${index < filled ? "is-filled" : ""}`.trim()}
        />
      ))}
    </span>
  );
}

export default function Landing({ video, tracks, onOpenWorkspace, sourceMode }) {
  /* The worked example is a real row from the dataset, picked as the highest
   * scoring safety finding. Nothing here is written by hand. */
  const worked = useMemo(() => {
    const safety = (video?.events || []).filter(
      (event) => event.signal_kind === "safety" && event.risk?.breakdown?.components
    );
    return safety.sort(
      (a, b) => (b.risk?.score || 0) - (a.risk?.score || 0)
    )[0] || null;
  }, [video]);

  const components = worked?.risk?.breakdown?.components || {};
  const weighted = worked?.risk?.breakdown?.weighted_contributions || {};
  const componentRows = Object.keys(components)
    .map((key) => ({
      key,
      raw: Number(components[key]) || 0,
      contribution: Number(weighted[key]) || 0,
    }))
    .sort((a, b) => b.contribution - a.contribution);

  const evidenceRows = Object.entries(worked?.evidence_summary || {}).slice(0, 6);

  const behaviourCounts = video?.statistics?.by_behaviour || {};
  const findings = Object.entries(behaviourCounts).sort((a, b) =>
    a[0].localeCompare(b[0])
  );

  const safetyCount = (video?.events || []).filter(
    (event) => event.signal_kind === "safety"
  ).length;

  return (
    <main id="main" className="site">
      {/* --- Cover -------------------------------------------------------- */}
      <header className="cover">
        <div className="page cover__inner">
          <div className="cover__masthead">
            <span className="cover__mark" aria-hidden="true">
              K
            </span>
            <span className="cover__wordmark">
              KAVACH
              <small>Explainable temporal video intelligence</small>
            </span>
            <dl className="cover__file">
              <div>
                <dt>Record</dt>
                <dd className="measure">KVC-{String(video?.events?.length || 0).padStart(3, "0")}</dd>
              </div>
              <div>
                <dt>Source</dt>
                <dd className="measure">
                  {video?.width || 1920}&#215;{video?.height || 1080} · {formatTime(video?.duration || 0)}
                </dd>
              </div>
              <div>
                <dt>Findings</dt>
                <dd className="measure">{safetyCount} safety</dd>
              </div>
            </dl>
          </div>

          <h1 className="cover__finding">
            Video of physical operations, turned into a record you can argue with.
          </h1>

          <p className="cover__lede">
            KAVACH watches for eleven operational behaviours, scores what it finds, and
            writes the result as a numbered incident with its evidence attached. Every
            score opens into the components that produced it. Every answer cites the
            timestamp it came from. The language model reads the record; it never decides
            what happened.
          </p>

          <div className="cover__actions">
            <button type="button" className="stamped" onClick={onOpenWorkspace}>
              Open the demo
              <Icon name="next" size={16} />
            </button>
            <a className="quiet-action" href={REPO_URL} target="_blank" rel="noreferrer">
              <Icon name="code" size={16} />
              Read the source
            </a>
          </div>

        </div>

        {/* Plate 1 breaks the fold: the record is visibly already open. */}
        <div className="page cover__plate">
          {tracks ? (
            <Reconstruction tracks={tracks} events={video?.events || []} />
          ) : (
            <div className="plate plate--loading">
              <div className="plate__frame" />
            </div>
          )}
        </div>

        {/* The provenance note sits under the plate because it describes the
            plate: what the reader is looking at, and what it is not. */}
        <div className="page">
          {sourceMode === "demo" ? (
            <p className="cover__provenance">
              This record runs entirely in your browser on a committed dataset. Its
              events, scores and evidence were produced by the real behaviour, risk and
              prevention engines from scripted trajectories — the input is synthetic, the
              analysis is not. Detection is not part of it.
            </p>
          ) : (
            <p className="cover__provenance">
              A local backend is running, so the workspace is connected to the real
              pipeline. You can analyse your own video.
            </p>
          )}
        </div>
      </header>

      {/* --- §1 Mechanism ------------------------------------------------- */}
      <section className="section" aria-labelledby="mechanism-heading">
        <div className="page section__grid">
          <p className="section__folio measure" aria-hidden="true">
            §1
          </p>
          <div className="section__body">
            <h2 id="mechanism-heading">Vision writes the record. Language only reads it back.</h2>
            <p className="prose">
              Most systems in this space put a model at the front and ask it what
              happened. KAVACH inverts that: computer vision and geometry produce
              structured evidence first, and the record they write is the thing that
              exists. A language model is allowed to explain and search it, and nothing
              else.
            </p>

            <ol className="stages">
              {STAGES.map((stage) => (
                <li
                  key={stage.stage}
                  className={`stage ${stage.downstream ? "stage--downstream" : ""}`.trim()}
                >
                  <div className="stage__head">
                    <h3>{stage.stage}</h3>
                    <span className="stage__module measure">{stage.module}</span>
                  </div>
                  <p>{stage.body}</p>
                </li>
              ))}
            </ol>
          </div>
        </div>
      </section>

      {/* --- §2 The worked example ---------------------------------------- */}
      {worked ? (
        <section className="section section--sunk" aria-labelledby="score-heading">
          <div className="page section__grid">
            <p className="section__folio measure" aria-hidden="true">
              §2
            </p>
            <div className="section__body">
              <h2 id="score-heading">A score that cannot be opened is not evidence.</h2>
              <p className="prose">
                Below is an actual row from the dataset the demo ships, not an
                illustration. This is what the interface shows a supervisor when they ask
                why a number is what it is.
              </p>

              <article className="exhibit">
                <header className="exhibit__head">
                  <div>
                    <h3>{prettyLabel(worked.behaviour)}</h3>
                    <p className="exhibit__meta measure">
                      {worked.event_id} · {worked.timestamp_display} ·{" "}
                      {(worked.entities || []).join(", ")}
                    </p>
                  </div>
                  <div className="exhibit__score">
                    <span className="exhibit__score-value measure">
                      {formatScore(worked.risk?.score)}
                    </span>
                    <span className="exhibit__score-category">
                      {worked.risk?.category}
                    </span>
                    <RiskScale category={worked.risk?.category} />
                  </div>
                </header>

                <div className="exhibit__split">
                  <div>
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
                        {componentRows.map((row) => (
                          <tr key={row.key}>
                            <th scope="row">{evidenceLabel(row.key)}</th>
                            <td className="measure">{formatScore(row.raw)}</td>
                            <td className="ledger__bar-cell">
                              <span className="measure">{formatScore(row.contribution)}</span>
                              <span
                                className="ledger__bar"
                                style={{
                                  width: `${Math.min(
                                    100,
                                    (row.contribution /
                                      Math.max(
                                        1,
                                        componentRows[0]?.contribution || 1
                                      )) *
                                      100
                                  )}%`,
                                }}
                                aria-hidden="true"
                              />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="exhibit__note">
                      Detection confidence and risk are separate quantities by design. A
                      confident detection of a low-severity situation stays low risk.
                    </p>
                  </div>

                  <div>
                    <h4>Measured evidence</h4>
                    <dl className="readings">
                      {evidenceRows.map(([key, value]) => (
                        <div key={key}>
                          <dt>{evidenceLabel(key)}</dt>
                          <dd className="measure">{formatEvidenceValue(value)}</dd>
                        </div>
                      ))}
                    </dl>
                    {worked.prevention?.length ? (
                      <div className="recommendation">
                        <h4>Recommended action</h4>
                        <p>
                          <strong>{worked.prevention[0].title}</strong>{" "}
                          {worked.prevention[0].action}
                        </p>
                        <p className="measure recommendation__rule">
                          {worked.prevention[0].rule_id} · root cause:{" "}
                          {worked.prevention[0].root_cause}
                        </p>
                      </div>
                    ) : null}
                  </div>
                </div>
              </article>

              <p className="prose">
                Note <code>ground_plane_calibration_available: false</code> in records like
                this one. The system knows it is measuring pixels, and says so in the
                evidence rather than quietly presenting them as distance.
              </p>
            </div>
          </div>
        </section>
      ) : null}

      {/* --- §3 What it looks for ----------------------------------------- */}
      <section className="section" aria-labelledby="findings-heading">
        <div className="page section__grid">
          <p className="section__folio measure" aria-hidden="true">
            §3
          </p>
          <div className="section__body">
            <h2 id="findings-heading">What the rules look for.</h2>
            <p className="prose">
              Safety findings, activity signals and one image-space anomaly signal. The
              counts are from the shipped dataset, so this index is the actual contents of
              the record rather than a feature list.
            </p>
            <table className="ledger ledger--index">
              <thead>
                <tr>
                  <th scope="col">Behaviour</th>
                  <th scope="col">Class</th>
                  <th scope="col">In this record</th>
                </tr>
              </thead>
              <tbody>
                {findings.map(([behaviour, count]) => {
                  const sample = (video?.events || []).find(
                    (event) => event.behaviour === behaviour
                  );
                  return (
                    <tr key={behaviour}>
                      <th scope="row">{prettyLabel(behaviour)}</th>
                      <td>{sample?.signal_kind || "safety"}</td>
                      <td className="measure">{count}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="exhibit__note">
              Aisle obstruction is implemented and silent here on purpose: its aisle
              polygon is empty until a real camera layout is configured, and firing it
              against a guessed polygon would be misleading.
            </p>
          </div>
        </div>
      </section>

      {/* --- §4 Measured, and not ----------------------------------------- */}
      <section className="section section--sunk" aria-labelledby="measured-heading">
        <div className="page section__grid">
          <p className="section__folio measure" aria-hidden="true">
            §4
          </p>
          <div className="section__body">
            <h2 id="measured-heading">What is measured, and what that measurement is worth.</h2>
            <p className="prose">
              Each figure below is reproducible from the repository. Each one is printed
              beside the reason it is not the claim it might look like.
            </p>

            <div className="measured">
              {MEASURED.map((item) => (
                <article key={item.claim} className="measured__row">
                  <div className="measured__value measure">{item.value}</div>
                  <div className="measured__body">
                    <h3>{item.claim}</h3>
                    <p>{item.detail}</p>
                    <p className="measured__limit">
                      <Icon name="caution" size={15} />
                      <span>{item.limit}</span>
                    </p>
                    <p className="measured__source measure">{item.source}</p>
                  </div>
                </article>
              ))}
            </div>

            <h3 className="limits__heading">Standing limitations</h3>
            <ul className="limits">
              {LIMITS.map((limit) => (
                <li key={limit}>{limit}</li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      {/* --- §5 Run it ---------------------------------------------------- */}
      <section className="section" aria-labelledby="run-heading">
        <div className="page section__grid">
          <p className="section__folio measure" aria-hidden="true">
            §5
          </p>
          <div className="section__body">
            <h2 id="run-heading">Run it against your own video.</h2>
            <p className="prose">
              The full system is local by necessity: video decoding, YOLO inference,
              SQLite and Ollama all need a runtime. The published site is the record and
              the demo; the pipeline runs on your machine.
            </p>
            <pre className="terminal" tabIndex={0}>
              <code>{`git clone ${REPO_URL}.git
cd kavach-video-intelligence
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt

cd frontend && npm install && npm run build && cd ..
python -m uvicorn app.api.main:app --port 8000`}</code>
            </pre>
            <p className="prose">
              Open <code>http://localhost:8000/</code> and the same interface connects to
              the real pipeline. For grounded explanations, run{" "}
              <code>ollama pull llama3.2:3b</code> first. Reproduction steps, model
              provenance and the evaluation workflow are in{" "}
              <a href={`${REPO_URL}/blob/main/docs/REPRODUCIBILITY.md`} target="_blank" rel="noreferrer">
                docs/REPRODUCIBILITY.md
              </a>
              .
            </p>

            <div className="cover__actions">
              <button type="button" className="stamped" onClick={onOpenWorkspace}>
                Open the demo
                <Icon name="next" size={16} />
              </button>
              <a className="quiet-action" href={REPO_URL} target="_blank" rel="noreferrer">
                <Icon name="external" size={16} />
                Repository, docs and reports
              </a>
            </div>
          </div>
        </div>
      </section>

      <footer className="colophon">
        <div className="page colophon__inner">
          <p>
            KAVACH · MIT licensed · built by{" "}
            <a href="https://github.com/Vaishnavi220506" target="_blank" rel="noreferrer">
              Vaishnavi
            </a>
          </p>
          <p className="colophon__note">
            Assets in <code>assets/</code> belong to the original zone-alert prototype and
            are retained under the provenance boundary recorded in{" "}
            <code>docs/KAVACH_BASELINE.md</code>. They are not KAVACH benchmarks.
          </p>
        </div>
      </footer>
    </main>
  );
}
