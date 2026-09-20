/* Deterministic retrieval over the shipped record, in the browser.
 *
 * The real assistant has two halves: SQL retrieval that computes counts and
 * filters exactly, and a local language model that phrases the result. Only
 * the second half needs Ollama.
 *
 * This provides the first half for the static demo, over a fixed set of
 * questions. It deliberately does not parse free text: pretending to
 * understand arbitrary questions without the router behind it would be the
 * one thing this project refuses to do. Each answer names the rows it used,
 * and the interface states plainly that no prose was generated.
 */

import { formatScore, formatTime, prettyLabel } from "./format.js";

function safetyEvents(events) {
  return events.filter((event) => event.signal_kind === "safety");
}

export function buildQueries(events) {
  const behaviours = [...new Set(safetyEvents(events).map((event) => event.behaviour))].sort();

  const entities = [
    ...new Set(events.flatMap((event) => event.entities || [])),
  ].sort();

  const queries = [
    {
      id: "count-by-behaviour",
      question: "How many findings of each behaviour are in this record?",
      run: () => {
        const counts = new Map();
        safetyEvents(events).forEach((event) => {
          counts.set(event.behaviour, (counts.get(event.behaviour) || 0) + 1);
        });
        const rows = [...counts.entries()].sort((a, b) => b[1] - a[1]);
        return {
          answer: `${safetyEvents(events).length} safety findings across ${rows.length} behaviours.`,
          rows: rows.map(([behaviour, count]) => ({
            label: prettyLabel(behaviour),
            value: String(count),
          })),
          references: [],
        };
      },
    },
    {
      id: "highest-risk",
      question: "Which finding scored highest, and why?",
      run: () => {
        const ranked = safetyEvents(events)
          .slice()
          .sort((a, b) => (b.risk?.score || 0) - (a.risk?.score || 0));
        const top = ranked[0];
        if (!top) return { answer: "No safety findings in this record.", rows: [], references: [] };
        const components = top.risk?.breakdown?.weighted_contributions || {};
        const rows = Object.entries(components)
          .sort((a, b) => b[1] - a[1])
          .filter(([, value]) => Number(value) > 0)
          .map(([key, value]) => ({
            label: key.replaceAll("_", " "),
            value: formatScore(value),
          }));
        return {
          answer: `${prettyLabel(top.behaviour)} at ${top.timestamp_display}, scoring ${formatScore(
            top.risk?.score
          )} (${top.risk?.category}). The contributing components are listed below, largest first.`,
          rows,
          references: [top],
        };
      },
    },
    {
      id: "risk-distribution",
      question: "How is risk distributed across the record?",
      run: () => {
        const counts = new Map();
        safetyEvents(events).forEach((event) => {
          const category = event.risk?.category || "UNKNOWN";
          counts.set(category, (counts.get(category) || 0) + 1);
        });
        const order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];
        return {
          answer: "Risk categories among safety findings:",
          rows: order
            .filter((category) => counts.has(category))
            .map((category) => ({ label: category, value: String(counts.get(category)) })),
          references: [],
        };
      },
    },
    {
      id: "review-order",
      question: "Which timestamps are worth reviewing first?",
      run: () => {
        const ranked = safetyEvents(events)
          .slice()
          .sort((a, b) => (b.risk?.score || 0) - (a.risk?.score || 0))
          .slice(0, 6);
        return {
          answer: `The ${ranked.length} highest scoring findings, in review order:`,
          rows: ranked.map((event) => ({
            label: `${event.timestamp_display} · ${prettyLabel(event.behaviour)}`,
            value: formatScore(event.risk?.score),
          })),
          references: ranked,
        };
      },
    },
  ];

  behaviours.slice(0, 4).forEach((behaviour) => {
    queries.push({
      id: `show-${behaviour}`,
      question: `Show every ${prettyLabel(behaviour).toLowerCase()} finding.`,
      run: () => {
        const matched = events.filter((event) => event.behaviour === behaviour);
        return {
          answer: `${matched.length} ${
            matched.length === 1 ? "record" : "records"
          } of ${prettyLabel(behaviour)}.`,
          rows: matched.map((event) => ({
            label: `${event.timestamp_display} · ${(event.entities || []).join(", ")}`,
            value: `${formatScore(event.risk?.score)} ${event.risk?.category || ""}`.trim(),
          })),
          references: matched,
        };
      },
    });
  });

  if (entities.length) {
    const entity = entities[0];
    queries.push({
      id: `entity-${entity}`,
      question: `Which findings involve ${entity.replaceAll("_", " ")}?`,
      run: () => {
        const matched = events.filter((event) => (event.entities || []).includes(entity));
        return {
          answer: `${matched.length} records name ${entity.replaceAll("_", " ")}.`,
          rows: matched.map((event) => ({
            label: `${event.timestamp_display} · ${prettyLabel(event.behaviour)}`,
            value: event.signal_kind,
          })),
          references: matched,
        };
      },
    });
  }

  queries.push({
    id: "coverage",
    question: "What does this record not tell me?",
    run: () => ({
      answer:
        "The record covers what the rules detected in image space over this source. It does not establish detector accuracy, real-world distance, physical impact, damage, injury or intent. Pixel measurements are not metres without camera calibration, and track identities can switch under occlusion. Findings are evidence for human review, not decisions.",
      rows: [],
      references: [],
    }),
  });

  return queries;
}

export function summariseRecord(events, duration) {
  const safety = safetyEvents(events);
  return `${events.length} records over ${formatTime(duration)}, of which ${
    safety.length
  } are safety findings.`;
}
