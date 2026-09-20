/* Presentation helpers shared by the site and the workspace. */

export function formatTime(value = 0) {
  const total = Math.max(0, Math.round(Number(value) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  const pad = (part) => String(part).padStart(2, "0");
  return hours ? `${pad(hours)}:${pad(minutes)}:${pad(seconds)}` : `${pad(minutes)}:${pad(seconds)}`;
}

/* Behaviour identifiers are stored as SCREAMING_SNAKE_CASE. Sentence case
 * reads better in prose without losing the term. */
export function prettyLabel(value = "") {
  const text = String(value).replaceAll("_", " ").toLowerCase().trim();
  return text ? text[0].toUpperCase() + text.slice(1) : "";
}

export function riskClass(category = "LOW") {
  return `risk-${String(category).toLowerCase()}`;
}

/* Severity as a four-segment printed scale.
 *
 * This is the carrier, not the colour. A reader in greyscale, with a colour
 * vision deficiency, or holding a monochrome printout still gets the value
 * from how many segments are filled. */
export const RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];

export function riskSegments(category = "LOW") {
  const index = RISK_ORDER.indexOf(String(category).toUpperCase());
  return index < 0 ? 1 : index + 1;
}

export function formatScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(1) : "--";
}

/* Evidence keys are engine field names. Expand them into the report's own
 * language without inventing meaning the record does not carry. */
export function evidenceLabel(key = "") {
  return String(key)
    .replaceAll("_px_per_second", " (px/s)")
    .replaceAll("_px_per_second_squared", " (px/s²)")
    .replaceAll("_px", " (px)")
    .replaceAll("_seconds", " (s)")
    .replaceAll("_", " ")
    .replace(/^\w/, (letter) => letter.toUpperCase());
}

export function formatEvidenceValue(value) {
  if (value === null || value === undefined) return "not recorded";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  if (Array.isArray(value)) return value.length ? value.join(", ") : "none";
  if (typeof value === "object") return "see record";
  return String(value);
}

export function signalLabel(kind) {
  if (kind === "anomaly") return "Anomaly signal";
  if (kind === "activity") return "Activity";
  return "Safety finding";
}
