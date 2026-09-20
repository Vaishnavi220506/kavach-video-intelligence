/* One authored icon set: 24px grid, 1.6 stroke, round caps and joins.
 *
 * Drawn rather than borrowed from a glyph font or emoji, and kept to a single
 * stroke weight so a row of icons reads as one instrument rather than a
 * collection. Every icon here is used; none are decorative.
 */

const PATHS = {
  overview: (
    <>
      <path d="M4 6h16" />
      <path d="M4 12h10" />
      <path d="M4 18h13" />
    </>
  ),
  findings: (
    <>
      <path d="M6 3h9l4 4v14H6z" />
      <path d="M15 3v4h4" />
      <path d="M9.5 13h5" />
      <path d="M9.5 17h3" />
    </>
  ),
  ask: (
    <>
      <path d="M20 15a3 3 0 0 1-3 3H9l-5 3V6a3 3 0 0 1 3-3h10a3 3 0 0 1 3 3z" />
      <path d="M9 9h6" />
      <path d="M9 12.5h4" />
    </>
  ),
  analysis: (
    <>
      <path d="M4 20V4" />
      <path d="M4 20h16" />
      <path d="M8 20v-6" />
      <path d="M13 20V9" />
      <path d="M18 20v-9" />
    </>
  ),
  play: <path d="M7 4.5 19 12 7 19.5z" />,
  pause: (
    <>
      <path d="M9 5v14" />
      <path d="M15 5v14" />
    </>
  ),
  replay: (
    <>
      <path d="M4 5v6h6" />
      <path d="M4.6 11a8 8 0 1 1 1.9 7.2" />
    </>
  ),
  next: <path d="M9 5l7 7-7 7" />,
  back: <path d="M15 5l-7 7 7 7" />,
  down: <path d="M5 9l7 7 7-7" />,
  check: <path d="M4.5 12.5 9.5 17.5 19.5 6.5" />,
  close: (
    <>
      <path d="M6 6l12 12" />
      <path d="M18 6L6 18" />
    </>
  ),
  caution: (
    <>
      <path d="M12 3.5 21.5 20H2.5z" />
      <path d="M12 10v4.5" />
      <path d="M12 17.5h.01" />
    </>
  ),
  entity: (
    <>
      <circle cx="12" cy="8" r="3.2" />
      <path d="M5.5 20a6.5 6.5 0 0 1 13 0" />
    </>
  ),
  equipment: (
    <>
      <path d="M3 17h7V7h4l4 5v5h-2" />
      <circle cx="7" cy="19" r="2" />
      <circle cx="17" cy="19" r="2" />
    </>
  ),
  plate: (
    <>
      <rect x="3" y="5" width="18" height="14" rx="1" />
      <path d="M3 16l5-4 3.5 3 3-2.5L21 16" />
      <circle cx="15.5" cy="9.5" r="1.4" />
    </>
  ),
  external: (
    <>
      <path d="M14 4h6v6" />
      <path d="M20 4l-9 9" />
      <path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" />
    </>
  ),
  code: (
    <>
      <path d="M9 7l-5 5 5 5" />
      <path d="M15 7l5 5-5 5" />
    </>
  ),
  scale: (
    <>
      <path d="M3 10h18v5H3z" />
      <path d="M7 10v5" />
      <path d="M11 10v5" />
      <path d="M15 10v5" />
      <path d="M19 10v5" />
    </>
  ),
  stamp: (
    <>
      <path d="M8 3h8v5l2 4H6l2-4z" />
      <path d="M4 16h16v4H4z" />
    </>
  ),
};

export default function Icon({ name, size = 18, className = "" }) {
  const path = PATHS[name];
  if (!path) return null;
  return (
    <svg
      className={`icon ${className}`.trim()}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {path}
    </svg>
  );
}
