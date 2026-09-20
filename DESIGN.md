---
name: KAVACH
description: A numbered, auditable incident record on report stock — the artifact the system produces, not a page about it.
colors:
  paper: "#f2efe7"
  paper-sunk: "#eae5d9"
  paper-deep: "#e0dacb"
  paper-edge: "#f7f5ef"
  plate: "#17170f"
  plate-sunk: "#0e0e09"
  plate-ink: "#e8e4d6"
  plate-ink-soft: "#a49e8c"
  ink: "#1a1a18"
  ink-2: "#4a463d"
  ink-3: "#6b6558"
  ink-4: "#8c8677"
  rule: "#c9c2b2"
  rule-strong: "#8c8677"
  rule-hair: "#ded8c9"
  stamp: "#b23a2e"
  stamp-deep: "#8d2c22"
  stamp-wash: "rgba(178, 58, 46, 0.08)"
  risk-low: "#4f7a4a"
  risk-medium: "#a8761a"
  risk-high: "#b23a2e"
  risk-critical: "#5a1f2d"
typography:
  display:
    fontFamily: "Spectral, Georgia, 'Times New Roman', serif"
    fontSize: "clamp(2.7rem, 1.8rem + 4vw, 5.2rem)"
    fontWeight: 700
    lineHeight: 1.12
    letterSpacing: "-0.032em"
  headline:
    fontFamily: "Spectral, Georgia, 'Times New Roman', serif"
    fontSize: "clamp(1.75rem, 1.5rem + 1.1vw, 2.5rem)"
    fontWeight: 700
    lineHeight: 1.12
    letterSpacing: "-0.022em"
  title:
    fontFamily: "Spectral, Georgia, 'Times New Roman', serif"
    fontSize: "clamp(1.13rem, 1.06rem + 0.32vw, 1.33rem)"
    fontWeight: 700
    lineHeight: 1.12
    letterSpacing: "-0.022em"
  body:
    fontFamily: "Spectral, Georgia, 'Times New Roman', serif"
    fontSize: "clamp(0.95rem, 0.92rem + 0.15vw, 1.03rem)"
    fontWeight: 400
    lineHeight: 1.65
    letterSpacing: "normal"
  label:
    fontFamily: "Archivo, ui-sans-serif, system-ui, sans-serif"
    fontSize: "clamp(0.78rem, 0.76rem + 0.1vw, 0.84rem)"
    fontWeight: 600
    lineHeight: 1.6
    letterSpacing: "0.08em"
  measure:
    fontFamily: "'JetBrains Mono', ui-monospace, 'Cascadia Mono', monospace"
    fontSize: "clamp(0.78rem, 0.76rem + 0.1vw, 0.84rem)"
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: "normal"
    fontFeature: "tnum 1"
rounded:
  none: "0"
  plate: "2px"
  focus: "1px"
spacing:
  space-1: "0.25rem"
  space-2: "0.5rem"
  space-3: "0.75rem"
  space-4: "1rem"
  space-5: "1.5rem"
  space-6: "2rem"
  space-7: "3rem"
  space-8: "4.5rem"
  space-9: "7rem"
  gutter: "clamp(1rem, 0.6rem + 2vw, 3.5rem)"
components:
  stamped:
    backgroundColor: "{colors.stamp}"
    textColor: "{colors.paper-edge}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "0.85rem 1.35rem"
  stamped-hover:
    backgroundColor: "{colors.stamp-deep}"
    textColor: "{colors.paper-edge}"
  quiet-action:
    textColor: "{colors.ink-2}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "0.85rem 0"
  quiet-action-hover:
    textColor: "{colors.stamp-deep}"
  exhibit:
    backgroundColor: "{colors.paper-edge}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "{spacing.space-5}"
  record:
    backgroundColor: "{colors.paper-edge}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "{spacing.space-4}"
  plate:
    backgroundColor: "{colors.plate}"
    textColor: "{colors.plate-ink}"
    rounded: "{rounded.none}"
  plate-caption:
    backgroundColor: "{colors.plate-sunk}"
    textColor: "{colors.plate-ink-soft}"
    padding: "{spacing.space-3} {spacing.space-4}"
  finding-row:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "{spacing.space-3}"
  finding-row-hover:
    backgroundColor: "{colors.paper-sunk}"
  finding-row-selected:
    backgroundColor: "{colors.stamp-wash}"
    textColor: "{colors.ink}"
  chip:
    backgroundColor: "{colors.paper-deep}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "0.2rem 0.45rem"
  review-set:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink-2}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "0.35rem 0.6rem"
  review-set-active:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
  input-text:
    backgroundColor: "{colors.paper-edge}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "0.7rem 0.7rem"
  select-filter:
    backgroundColor: "{colors.paper-edge}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "0.45rem 0.5rem"
  banner-error:
    backgroundColor: "{colors.stamp}"
    textColor: "{colors.paper-edge}"
    rounded: "{rounded.none}"
    padding: "{spacing.space-3} {spacing.gutter}"
---

# Design System: KAVACH

## Overview

**Creative North Star: "The Evidence Board"**

KAVACH presents itself as the artifact it produces: a numbered, auditable incident record printed on report stock. The page is not a marketing surface that describes an analysis system; it is the analysis output, with a case header, ruled sections carried by marginal folios, an exhibit with its own keyline, ledger tables of contributing components, and stated limits printed beside the figures they qualify rather than banished to footnotes. Structure comes from rules, margins and numbering — the devices a document actually uses — not from a grid of cards.

The ground is light, and that is a scene decision, not a default. The use scene is a supervisor or reviewer reading a printed finding at a desk; a record is read on paper, and paper is what `--paper: #f2efe7` is. The one dark ground in the system belongs to the tipped-in plate (`--plate: #17170f`), where photographic and moving material is mounted with its own caption bar and pixel scale bar. That inversion is the system's contrast device, and it is already doing the work a dark theme would claim to do. A global dark mode would flatten the distinction between the page and the mounted image, which is the single strongest structural signal in the world.

State reads in weight, rule and texture before it reads in colour. Severity is printed as a four-segment scale (`.scale__seg`, four boxes, `riskSegments()` fills one through four), so a CRITICAL finding is still CRITICAL in greyscale, in monochrome print, and to a colour-blind reviewer; hue only reinforces. Emptiness is composed rather than filled: the empty record panel is a dashed keyline with a centred sentence, and an unavailable capability is a hatched field stating why, not a hidden control. The typographic discipline is strict: Spectral sets the document, Archivo is the label voice, JetBrains Mono is reserved for measured quantities only.

**Key Characteristics:**
- Ruled document structure — hairlines, margins, folios and numbering do the work cards would otherwise do
- Zero corner radius everywhere except the 2px plate keyline (`--plate-radius`)
- One accent (`--stamp: #b23a2e`), spent on the primary action and the current selection and nothing else
- Every state survives colour removal
- Photographic and moving material confined to tipped-in plates with their own dark ground
- Mono strictly for measured quantities, with tabular figures so columns align
- One authored motion moment, on the cover, from an already-visible default

## Colors

A report-stock palette: warm off-white paper, four ink greys, three hairline weights, one stamp red, and a four-step severity ramp that only ever reinforces a reading already carried by geometry.

### Primary
- **Exhibit Stamp Red** (`--stamp`, #b23a2e): The single accent. It appears on the stamped primary action, the selected finding's inset marker and wash, the current section tab's baseline rule, the ledger contribution bar (at 0.55 opacity), the caret and `accent-color`, the selection highlight, the focus ring, and the error banner ground. Nothing else may claim it.
- **Stamp Deep** (`--stamp-deep`, #8d2c22): The hover ink of the stamped control, the keyline around it, and the colour of text that reaches for the accent (`a`, `.quiet-action:hover`, `.workspace__back:hover`, the downstream stage heading). Used for text because it carries contrast on paper that `--stamp` does not.
- **Stamp Wash** (`--stamp-wash`, rgba(178,58,46,0.08)): The selected-row ground and the hovered evidence-reference ground. Never used alone — it always ships with an inset marker or a colour change, so selection survives greyscale.

### Secondary
- **Severity Ramp** (`--risk-low` #4f7a4a, `--risk-medium` #a8761a, `--risk-high` #b23a2e, `--risk-critical` #5a1f2d): Applied as `color` on `.risk-*` wrappers and inherited by the segment scale through `currentColor`. `--risk-medium` also inks the caution icon beside a stated limit and beside an unavailable capability. These four are reinforcement only; the segment count is the reading.

### Tertiary
- **Plate Ground** (`--plate` #17170f, `--plate-sunk` #0e0e09) with **Plate Ink** (`--plate-ink` #e8e4d6) and **Plate Ink Soft** (`--plate-ink-soft` #a49e8c): The inverted set, licensed only inside a tipped-in plate or the terminal plate. Selection and focus ring invert with it (`.plate ::selection`, `.plate :focus-visible`).

### Neutral
- **Report Stock** (`--paper`): The page ground, and the sticky workspace header ground so rows pass cleanly beneath it.
- **Sunk Stock** (`--paper-sunk`): Alternating section ground (`.section--sunk`), findings-row hover, and the scrollbar track.
- **Deep Stock** (`--paper-deep`): Inline code ground, chips, tab counts, the evidence-reference row, and the review-button hover.
- **Edge Stock** (`--paper-edge`): The raised surfaces — the exhibit, the record panel, text inputs and filter selects — a half-tone brighter than the page so a bounded figure reads as mounted stock. Also the text colour on stamped and inked grounds.
- **Ink** (`--ink`): Body and heading text, the 2px authority rules (masthead, panel headings, record head, colophon, downstream stage), the active review-state fill, and the plate's outer keyline.
- **Ink 2** (`--ink-2`): Prose and secondary body copy — the lede, section prose, table cells, answers.
- **Ink 3** (`--ink-3`): Labels, captions, metadata, placeholders, notes.
- **Ink 4** (`--ink-4`): The quietest register — section folios, stage numbers, answer provenance.
- **Rule** (`--rule`): The standard hairline — section and exhibit divisions, list tops, chip borders.
- **Rule Strong** (`--rule-strong`): The structural hairline — bounded-figure keylines, table head underline, control borders, scrollbar thumb, hatched-field border.
- **Rule Hair** (`--rule-hair`): The faintest division — between repeated rows in a list, ledger and readings.

### Named Rules
**The One Stamp Rule.** The stamp red marks the primary action and the current selection. It is not a decoration, a heading colour, an icon tint, or a hover on anything that is not reaching for the accent. If a second thing on screen is stamp red without being the primary action or the current selection, one of them is wrong.

**The Greyscale Test.** Print the screen in greyscale. Every state — severity, selection, current tab, active review verdict, signal class, availability — must still be readable. Severity reads as filled segments; selection reads as an inset 2px marker; current reads as a baseline rule; active reads as an inverted fill; unavailable reads as a hatch and a dashed border. Colour is allowed to agree; it is never allowed to be the only witness.

**The Plate Licence Rule.** A dark ground is licensed only where photographic, moving, or terminal material is mounted. Code counts as photographic material here; the terminal block sits on the plate ground for exactly that reason. Nothing else on the page may go dark.

## Typography

**Display Font:** Spectral (self-hosted 400 / 400 italic / 600 / 700; falls back to Georgia, Times New Roman)
**Body Font:** Spectral — the document face sets both the finding and the prose
**Label Font:** Archivo (self-hosted variable 400–700; falls back to ui-sans-serif, system-ui)
**Measured-Quantity Font:** JetBrains Mono (self-hosted variable 400–700; falls back to ui-monospace, Cascadia Mono)

All three families are self-hosted as latin and latin-ext woff2 subsets generated by `scripts/fetch_fonts.py` into `frontend/public/fonts/`, with `font-display: swap`. No webfont CDN, and no system display face standing in for a document face.

**Character:** A serious printed record. Spectral's transitional serif is the setting voice and does all the reading work; Archivo's grotesk appears only in small uppercase labels, so the two never compete at the same size; JetBrains Mono appears only where a number needs to be exact and column-aligned. `font-synthesis-weight: none` is set globally, so only the shipped weights ever render.

### Hierarchy
- **Display** (Spectral 700, `--step-5` clamp 2.7→5.2rem, line-height 1.12, tracking −0.032em, max 30ch): The finding sentence on the cover. One per page.
- **Headline** (Spectral 700, `--step-3` clamp 1.75→2.5rem, max 34ch): Section headings; the exhibit score value uses the same step at weight 600.
- **Title** (Spectral 700, `--step-2` clamp 1.4→1.8rem for the exhibit head; `--step-1` clamp 1.13→1.33rem for stage, panel and record headings).
- **Body** (Spectral 400, `--step-0` clamp 0.95→1.03rem, line-height 1.65; prose at 1.75): Running text, capped at `--measure` (68ch), or `--measure-tight` (54ch) where a column is narrow.
- **Small body** (Spectral 400, `--step--1` clamp 0.78→0.84rem, line-height 1.65–1.7): Ledger tables, readings, limits, notes, provenance.
- **Label** (Archivo 600–700, `--step--1` or 0.62–0.72rem, tracking 0.06–0.12em, uppercase, `--ink-3`): Field names, table heads, section sub-heads, controls, captions. The `.label` utility is the canonical form.
- **Measure** (JetBrains Mono, tabular figures via `font-variant-numeric: tabular-nums` and `font-feature-settings: "tnum" 1`): Applied by the `.measure` class and to `time`, `code`, `kbd`, `samp`; also the stage counters (`decimal-leading-zero`) and the plate caption mark.

### Named Rules
**The Measured-Quantity Rule.** JetBrains Mono is reserved for values that were measured: timecodes, pixel distances, speeds, scores, identifiers, exhibit numbers. It is never applied to prose, headings, labels, buttons, or navigation to make something look technical. If the text is not a number or an identifier, it is not set in mono.

**The No-Kicker Rule.** Nothing stacks above a heading. A section's number lives in the margin as a folio (`.section__folio`, a 4rem grid column beside the body), which is how a document numbers itself for citation. An uppercase label above a heading is an eyebrow and is not part of this system — including at the mobile breakpoint, where the folio moves inline but keeps its quiet document-face treatment rather than becoming a label.

**The Measure Rule.** Running text never exceeds `--measure` (68ch). A headline never exceeds 34ch; the cover finding never exceeds 30ch. Long lines are how a record starts reading as a web page.

## Layout

The model is a page with margins, not a card grid. `.page` is `min(100% - var(--gutter) * 2, var(--page-max))` centred, with `--page-max: 78rem` and a fluid `--gutter` of clamp(1rem, 0.6rem + 2vw, 3.5rem). Minimum supported width is 320px (`body { min-width: 320px }`).

Vertical rhythm comes from a nine-step scale (`--space-1` 0.25rem through `--space-9` 7rem). Sections breathe on `--space-8` (4.5rem) `padding-block`, with `--space-6`/`--space-7` inside sections and `--space-3`/`--space-4` inside dense rows. Sections are separated by a `--rule-hair` hairline; alternating grounds (`.section--sunk`) mark a change of register.

Recurring column structures, all CSS grid, all with explicit proportions:
- **Folio + body**: `4rem minmax(0, 1fr)`, gap `--space-5` — every public section.
- **Stage row**: `2.5rem minmax(0, 1fr)`, with a `decimal-leading-zero` counter in the first column.
- **Measured figure row**: `9rem minmax(0, 1fr)` — the number, then its reading and its stated limit.
- **Exhibit split**: `1.15fr / 0.85fr`, divided by a `--rule` left border.
- **Findings workspace**: `1.25fr / 0.9fr`, with the record panel sticky at `top: 8.5rem` beneath the sticky header.
- **Finding row**: `4.2rem minmax(0, 1fr) auto auto` — time, body, class, risk reading.
- **Ask**: two equal columns.
- **Workspace panels**: `.split` is `repeat(auto-fit, minmax(min(100%, 22rem), 1fr))` at `--space-6` gap.

Breakpoints are content-driven, not device-driven: **1080px** collapses the findings and ask splits to one column and un-sticks the record panel; **900px** collapses the folio grid, the exhibit split (its divider becomes a top rule) and the cover file block; **720px** rewraps the finding row so class and risk fall under the body; **640px** collapses the measured and stage rows, stacks the exhibit head, drops the plate scale bar, and reflows the plate transport.

### Named Rules
**The Ruled-Structure Rule.** Grouping is expressed by a hairline and a margin, not by a box. `--rule-hair` separates repeated rows, `--rule` separates parts of a figure, `--rule-strong` bounds a figure, and a 2px `--ink` rule marks an authority boundary (masthead, panel heading, record head, the downstream stage, the limits heading, the colophon). Four weights, four meanings; do not invent a fifth.

## Elevation & Depth

Depth here is print depth, and there are exactly two elevations plus the page. Surfaces are flat by default; the page itself carries a faint tooth from two very low-contrast fixed radial gradients on `body::before` (no texture image is downloaded). Bounded figures — the exhibit, the populated record panel — sit slightly proud on `--lift-raised`. A tipped-in plate sits proudest on `--lift-plate`, because it is literally a separate piece of stock mounted onto the page. No glow, no coloured shadow, no hard offset shadow, and no shadow used to indicate state.

### Shadow Vocabulary
- **Raised figure** (`--lift-raised`: `0 1px 1px rgba(26,26,24,0.07), 0 4px 12px -6px rgba(26,26,24,0.25)`): A bounded document figure on the page — the exhibit, the record panel.
- **Tipped-in plate** (`--lift-plate`: `0 1px 2px rgba(26,26,24,0.1), 0 10px 24px -12px rgba(26,26,24,0.35)`): Mounted photographic, moving, or terminal material only.
- **Stamp keyline** (`inset 0 0 0 1px rgba(242,239,231,0.28)` on `.stamped`): Not elevation — the double edge a rubber stamp leaves.
- **Selection marker** (`inset 2px 0 0 var(--stamp)` on `.finding.is-selected > button`): Not elevation — the greyscale-legible half of the selected state.

### Named Rules
**The Two-Lift Rule.** There are two lifts and no others. A new surface either lies on the page (flat), is a bounded figure (`--lift-raised`), or is mounted material (`--lift-plate`). Hover and focus never add elevation; they change ground, rule colour, or position by at most 1–3px.

## Shapes

Square by default. `border-radius` is `0` everywhere the system authored it — buttons, chips, inputs, selects, panels, tables, the scrub thumb — because a printed form has no rounded corners, and inputs explicitly reset the browser's own radius. The only radii in the system are `--plate-radius: 2px` (the plate keyline), the `1px` softening on the global focus ring, and the `99px` scrollbar thumb, which is browser furniture themed rather than a design element.

Form language is hairline-and-fill: a 1px border defines every bounded thing, and fill is reserved for inked states. Recurring silhouettes are rectangular and small — the 2.6rem bordered square holding the `K` wordmark (a binding brand commitment from PRODUCT.md, kept and set in the label voice rather than redrawn), the 11×7px severity segment, the 9px square event marker on the plate track, the 3×17px scrub thumb, the 2.1rem square play control.

Icons are the authored set in `frontend/src/lib/Icon.jsx`: a 24px `viewBox`, 1.6 stroke, round caps and joins, `fill: none`, `stroke: currentColor`, rendered `aria-hidden` and non-focusable at a caller-supplied size (16–18px in practice). Every icon in the set is in use and none are decorative.

### Named Rules
**The Square-Corner Rule.** New surfaces get `border-radius: 0`. The only exception in the system is the 2px plate keyline. A rounded card belongs to a different world.

**The One-Instrument Icon Rule.** Icons are drawn into `Icon.jsx` on the 24px grid at 1.6 stroke with round caps, so a row of icons reads as one instrument. No icon fonts, no emoji, no glyph characters standing in for icons, no second stroke weight, no borrowed icon package.

## Components

### Buttons
- **Shape:** Square (`border-radius: 0`).
- **Stamped (primary):** An inked impression — stamp ground, edge-stock text, a `--stamp-deep` 1px keyline plus a translucent inset keyline for the double edge a stamp leaves; Archivo 700, `--step--1`, tracking 0.12em, uppercase; padding `0.85rem 1.35rem`; a trailing 16px icon.
- **Hover / Active:** Ground deepens to `--stamp-deep` with `translateY(-1px)`, and the trailing icon slides `translateX(3px)`; active returns to `translateY(0)`. Both run `--duration-fast` (180ms) on `--ease-out`.
- **Quiet action (secondary):** Text-only in the label voice at `--ink-2` with a `--rule-strong` bottom border; on hover the text goes `--stamp-deep` and the border goes stamp. Padding `0.85rem 0`, so it shares a baseline with the stamped control without claiming a box.
- **Back control:** Same label voice; hover moves the leading icon `translateX(-3px)`.
- One stamped control per view. A second primary action is a design error, not a variant.

### Chips
- **Style:** `--paper-deep` ground, 1px `--rule` border, square, padding `0.2rem 0.45rem`, 0.72rem, `--ink` text. Chips are read-only evidence labels (entities, components), not controls.
- **Muted variant:** `--ink-3` text in the document face, for chips carrying prose rather than an identifier.
- **Signal class marks** (`.finding__class`) are a separate family and carry their state in the border, not the fill: safety takes a stamp border with `--stamp-deep` text, anomaly takes `border-style: dashed`, activity takes a lighter `--rule` border with `--ink-3` text. All three survive greyscale.

### Cards / Containers
There are no cards. Bounded figures exist in two forms, both keylined and neither rounded:
- **Exhibit** (`.exhibit`): edge-stock ground, 1px `--rule-strong` keyline, `--lift-raised`, `--space-5` padding, head divided from body by a `--rule` hairline. There is exactly one exhibit on the public record.
- **Record panel** (`.record`): the workspace's per-finding readout. Same ground and keyline; head bounded by a 2px `--ink` rule; internal sections at `--space-4` divided by `--rule-hair`, with the last section's rule removed.
- **Empty record** (`.record--empty`): no ground, no shadow, dashed border, 18rem min-height, one centred sentence in `--ink-3`. Emptiness is composed, not filled and not hidden.

### Inputs / Fields
- **Style:** Edge-stock ground, 1px `--rule-strong` border, explicit `border-radius: 0`, `font: inherit`. Text inputs pad `0.7rem`; filter selects pad `0.45rem 0.5rem`. Placeholders are `--ink-3`.
- **Labels:** Always visible above the field in the label voice at 0.64rem / 0.1em uppercase. No placeholder-as-label.
- **Focus:** The global ring — `2px solid var(--stamp)` at `outline-offset: 3px`, `border-radius: 1px` — with no border or ground change. Inside a plate the ring inverts to `--plate-ink`.
- **Disabled:** `opacity: 0.5` with `cursor: not-allowed` (see `.review__set:disabled`).
- **Unavailable capability** (`.unavailable`): a 135° hatched field at 13% `--rule-strong`, dashed `--rule-strong` border, a caution icon in `--risk-medium`, and a sentence naming why the region is not in force. Capabilities that are off are printed as off, never removed.

### Navigation
- **Section index** (`.workspace__nav`): horizontally scrollable ruled tabs in the label voice (Archivo 600, `--step--1`, tracking 0.06em, `--ink-3`), each with a 2px transparent bottom border. Hover darkens to `--ink`; `.is-current` takes `--ink` text plus a stamp baseline rule. Never a filled pill. The scrollbar is hidden on this strip only.
- **Workspace header:** sticky at `top: 0`, page ground, 1px `--rule-strong` bottom rule, carrying the back control, the tabs, and the file block (`dt` label / `dd` value pairs, right-aligned above 720px).
- **Tab counts** (`.workspace__count`): a small `--paper-deep` block beside a tab label, 0.7rem, square.

### Signature Component: The Tipped-In Plate
The one dark-ground component, and the only place moving or photographic material may appear. A plate is plate ground with a 1px `--ink` outer keyline and `--lift-plate`, and it always carries:
- **Frame** (`--plate-sunk`) holding the canvas; while loading, a 135° 12px repeating hatch at 16:9 instead of a spinner.
- **Scale bar**, bottom-left, on its own 78%-opaque dark backing so it stays legible over any frame, stating that coordinates are image-space pixels. Hidden below 640px.
- **Transport**: a square 2.1rem play control with a translucent keyline that inverts on hover; a hand-drawn range input (both `-webkit-` and `-moz-` tracks and thumbs themed; 3px track, 3×17px square thumb) because the browser default renders as a bright slab on a dark plate; event markers as 9px squares below the track so they never sit under the thumb, scaling 1.5× on hover; a mono clock in `--plate-ink-soft`.
- **Caption bar** (`--plate-sunk`): Archivo 0.72rem in `--plate-ink-soft`, with a mono, uppercase, 0.08em-tracked exhibit mark in `--plate-ink`. `.plate--compact` suppresses the caption where the enclosing record already names the plate.
- Selection and focus ring invert inside `.plate`.

### Signature Component: The Severity Scale
Four 11×7px boxes with a 1px `currentColor` border and a 2px gap; `riskSegments()` in `frontend/src/lib/format.js` fills one for LOW through four for CRITICAL, and the wrapper's `.risk-*` class sets `currentColor` to the matching ramp value. The scale is `aria-hidden`; a `.sr-only` sentence carries "`{category}` risk, score `{score}`", and the numeric score prints beside it in mono. This component is the reason severity never needs hue.

### Ledger Table
Zero radius, `border-collapse: collapse`, `--step--1`. The head row is the label voice at 0.68rem / 0.1em uppercase over a `--rule-strong` rule; body rows divide on `--rule-hair` with the last row's rule removed; row headers stay in the document face at weight 400. The contribution bar (`.ledger__bar`) is a 3px stamp bar at 0.55 opacity drawn *under* the figure it belongs to — a second reading of the same value, never a replacement for the number.

### Error Banner
Sticky at `top: 0`, stamp ground, edge-stock text, `--step--1`, gutter-width padding, with a caution icon and a dismiss control whose opacity lifts from 0.8 to 1 on hover. The focus ring inverts to edge stock. The banner names the actual failure; it is not a generic toast.

### Boot State
A centred 2.6rem bordered `K` square with a label-voice note in `--ink-3`. No spinner.

## Do's and Don'ts

### Do:
- **Do** build structure from rules, margins and folios — `--rule-hair` between repeated rows, `--rule` inside a figure, `--rule-strong` around a figure, 2px `--ink` for an authority boundary.
- **Do** give every state a non-colour carrier: filled segments for severity, an inset 2px marker for selection, a baseline rule for the current tab, an inverted fill for an active verdict, a border style for signal class, a hatch for unavailable.
- **Do** keep the stamp red for the single primary action and the current selection only.
- **Do** set every measured quantity in JetBrains Mono with tabular figures, and nothing else in mono.
- **Do** mount photographic, moving, or terminal material on a plate with its own dark ground, caption bar and pixel scale bar.
- **Do** print the limit beside the figure it qualifies (`.measured__limit`), not in a footnote — the conspicuously-honest-about-limits voice is a binding brand commitment.
- **Do** cap running text at `--measure` (68ch) and set headings against explicit ch maxima.
- **Do** theme browser-owned surfaces from the palette: selection, caret, `accent-color`, scrollbars, focus ring, underline offset, and range inputs in both engines.
- **Do** compose emptiness — a dashed keyline and one sentence — rather than filling or hiding it.
- **Do** keep the accessibility floor intact: a genuinely reachable skip link, `aria-pressed`/`aria-label` on icon controls, a visible 2px focus ring at 3px offset, `.sr-only` text behind every `aria-hidden` graphic reading, a `prefers-reduced-motion` block, and a 320px minimum width.

### Don't:
- **Don't** use cards as page structure. A bounded figure is an exhibit or a record panel — keylined and singular; a grid of boxes is a different world.
- **Don't** put an eyebrow or kicker above a heading. Section numbering lives in the margin as a folio.
- **Don't** use a coloured left border thicker than 1px as a state device. The selected finding's `inset 2px 0 0` marker paired with a wash is the only left-edge mark in the system.
- **Don't** use monospace as a costume for "technical". Mono means measured.
- **Don't** let colour be the only carrier of any state.
- **Don't** add a global dark mode. The light ground is the use scene — a reviewer reading a printed finding at a desk — and the plate inversion already provides the contrast a dark theme would claim, while marking what is photographic and what is record.
- **Don't** introduce `border-radius` beyond `--plate-radius: 2px`.
- **Don't** add a third elevation, a coloured shadow, a glow, or a hard offset shadow; and don't use shadow to signal state.
- **Don't** ship a second stamped control in one view.
- **Don't** add icon fonts, emoji, glyph characters, or a borrowed icon package — extend `Icon.jsx` on the 24px / 1.6-stroke grid instead.
- **Don't** load fonts from a CDN or let a system display face stand in for Spectral; all three families are self-hosted subsets generated by `scripts/fetch_fonts.py`.
- **Don't** animate beyond the cover's one authored moment: `settle` (a 14px rise) and `rule-in` (a masthead clip-path wipe), running once on load at `--duration` 420ms on `--ease-out` cubic-bezier(0.16, 1, 0.3, 1), staggered 60/110/160ms with the plate at 640ms, wrapped in `prefers-reduced-motion: no-preference`, and starting from an already-visible default so nothing depends on the animation running. Everything else is a `--duration-fast` (180ms) state transition on ground, colour, border, or a 1–3px nudge.
