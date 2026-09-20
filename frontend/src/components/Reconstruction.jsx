import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Icon from "../lib/Icon.jsx";
import { formatTime, prettyLabel, riskClass } from "../lib/format.js";

/* The exhibit plate.
 *
 * This replays the same tracked geometry the behaviour rules consumed, so the
 * picture and the evidence cannot disagree. It is deliberately not dressed up
 * as camera footage: it is a measured reconstruction on a dark plate ground,
 * with a printed caption bar and a pixel scale bar, the way a report tips in a
 * figure rather than a photograph.
 */

const ZONE_LABELS = {
  RESTRICTED_ZONE: "Restricted",
  STAGING_ZONE: "Staging",
  LOADING_ZONE: "Loading",
  PALLET_ZONE: "Pallet",
};

const CLASS_TONE = {
  person: "#e8c15a",
  worker: "#e8c15a",
  forklift: "#7fb2d9",
  pallet_truck: "#7fb2d9",
  truck: "#7fb2d9",
  carton: "#d9a98a",
  box: "#d9a98a",
  package: "#d9a98a",
  pallet: "#9fbf8f",
};

/* The source frame is 1920x1080, but the tracked geometry and the zone
 * polygons only ever occupy part of it. Drawing the full frame would spend the
 * top third of the plate on empty ground. The view is the bounding box of
 * everything that is ever drawn, computed once so the framing stays put while
 * scrubbing rather than lurching per frame. */
const VIEW_PAD = 28;

function toneFor(className) {
  return CLASS_TONE[className] || "#b9b3a2";
}

function computeView(tracks) {
  if (!tracks) return { x: 0, y: 0, w: 1920, h: 1080 };
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  const include = (x1, y1, x2, y2) => {
    if (x1 < minX) minX = x1;
    if (y1 < minY) minY = y1;
    if (x2 > maxX) maxX = x2;
    if (y2 > maxY) maxY = y2;
  };

  (tracks.zones || []).forEach((zone) => {
    (zone.polygon || []).forEach(([x, y]) => include(x, y, x, y));
  });
  (tracks.frames || []).forEach((frame) => {
    frame.o.forEach((object) => {
      const [x1, y1, x2, y2] = object.b;
      include(x1, y1, x2, y2);
    });
  });

  if (!Number.isFinite(minX)) {
    return { x: 0, y: 0, w: tracks.width || 1920, h: tracks.height || 1080 };
  }

  const x = Math.max(0, minX - VIEW_PAD);
  const y = Math.max(0, minY - VIEW_PAD);
  const right = Math.min(tracks.width || 1920, maxX + VIEW_PAD);
  const bottom = Math.min(tracks.height || 1080, maxY + VIEW_PAD);
  return { x, y, w: right - x, h: bottom - y };
}

/* Nearest sampled frame at or before `time`, so scrubbing lands on real
 * sampled geometry rather than interpolating something the engines never saw. */
function frameAt(frames, time) {
  if (!frames.length) return null;
  let low = 0;
  let high = frames.length - 1;
  while (low < high) {
    const mid = (low + high + 1) >> 1;
    if (frames[mid].t <= time) low = mid;
    else high = mid - 1;
  }
  return frames[low];
}

export default function Reconstruction({
  tracks,
  events = [],
  selectedEvent = null,
  onSelectEvent,
  compact = false,
}) {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const rafRef = useRef(0);
  const clockRef = useRef(0);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [width, setWidth] = useState(960);
  /* Canvas text does not re-layout when a webfont arrives, so a first paint
   * before the faces load would bake the platform sans and mono into the
   * plate. Paint once the faces are ready, and repaint when they land. */
  const [fontsReady, setFontsReady] = useState(false);

  const frames = tracks?.frames || [];
  const duration = tracks?.duration || 0;
  const view = useMemo(() => computeView(tracks), [tracks]);
  const height = Math.round((width * view.h) / view.w);

  const highlighted = useMemo(
    () => new Set(selectedEvent?.entities || []),
    [selectedEvent]
  );

  useEffect(() => {
    let cancelled = false;
    const ready = document.fonts?.ready;
    if (!ready) {
      setFontsReady(true);
      return undefined;
    }
    ready.then(() => {
      if (!cancelled) setFontsReady(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  /* Keep the canvas backing store matched to its CSS box and the device pixel
   * ratio, or the strokes go soft on a high-density display. */
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return undefined;
    const observer = new ResizeObserver(([entry]) => {
      setWidth(Math.max(240, Math.round(entry.contentRect.width)));
    });
    observer.observe(wrap);
    return () => observer.disconnect();
  }, []);

  const draw = useCallback(
    (at) => {
      const canvas = canvasRef.current;
      if (!canvas || !frames.length) return;
      const context = canvas.getContext("2d");
      const ratio = window.devicePixelRatio || 1;

      if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
        canvas.width = width * ratio;
        canvas.height = height * ratio;
      }
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);

      const scale = width / view.w;
      const sx = (value) => (value - view.x) * scale;
      const sy = (value) => (value - view.y) * scale;
      /* Label type never drops below 11px: smaller than that the zone names
       * and track tags stop being readable at phone width. */
      const typeSize = Math.max(11, 11.5 * (width / 960));

      context.fillStyle = "#0e0e09";
      context.fillRect(0, 0, width, height);

      /* A faint measuring grid: this is a plate from a measured record, and
       * the grid says the coordinates are image-space pixels. */
      context.strokeStyle = "rgba(232, 228, 214, 0.05)";
      context.lineWidth = 1;
      const gridStep = 240;
      for (let x = Math.ceil(view.x / gridStep) * gridStep; x <= view.x + view.w; x += gridStep) {
        context.beginPath();
        context.moveTo(sx(x), 0);
        context.lineTo(sx(x), height);
        context.stroke();
      }
      for (let y = Math.ceil(view.y / gridStep) * gridStep; y <= view.y + view.h; y += gridStep) {
        context.beginPath();
        context.moveTo(0, sy(y));
        context.lineTo(width, sy(y));
        context.stroke();
      }

      /* Zones, drawn as surveyed boundaries. Their names sit along the bottom
       * edge of each polygon, clear of the track tags that sit above boxes. */
      const zoneLabels = [];
      context.setLineDash([6, 5]);
      context.lineWidth = 1.2;
      (tracks?.zones || []).forEach((zone) => {
        const points = zone.polygon || [];
        if (points.length < 3) return;
        const restricted = zone.name === "RESTRICTED_ZONE";
        context.strokeStyle = restricted
          ? "rgba(199, 96, 82, 0.75)"
          : "rgba(232, 228, 214, 0.26)";
        context.beginPath();
        points.forEach(([x, y], index) => {
          if (index === 0) context.moveTo(sx(x), sy(y));
          else context.lineTo(sx(x), sy(y));
        });
        context.closePath();
        context.stroke();

        if (!compact) {
          const xs = points.map(([x]) => x);
          const ys = points.map(([, y]) => y);
          zoneLabels.push({
            text: (ZONE_LABELS[zone.name] || zone.name).toUpperCase(),
            x: sx(Math.min(...xs)) + 7,
            y: sy(Math.max(...ys)) - 7,
            restricted,
          });
        }
      });
      context.setLineDash([]);

      zoneLabels.forEach((label) => {
        context.font = `600 ${typeSize}px Archivo, sans-serif`;
        context.fillStyle = label.restricted
          ? "rgba(224, 140, 126, 0.95)"
          : "rgba(180, 174, 156, 0.95)";
        context.fillText(label.text, label.x, label.y);
      });

      const frame = frameAt(frames, at);
      if (!frame) return;

      /* Two entities standing close together would otherwise print their tags
       * on top of one another, which is exactly the moment the plate is being
       * read — a proximity finding. Placed tags are remembered and a colliding
       * one steps up until it is clear. */
      const placedTags = [];
      const collides = (a, b) =>
        a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;

      frame.o.forEach((object) => {
        const [x1, y1, x2, y2] = object.b;
        const entityId = `${object.c}_${object.id}`;
        const isHighlighted = highlighted.has(entityId);
        const tone = toneFor(object.c);

        context.lineWidth = isHighlighted ? 2.4 : 1.3;
        context.strokeStyle = isHighlighted ? "#e8635a" : tone;
        context.strokeRect(sx(x1), sy(y1), sx(x2) - sx(x1), sy(y2) - sy(y1));

        if (isHighlighted) {
          /* Corner ticks mark the entity the selected finding names, so the
             link between the record and the picture is explicit. */
          const tick = Math.max(6, 28 * scale);
          context.lineWidth = 2.4;
          [
            [sx(x1), sy(y1), 1, 1],
            [sx(x2), sy(y1), -1, 1],
            [sx(x1), sy(y2), 1, -1],
            [sx(x2), sy(y2), -1, -1],
          ].forEach(([cx, cy, dx, dy]) => {
            context.beginPath();
            context.moveTo(cx + tick * dx, cy);
            context.lineTo(cx, cy);
            context.lineTo(cx, cy + tick * dy);
            context.stroke();
          });
        }

        if (!compact) {
          const label = `${object.c} #${object.id}`;
          context.font = `500 ${typeSize}px "JetBrains Mono", monospace`;
          const textWidth = context.measureText(label).width;
          const labelHeight = typeSize + 7;
          /* The tag sits above the box, and drops inside it when there is no
           * room above — which is also what keeps it clear of a zone name
           * printed along a boundary near the top of the view. */
          const above = sy(y1) - labelHeight >= 2;
          let tagY = above ? sy(y1) - labelHeight : sy(y1);
          const rect = { x: sx(x1), y: tagY, w: textWidth + 10, h: labelHeight };
          let guard = 0;
          while (placedTags.some((placed) => collides(rect, placed)) && guard < 6) {
            rect.y -= labelHeight + 2;
            guard += 1;
          }
          if (rect.y < 0) rect.y = tagY + labelHeight + 2;
          tagY = rect.y;
          placedTags.push(rect);

          context.fillStyle = isHighlighted ? "#e8635a" : tone;
          context.fillRect(rect.x, tagY, rect.w, labelHeight);
          context.fillStyle = "#0e0e09";
          context.fillText(label, rect.x + 5, tagY + labelHeight - 6);
        }
      });
    },
    [frames, width, height, view, tracks, highlighted, compact]
  );

  useEffect(() => {
    draw(time);
  }, [draw, time, fontsReady]);

  useEffect(() => {
    if (!playing) return undefined;
    clockRef.current = performance.now();
    const step = (now) => {
      const delta = (now - clockRef.current) / 1000;
      clockRef.current = now;
      setTime((current) => {
        const next = current + delta;
        if (next >= duration) {
          setPlaying(false);
          return duration;
        }
        return next;
      });
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, duration]);

  /* Jumping to a finding is the point of the plate: the record and the
   * picture stay in step. */
  useEffect(() => {
    if (selectedEvent?.timestamp === undefined) return;
    setPlaying(false);
    setTime(Math.max(0, Math.min(duration, Number(selectedEvent.timestamp) || 0)));
  }, [selectedEvent, duration]);

  const markers = useMemo(
    () =>
      events
        .filter((event) => event.signal_kind === "safety")
        .map((event) => ({
          id: event.event_id,
          at: Number(event.timestamp) || 0,
          category: event.risk?.category || "LOW",
          label: prettyLabel(event.behaviour),
          event,
        })),
    [events]
  );

  const liveCount = frameAt(frames, time)?.o.length || 0;
  const scaleBarPx = 400;
  const scaleBarWidth = (scaleBarPx / view.w) * 100;

  return (
    <figure className={`plate ${compact ? "plate--compact" : ""}`.trim()}>
      <div className="plate__frame" ref={wrapRef}>
        <canvas
          ref={canvasRef}
          className="plate__canvas"
          style={{ width: "100%", height: `${height}px` }}
          role="img"
          aria-label={`Reconstruction at ${formatTime(time)} of ${formatTime(
            duration
          )}. ${liveCount} tracked ${liveCount === 1 ? "object" : "objects"}.`}
        />

        <div className="plate__scale" aria-hidden="true">
          <span className="plate__scale-bar" style={{ width: `${scaleBarWidth}%` }} />
          <span className="measure">{scaleBarPx} px</span>
        </div>
      </div>

      <div className="plate__transport">
        <button
          type="button"
          className="plate__play"
          onClick={() => {
            if (time >= duration) setTime(0);
            setPlaying((value) => !value);
          }}
          aria-label={playing ? "Pause the reconstruction" : "Play the reconstruction"}
        >
          <Icon name={playing ? "pause" : "play"} size={15} />
        </button>

        <div className="plate__track">
          <input
            type="range"
            min={0}
            max={duration || 1}
            step={tracks?.step_seconds || 0.2}
            value={time}
            onChange={(sourceEvent) => {
              setPlaying(false);
              setTime(Number(sourceEvent.target.value));
            }}
            aria-label="Scrub the reconstruction"
          />
          <div className="plate__markers">
            {markers.map((marker) => (
              <button
                key={marker.id}
                type="button"
                className={`plate__marker ${riskClass(marker.category)}`}
                style={{ left: `${duration ? (marker.at / duration) * 100 : 0}%` }}
                title={`${marker.label} at ${formatTime(marker.at)}`}
                onClick={() => onSelectEvent?.(marker.event)}
              >
                <span className="sr-only">
                  {marker.label} at {formatTime(marker.at)}
                </span>
              </button>
            ))}
          </div>
        </div>

        <span className="plate__clock measure">
          {formatTime(time)} / {formatTime(duration)}
        </span>
      </div>

      <figcaption className="plate__caption">
        <span className="plate__caption-mark measure">Plate 1</span>
        <span>
          Synthetic reconstruction, cropped to the tracked region of a{" "}
          {tracks?.width || 1920}&#215;{tracks?.height || 1080} source at{" "}
          {tracks?.fps || 25} fps. Tracked geometry replayed from the controlled
          dataset, not camera footage.
        </span>
      </figcaption>
    </figure>
  );
}
