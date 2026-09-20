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

function toneFor(className) {
  return CLASS_TONE[className] || "#b9b3a2";
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
  const [size, setSize] = useState({ width: 960, height: 540 });

  const duration = tracks?.duration || 0;
  const frames = tracks?.frames || [];
  const sourceWidth = tracks?.width || 1920;
  const sourceHeight = tracks?.height || 1080;

  const highlighted = useMemo(
    () => new Set(selectedEvent?.entities || []),
    [selectedEvent]
  );

  /* Keep the canvas backing store matched to its CSS box and the device pixel
   * ratio, or the strokes go soft on a high-density display. */
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return undefined;
    const observer = new ResizeObserver(([entry]) => {
      const width = Math.max(240, Math.round(entry.contentRect.width));
      setSize({ width, height: Math.round((width * sourceHeight) / sourceWidth) });
    });
    observer.observe(wrap);
    return () => observer.disconnect();
  }, [sourceWidth, sourceHeight]);

  const draw = useCallback(
    (at) => {
      const canvas = canvasRef.current;
      if (!canvas || !frames.length) return;
      const context = canvas.getContext("2d");
      const ratio = window.devicePixelRatio || 1;
      const { width, height } = size;

      if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
        canvas.width = width * ratio;
        canvas.height = height * ratio;
      }
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);

      const scale = width / sourceWidth;
      const sx = (value) => value * scale;

      context.fillStyle = "#0e0e09";
      context.fillRect(0, 0, width, height);

      /* A faint measuring grid: this is a plate from a measured record, and
       * the grid says the coordinates are image-space pixels. */
      context.strokeStyle = "rgba(232, 228, 214, 0.05)";
      context.lineWidth = 1;
      for (let x = 0; x <= sourceWidth; x += 240) {
        context.beginPath();
        context.moveTo(sx(x), 0);
        context.lineTo(sx(x), height);
        context.stroke();
      }
      for (let y = 0; y <= sourceHeight; y += 240) {
        context.beginPath();
        context.moveTo(0, sx(y));
        context.lineTo(width, sx(y));
        context.stroke();
      }

      /* Zones, drawn as surveyed boundaries with their names set small. */
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
          if (index === 0) context.moveTo(sx(x), sx(y));
          else context.lineTo(sx(x), sx(y));
        });
        context.closePath();
        context.stroke();

        if (!compact) {
          context.setLineDash([]);
          context.font = `600 ${Math.max(9, 10 * (width / 960))}px Archivo, sans-serif`;
          context.fillStyle = restricted
            ? "rgba(214, 126, 112, 0.95)"
            : "rgba(164, 158, 140, 0.9)";
          context.fillText(
            (ZONE_LABELS[zone.name] || zone.name).toUpperCase(),
            sx(points[0][0]) + 6,
            sx(points[0][1]) + 14
          );
          context.setLineDash([6, 5]);
        }
      });
      context.setLineDash([]);

      const frame = frameAt(frames, at);
      if (!frame) return;

      frame.o.forEach((object) => {
        const [x1, y1, x2, y2] = object.b;
        const entityId = `${object.c}_${object.id}`;
        const isHighlighted = highlighted.has(entityId);
        const tone = toneFor(object.c);

        context.lineWidth = isHighlighted ? 2.4 : 1.3;
        context.strokeStyle = isHighlighted ? "#e8635a" : tone;
        context.strokeRect(sx(x1), sx(y1), sx(x2 - x1), sx(y2 - y1));

        if (isHighlighted) {
          /* Corner ticks mark the entity the selected finding names, so the
             link between the record and the picture is explicit. */
          const tick = Math.max(6, sx(28));
          context.lineWidth = 2.4;
          [
            [sx(x1), sx(y1), 1, 1],
            [sx(x2), sx(y1), -1, 1],
            [sx(x1), sx(y2), 1, -1],
            [sx(x2), sx(y2), -1, -1],
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
          context.font = `500 ${Math.max(9, 10.5 * (width / 960))}px "JetBrains Mono", monospace`;
          const metrics = context.measureText(label);
          const labelHeight = Math.max(13, sx(26));
          context.fillStyle = isHighlighted ? "#e8635a" : tone;
          context.fillRect(
            sx(x1),
            sx(y1) - labelHeight,
            metrics.width + 10,
            labelHeight
          );
          context.fillStyle = "#0e0e09";
          context.fillText(label, sx(x1) + 5, sx(y1) - labelHeight / 2 + 3.5);
        }
      });
    },
    [frames, size, sourceWidth, sourceHeight, tracks, highlighted, compact]
  );

  useEffect(() => {
    draw(time);
  }, [draw, time]);

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

  const scaleBarPx = 400;
  const scaleBarWidth = (scaleBarPx / sourceWidth) * 100;

  return (
    <figure className={`plate ${compact ? "plate--compact" : ""}`.trim()}>
      <div className="plate__frame" ref={wrapRef}>
        <canvas
          ref={canvasRef}
          className="plate__canvas"
          style={{ width: "100%", height: `${size.height}px` }}
          role="img"
          aria-label={`Reconstruction at ${formatTime(time)} of ${formatTime(duration)}. ${
            frameAt(frames, time)?.o.length || 0
          } tracked objects.`}
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
          Synthetic reconstruction, {sourceWidth}&#215;{sourceHeight} at {tracks?.fps || 25} fps.
          Tracked geometry replayed from the controlled dataset, not camera footage.
        </span>
      </figcaption>
    </figure>
  );
}
