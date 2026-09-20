/* Where the workspace gets its evidence.
 *
 * The same build serves two situations. Run locally alongside the FastAPI
 * service and it talks to the real pipeline: upload video, analyse, generate
 * replay clips, ask the grounded assistant. Served as a static bundle with no
 * backend -- which is what the published site is -- it reads the committed
 * demo fixtures instead.
 *
 * Both paths return the identical shape, because the fixtures are generated
 * through `kavach.presentation`, the same module that shapes live API
 * responses. The workspace never branches on which one it got; it only asks
 * the source whether a capability is available.
 */

const API_BASE = import.meta.env.VITE_API_BASE || "";

export const DEMO_VIDEO_ID = "demo-warehouse-reconstruction";

const PROBE_TIMEOUT_MS = 2500;

class UnavailableError extends Error {
  constructor(message) {
    super(message);
    this.name = "UnavailableError";
    this.unavailable = true;
  }
}

async function request(path, options = {}) {
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

/* A static host answers every unknown path with index.html and a 200, so a
 * bare `response.ok` would report a backend that is not there. The probe only
 * accepts a JSON body that actually identifies the KAVACH service. */
export async function probeBackend() {
  try {
    const response = await fetch(`${API_BASE}/api/health`, {
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
      headers: { accept: "application/json" },
    });
    if (!response.ok) return false;
    if (!(response.headers.get("content-type") || "").includes("application/json")) {
      return false;
    }
    const body = await response.json();
    return body?.status === "ok" && body?.service === "kavach-api";
  } catch {
    return false;
  }
}

async function fixture(name) {
  const response = await fetch(`${import.meta.env.BASE_URL}demo/${name}`);
  if (!response.ok) {
    throw new Error(`The demo dataset could not be loaded (${name}).`);
  }
  return response.json();
}

/* --- Live source: the local FastAPI service ------------------------------ */

const liveSource = {
  mode: "live",
  can: () => true,

  listVideos: () => request("/api/videos").then((result) => result.videos || []),
  getVideo: (id) => request(`/api/videos/${id}`),
  models: () => request("/api/models").then((result) => result.models || []),
  tracks: async () => null,

  upload: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/api/videos/upload", { method: "POST", body: form });
  },

  ingestUrl: (url) =>
    request("/api/videos/url", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ url }),
    }),

  analyse: (id, { confidence = 0.25, model = "bundled" } = {}) =>
    request(`/api/videos/${id}/analyse`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ confidence, model }),
    }),

  jobStatus: (id) => request(`/api/videos/${id}/job`),

  replay: (eventId) => request(`/api/events/${eventId}/replay`, { method: "POST" }),

  review: (eventId, status) =>
    request(`/api/events/${eventId}/review`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ status }),
    }),

  ask: (videoId, question) =>
    request("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ video_id: videoId, question, use_llm: true }),
    }),

  clipUrl: (path) => `${API_BASE}${path}`,
};

/* --- Demo source: the committed reconstruction -------------------------- */

/* Capabilities the demo genuinely cannot provide. Each carries the reason the
 * interface shows the visitor, so a disabled control explains itself instead
 * of failing silently. */
const DEMO_LIMITS = {
  upload: "Analysing a video needs the local backend: YOLO inference, OpenCV decoding and SQLite writes all run on your machine.",
  ingestUrl: "Fetching a video URL needs the local backend to decode and analyse it.",
  analyse: "Running the pipeline needs the local backend.",
  replay: "Cutting an evidence clip needs the local backend and the source video.",
  ask: "The grounded assistant needs a local Ollama service. Retrieval is deterministic, but the explanation is generated on your machine.",
};

/* Review status is deliberately absent from the list above: the demo does
 * support it, in memory only, so a visitor can work a queue. The interface
 * says so at the control rather than implying anything was saved. */

function createDemoSource() {
  let cachedVideo = null;
  let cachedTracks = null;
  /* Review status is the one piece of state the demo keeps. It is held in
   * memory only, so a visitor can work through the queue without the page
   * pretending anything was persisted. */
  const reviewOverrides = new Map();

  const applyOverrides = (video) => ({
    ...video,
    events: video.events.map((event) =>
      reviewOverrides.has(event.event_id)
        ? { ...event, review_status: reviewOverrides.get(event.event_id) }
        : event
    ),
  });

  const load = async () => {
    if (!cachedVideo) cachedVideo = await fixture("video.json");
    return applyOverrides(cachedVideo);
  };

  const refuse = (capability) => () =>
    Promise.reject(new UnavailableError(DEMO_LIMITS[capability]));

  return {
    mode: "demo",
    can: (capability) => !(capability in DEMO_LIMITS),
    reason: (capability) => DEMO_LIMITS[capability] || null,

    listVideos: async () => {
      const video = await load();
      return [
        {
          video_id: video.video_id,
          name: video.name,
          fps: video.fps,
          width: video.width,
          height: video.height,
          total_frames: video.total_frames,
          duration: video.duration,
          processed: false,
          processed_url: null,
        },
      ];
    },

    getVideo: load,
    models: async () => [],

    tracks: async () => {
      if (!cachedTracks) cachedTracks = await fixture("tracks.json");
      return cachedTracks;
    },

    meta: () => fixture("meta.json"),

    /* Review is the exception: it is allowed, but only in memory. */
    review: async (eventId, status) => {
      reviewOverrides.set(eventId, status);
      const video = await load();
      return video.events.find((event) => event.event_id === eventId);
    },

    upload: refuse("upload"),
    ingestUrl: refuse("ingestUrl"),
    analyse: refuse("analyse"),
    jobStatus: async () => null,
    replay: refuse("replay"),
    ask: refuse("ask"),

    clipUrl: (path) => path,
  };
}

export async function resolveSource() {
  const live = await probeBackend();
  return live ? liveSource : createDemoSource();
}

export { UnavailableError };
