import { useCallback, useEffect, useState } from "react";
import Icon from "./lib/Icon.jsx";
import Landing from "./site/Landing.jsx";
import Workspace from "./workspace/Workspace.jsx";
import { resolveSource } from "./lib/source.js";

/* Two surfaces, one bundle.
 *
 * The record (a Persuade surface) and the workspace (an Operate surface) live
 * at `#/` and `#/workspace`. A hash route keeps the workspace linkable and
 * survives the static host's catch-all rewrite without needing a router
 * dependency for two routes.
 */

function routeFromHash() {
  return window.location.hash.replace(/^#\/?/, "") === "workspace" ? "workspace" : "record";
}

export default function App() {
  const [route, setRoute] = useState(routeFromHash);
  const [source, setSource] = useState(null);
  const [video, setVideo] = useState(null);
  const [tracks, setTracks] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const onHashChange = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const resolved = await resolveSource();
        if (cancelled) return;
        setSource(resolved);

        const videos = await resolved.listVideos();
        if (cancelled) return;

        if (!videos.length) {
          setLoading(false);
          return;
        }

        const [loadedVideo, loadedTracks] = await Promise.all([
          resolved.getVideo(videos[0].video_id),
          resolved.tracks(),
        ]);
        if (cancelled) return;

        setVideo(loadedVideo);
        setTracks(loadedTracks);
      } catch (reason) {
        if (!cancelled) setError(reason.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const go = useCallback((next) => {
    window.location.hash = next === "workspace" ? "#/workspace" : "#/";
    setRoute(next);
    window.scrollTo({ top: 0, behavior: "instant" });
  }, []);

  if (loading) {
    return (
      <div className="boot">
        <p className="boot__mark" aria-hidden="true">
          K
        </p>
        <p className="boot__note">Opening the record…</p>
      </div>
    );
  }

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to main content
      </a>

      {error ? (
        <div className="banner" role="alert">
          <Icon name="caution" size={16} />
          <span>{error}</span>
          <button type="button" onClick={() => setError("")} aria-label="Dismiss this message">
            <Icon name="close" size={15} />
          </button>
        </div>
      ) : null}

      {route === "workspace" && source && video ? (
        <Workspace
          source={source}
          video={video}
          tracks={tracks}
          onExit={() => go("record")}
          onError={setError}
        />
      ) : (
        <Landing
          video={video}
          tracks={tracks}
          sourceMode={source?.mode}
          onOpenWorkspace={() => go("workspace")}
        />
      )}
    </>
  );
}
