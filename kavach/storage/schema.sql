PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    fps REAL,
    width INTEGER,
    height INTEGER,
    total_frames INTEGER,
    duration REAL,
    analysis_metadata_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    timestamp REAL NOT NULL CHECK (timestamp >= 0),
    behaviour TEXT NOT NULL,
    entities_json TEXT NOT NULL,
    risk_score REAL NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
    risk_category TEXT NOT NULL CHECK (
        risk_category IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    detection_confidence REAL NOT NULL CHECK (
        detection_confidence >= 0 AND detection_confidence <= 1
    ),
    risk_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    review_status TEXT NOT NULL CHECK (
        review_status IN ('NEW', 'REVIEWED', 'FALSE_POSITIVE')
    ),
    lifecycle TEXT NOT NULL CHECK (lifecycle IN ('OPEN', 'CLOSED')),
    clip_path TEXT,
    first_seen_timestamp REAL NOT NULL CHECK (first_seen_timestamp >= 0),
    last_seen_timestamp REAL NOT NULL CHECK (last_seen_timestamp >= 0),
    occurrence_count INTEGER NOT NULL CHECK (occurrence_count > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS event_entities (
    event_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY (event_id, entity_id),
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_video_time
    ON events(video_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_events_behaviour
    ON events(behaviour);

CREATE INDEX IF NOT EXISTS idx_events_risk_category_score
    ON events(risk_category, risk_score);

CREATE INDEX IF NOT EXISTS idx_events_review_status
    ON events(review_status);

CREATE INDEX IF NOT EXISTS idx_event_entities_entity
    ON event_entities(entity_id);
