-- Base schemas and the ingest ledger.
--
-- Three layers, matching the dbt project that arrives on Day 3:
--   raw       exactly what AEMO published, every column text, append only
--   staging   typed, deduplicated, one row per interval per vintage
--   marts     the feature table and anything a model or the API reads
--
-- Every timestamp in this database is UTC. NEM local time is AEST (UTC+10, no
-- daylight saving) and is converted at ingest. See docs/CONVENTIONS.md.

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS marts;

-- One row per file pulled from NEMWeb.
--
-- Both timestamps are kept on purpose. generated_at is what AEMO stamped
-- inside the file; listed_at is when the file actually appeared on the site.
-- They differ, sometimes by more than ten minutes, and a forecast is not
-- usable until it is published. known_at is the later of the two and is the
-- only one the feature layer is allowed to join on.
CREATE TABLE IF NOT EXISTS raw.ingest_log (
    id              bigserial PRIMARY KEY,
    source_file     text        NOT NULL UNIQUE,
    source_url      text,
    report          text,
    interval_label  timestamptz,
    generated_at    timestamptz,
    listed_at       timestamptz,
    known_at        timestamptz,
    size_bytes      bigint,
    sha256          text,
    tables_loaded   jsonb       NOT NULL DEFAULT '{}'::jsonb,
    anomalies       jsonb       NOT NULL DEFAULT '[]'::jsonb,
    row_count_ok    boolean,
    ingested_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ingest_log_report_known_idx
    ON raw.ingest_log (report, known_at DESC);

CREATE INDEX IF NOT EXISTS ingest_log_report_interval_idx
    ON raw.ingest_log (report, interval_label DESC);

COMMENT ON COLUMN raw.ingest_log.generated_at IS
    'Timestamp AEMO wrote inside the file or filename. Earlier than publication.';
COMMENT ON COLUMN raw.ingest_log.listed_at IS
    'When the file appeared in the NEMWeb directory index.';
COMMENT ON COLUMN raw.ingest_log.known_at IS
    'max(generated_at, listed_at). The conservative point-in-time vintage.';
