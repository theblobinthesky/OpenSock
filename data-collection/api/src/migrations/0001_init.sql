CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL,
  finalized_at TIMESTAMPTZ NULL,
  name TEXT,
  handle TEXT,
  email TEXT NOT NULL,
  notify_opt_in BOOLEAN NOT NULL DEFAULT FALSE,
  mode TEXT,
  image_count INT NOT NULL DEFAULT 0,
  focal_length REAL
);

CREATE TABLE IF NOT EXISTS images (
  id BIGSERIAL PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  server_filename TEXT NOT NULL,
  original_mime TEXT,
  size_bytes BIGINT,
  sha256 TEXT NOT NULL,
  hash_source TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dedup_index (
  sha256 TEXT PRIMARY KEY,
  image_id BIGINT NOT NULL REFERENCES images(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS uploads (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  filename TEXT NOT NULL,
  mime TEXT,
  total_size BIGINT NOT NULL,
  received_size BIGINT NOT NULL DEFAULT 0,
  tmp_path TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

