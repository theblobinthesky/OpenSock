OpenSock Data Collection API (Go)

Overview
- Minimal HTTP API to collect sock image sessions.
- Endpoints: create session, upload images, finalize session, and admin deletion.
- Stores metadata in JSON and images on disk for now; Postgres integration TBD.

Run (development)
- Go 1.21+
- Env:
  - `PORT` (default: 8080)
  - `STORAGE_DIR` (default: ./storage)
  - `SESSION_IMAGE_CAP` (default: 10)
  - `ADMIN_PASSWORD` (required for admin routes)
  - `DATABASE_URL` (required; Postgres connection string)

Start
```
cd data-collection/api
go run ./...
```

API
- POST `/sessions` → { name?, handle?, email?, notify_opt_in?, language? } → { session_id }
- POST `/sessions/{id}/images` (multipart/form-data; field `image`) → { uploaded, rejected_duplicate, remaining }
- POST `/sessions/{id}/finalize` → { ok: true }
- Admin (HTTP Basic Auth, password = `ADMIN_PASSWORD`):
  - GET `/admin/sessions?email=...`
  - DELETE `/admin/sessions/{id}`
  - DELETE `/admin/users/{email}`

Notes
- EXIF stripping: images decodable by Go get re-encoded (dropping metadata). Unknown formats are saved as-is [todo: extend coverage].
- Dedup: global exact-duplicate detection using SHA-256 on normalized bytes.
- Mode: include `mode` in session payload as `MIXED_UNIQUES` (default) or `SAME_TYPE`. Finalize enforces min images (4 vs 3).

Local Postgres (dev only)
- `docker compose up -d db` inside `data-collection/api` starts a Postgres 16 instance used by the API.

Connect‑Web (gRPC‑web)
- Proto location: `proto/opensock/dc/v1/dc.proto`
- Buf configs: `proto/buf.yaml`, `proto/buf.gen.yaml`
- Install buf:
  - macOS (Homebrew): `brew install bufbuild/buf/buf`
  - Linux (dl): download a release from https://github.com/bufbuild/buf/releases and place `buf` in your PATH.
  - Linux (Debian/Ubuntu): download the `buf-*-linux-amd64` tarball, extract, and move the binary to `/usr/local/bin`.
- Generate stubs (requires buf + protoc):
  - `cd data-collection/api/proto`
  - `buf generate`
- Server registers Connect handlers alongside REST in `main.go` (Create/Finalize over Connect; uploads stay REST multipart).



Production migrations
- In dev, the API auto-applies embedded SQL migrations.
- In production, apply migrations manually (example):
  - `psql "$DATABASE_URL" -f migrations/0001_init.sql`
  - Repeat for any future migration files in `migrations/`.
