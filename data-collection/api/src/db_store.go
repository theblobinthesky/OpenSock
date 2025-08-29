package main

import (
    "crypto/sha256"
    "database/sql"
    "encoding/hex"
    "errors"
    "fmt"
    "io"
    "net/http"
    "os"
    "path/filepath"
    "strings"
    "time"

    _ "github.com/lib/pq"
)

type DBStore struct {
    db   *sql.DB
    dir  string
    cap  int
}

func NewDBStore(dsn, dir string, cap int) (*DBStore, error) {
    if err := os.MkdirAll(dir, 0o755); err != nil { return nil, err }
    db, err := sql.Open("postgres", dsn)
    if err != nil { return nil, err }
    if err := db.Ping(); err != nil { return nil, err }
    if err := runMigrations(db); err != nil { return nil, err }
    s := &DBStore{db: db, dir: dir, cap: cap}
    return s, nil
}

func (s *DBStore) ensureTables() error {
    stmts := []string{
        `CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL,
            finalized_at TIMESTAMPTZ NULL,
            name TEXT, handle TEXT, email TEXT NOT NULL,
            notify_opt_in BOOLEAN NOT NULL DEFAULT FALSE,
            mode TEXT, image_count INT NOT NULL DEFAULT 0,
            focal_length REAL
        )`,
        `CREATE TABLE IF NOT EXISTS images (
            id BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            server_filename TEXT NOT NULL,
            original_mime TEXT,
            size_bytes BIGINT,
            sha256 TEXT NOT NULL,
            hash_source TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )`,
        `CREATE TABLE IF NOT EXISTS dedup_index (
            sha256 TEXT PRIMARY KEY,
            image_id BIGINT NOT NULL REFERENCES images(id) ON DELETE CASCADE
        )`,
    }
    for _, q := range stmts {
        if _, err := s.db.Exec(q); err != nil { return err }
    }
    _, _ = s.db.Exec(`ALTER TABLE sessions ALTER COLUMN email SET NOT NULL`)
    return nil
}

func (s *DBStore) sessionDir(id string) string { return filepath.Join(s.dir, id) }

func (s *DBStore) CreateSession(req CreateSessionRequest) (*Session, error) {
    id := newID()
    sess := &Session{
        ID: id,
        CreatedAt: time.Now().UTC(),
        Name: strings.TrimSpace(req.Name),
        Handle: strings.TrimSpace(req.Handle),
        Email: strings.TrimSpace(req.Email),
        NotifyOptIn: req.NotifyOptIn,
        Language: strings.TrimSpace(req.Language),
        Mode: strings.TrimSpace(req.Mode),
    }
    if err := os.MkdirAll(s.sessionDir(id), 0o755); err != nil { return nil, err }
    _, err := s.db.Exec(`INSERT INTO sessions(id, created_at, name, handle, email, notify_opt_in, mode, image_count) VALUES($1,$2,$3,$4,$5,$6,$7,$8)`,
        sess.ID, sess.CreatedAt, sess.Name, sess.Handle, sess.Email, sess.NotifyOptIn, sess.Mode, 0)
    if err != nil { return nil, err }
    return sess, nil
}

func (s *DBStore) UploadImages(r *http.Request, sessionID string) (uploaded int, duplicates int, remaining int, err error) {
    if err = r.ParseMultipartForm(32 << 20); err != nil { return 0, 0, 0, err }
    if err := s.ensureTables(); err != nil { return 0, 0, 0, err }
    // Count images already
    var count int
    if err = s.db.QueryRow(`SELECT image_count FROM sessions WHERE id=$1`, sessionID).Scan(&count); err != nil { return 0, 0, 0, fmt.Errorf("session not found") }
    files := r.MultipartForm.File["image"]
    for _, fh := range files {
        if count >= s.cap { break }
        f, e := fh.Open(); if e != nil { continue }
        data, e := io.ReadAll(f); f.Close(); if e != nil { continue }
        outBytes, outExt := reencodeToStripMetadata(data)
        if outBytes == nil { outBytes = data; outExt = strings.ToLower(filepath.Ext(fh.Filename)); if outExt == "" { outExt = ".bin" } }
        h := sha256.Sum256(outBytes)
        hash := hex.EncodeToString(h[:])
        var exists int
        if e = s.db.QueryRow(`SELECT 1 FROM dedup_index WHERE sha256=$1`, hash).Scan(&exists); e == nil {
            duplicates++
            continue
        }
        serverName := fmt.Sprintf("%d_%s%s", time.Now().UnixNano(), hash[:8], outExt)
        outPath := filepath.Join(s.sessionDir(sessionID), serverName)
        if e = os.WriteFile(outPath, outBytes, 0o644); e != nil { continue }
        tx, e := s.db.Begin(); if e != nil { continue }
        var imgID int64
        if e = tx.QueryRow(`INSERT INTO images(session_id, server_filename, original_mime, size_bytes, sha256, hash_source) VALUES($1,$2,$3,$4,$5,$6) RETURNING id`, sessionID, serverName, fh.Header.Get("Content-Type"), len(outBytes), hash, "stripped").Scan(&imgID); e != nil { tx.Rollback(); continue }
        if _, e = tx.Exec(`INSERT INTO dedup_index(sha256, image_id) VALUES($1,$2)`, hash, imgID); e != nil { tx.Rollback(); continue }
        if _, e = tx.Exec(`UPDATE sessions SET image_count=image_count+1 WHERE id=$1`, sessionID); e != nil { tx.Rollback(); continue }
        if e = tx.Commit(); e != nil { continue }
        uploaded++
        count++
    }
    remaining = s.cap - count
    if remaining < 0 { remaining = 0 }
    return
}

func (s *DBStore) FinalizeSession(id string) error {
    if err := s.ensureTables(); err != nil { return err }
    // Finalize without enforcing minimum image count (policy updated)
    // Ensure session exists
    var exists int
    if err := s.db.QueryRow(`SELECT 1 FROM sessions WHERE id=$1`, id).Scan(&exists); err != nil {
        return fmt.Errorf("not found")
    }
    _, err := s.db.Exec(`UPDATE sessions SET finalized_at=now() WHERE id=$1`, id)
    return err
}

func (s *DBStore) SessionsByEmail(email string) []Session {
    rows, err := s.db.Query(`SELECT id, created_at, finalized_at, name, handle, email, notify_opt_in, mode, image_count FROM sessions WHERE email=$1`, email)
    if err != nil { return nil }
    defer rows.Close()
    out := []Session{}
    for rows.Next() {
        var sess Session
        var fin sql.NullTime
        if err := rows.Scan(&sess.ID, &sess.CreatedAt, &fin, &sess.Name, &sess.Handle, &sess.Email, &sess.NotifyOptIn, &sess.Mode, &sess.ImageCount); err == nil {
            if fin.Valid { t := fin.Time; sess.FinalizedAt = &t }
            out = append(out, sess)
        }
    }
    return out
}

func (s *DBStore) DeleteSession(id string) error {
    tx, err := s.db.Begin(); if err != nil { return err }
    if _, err = tx.Exec(`DELETE FROM images WHERE session_id=$1`, id); err != nil { tx.Rollback(); return err }
    if _, err = tx.Exec(`DELETE FROM sessions WHERE id=$1`, id); err != nil { tx.Rollback(); return err }
    if err = tx.Commit(); err != nil { return err }
    _ = os.RemoveAll(s.sessionDir(id))
    return nil
}

func (s *DBStore) DeleteUser(email string) error {
    rows, err := s.db.Query(`SELECT id FROM sessions WHERE email=$1`, email)
    if err != nil { return err }
    defer rows.Close()
    ids := []string{}
    for rows.Next() { var id string; if err := rows.Scan(&id); err == nil { ids = append(ids, id) } }
    for _, id := range ids { _ = s.DeleteSession(id) }
    return nil
}

// Chunked upload implementation
func (s *DBStore) InitUpload(sessionID, filename, mime string, totalSize int64) (string, error) {
    if err := s.ensureTables(); err != nil { return "", err }
    // Enforce image MIME and max size (default 25MB)
    if !strings.HasPrefix(strings.ToLower(strings.TrimSpace(mime)), "image/") {
        return "", fmt.Errorf("unsupported format")
    }
    maxMB := 25
    if totalSize > int64(maxMB*1024*1024) { return "", fmt.Errorf("file too large") }
    if err := s.ensureTables(); err != nil { return "", err }
    if _, _, _, _, err := s.GetConfig(); err == nil { /* config is available */ }
    if totalSize > int64(maxMB*1024*1024) { return "", fmt.Errorf("file too large") }
    if err := s.ensureTables(); err != nil { return "", err }
    var count int
    if err := s.db.QueryRow(`SELECT image_count FROM sessions WHERE id=$1`, sessionID).Scan(&count); err != nil { return "", fmt.Errorf("not found") }
    if count >= s.cap { return "", fmt.Errorf("cap reached") }
    id := newID()
    tmp := filepath.Join(s.dir, "tmp_"+id)
    if err := os.WriteFile(tmp, []byte{}, 0o644); err != nil { return "", err }
    _, err := s.db.Exec(`INSERT INTO uploads(id, session_id, filename, mime, total_size, tmp_path) VALUES($1,$2,$3,$4,$5,$6)`, id, sessionID, filename, mime, totalSize, tmp)
    if err != nil { _ = os.Remove(tmp); return "", err }
    return id, nil
}

func (s *DBStore) UploadChunk(uploadID string, index int, body io.Reader) error {
    var tmp string
    if err := s.db.QueryRow(`SELECT tmp_path FROM uploads WHERE id=$1`, uploadID).Scan(&tmp); err != nil { return errors.New("upload not found") }
    f, err := os.OpenFile(tmp, os.O_APPEND|os.O_WRONLY, 0o644)
    if err != nil { return err }
    defer f.Close()
    n, err := io.Copy(f, body)
    if err != nil { return err }
    _, err = s.db.Exec(`UPDATE uploads SET received_size=received_size+$1 WHERE id=$2`, n, uploadID)
    return err
}

func (s *DBStore) CompleteUpload(uploadID string) (string, string, error) {
    var sessionID, filename, mime, tmp string
    var total, received int64
    if err := s.db.QueryRow(`SELECT session_id, filename, mime, total_size, received_size, tmp_path FROM uploads WHERE id=$1`, uploadID).Scan(&sessionID, &filename, &mime, &total, &received, &tmp); err != nil {
        return "", "", errors.New("upload not found")
    }
    if received != total { return "", "", fmt.Errorf("incomplete upload") }
    data, err := os.ReadFile(tmp)
    if err != nil { return "", "", err }
    // Parse focal length if present; store numeric only on session row (best-effort)
    if f, ok := readEXIFFocalLengthJPEG(data); ok {
        _, _ = s.db.Exec(`UPDATE sessions SET focal_length=$1 WHERE id=$2 AND focal_length IS NULL`, f, sessionID)
    }
    outBytes, outExt := reencodeToStripMetadata(data)
    if outBytes == nil {
        return "", "", fmt.Errorf("unsupported format")
    }
    h := sha256.Sum256(outBytes)
    hash := hex.EncodeToString(h[:])
    var exists int
    if e := s.db.QueryRow(`SELECT 1 FROM dedup_index WHERE sha256=$1`, hash).Scan(&exists); e == nil {
        _, _ = s.db.Exec(`DELETE FROM uploads WHERE id=$1`, uploadID)
        _ = os.Remove(tmp)
        return "", "", fmt.Errorf("duplicate")
    }
    serverName := fmt.Sprintf("%d_%s%s", time.Now().UnixNano(), hash[:8], outExt)
    outPath := filepath.Join(s.sessionDir(sessionID), serverName)
    if err := os.WriteFile(outPath, outBytes, 0o644); err != nil { return "", "", err }
    tx, e := s.db.Begin(); if e != nil { return "", "", e }
    var imgID int64
    if e = tx.QueryRow(`INSERT INTO images(session_id, server_filename, original_mime, size_bytes, sha256, hash_source) VALUES($1,$2,$3,$4,$5,$6) RETURNING id`, sessionID, serverName, mime, len(outBytes), hash, "stripped").Scan(&imgID); e != nil { tx.Rollback(); return "", "", e }
    if _, e = tx.Exec(`INSERT INTO dedup_index(sha256, image_id) VALUES($1,$2)`, hash, imgID); e != nil { tx.Rollback(); return "", "", e }
    if _, e = tx.Exec(`UPDATE sessions SET image_count=image_count+1 WHERE id=$1`, sessionID); e != nil { tx.Rollback(); return "", "", e }
    if _, e = tx.Exec(`DELETE FROM uploads WHERE id=$1`, uploadID); e != nil { tx.Rollback(); return "", "", e }
    if e = tx.Commit(); e != nil { return "", "", e }
    _ = os.Remove(tmp)
    return sessionID, serverName, nil
}

func (s *DBStore) SessionsQuery(email, name, mode, status string) []Session {
    where := `WHERE ($1='' OR LOWER(email)=LOWER($1))
          AND ($2='' OR LOWER(name) LIKE LOWER('%' || $2 || '%'))
          AND ($3='' OR mode=$3)`
    if strings.EqualFold(status, "finalized") { where += " AND finalized_at IS NOT NULL" } else if strings.EqualFold(status, "open") { where += " AND finalized_at IS NULL" }
    q := `SELECT id, created_at, finalized_at, name, handle, email, notify_opt_in, mode, image_count FROM sessions ` + where
    rows, err := s.db.Query(q, email, name, mode)
    if err != nil { return nil }
    defer rows.Close()
    out := []Session{}
    for rows.Next() {
        var sess Session
        var fin sql.NullTime
        if err := rows.Scan(&sess.ID, &sess.CreatedAt, &fin, &sess.Name, &sess.Handle, &sess.Email, &sess.NotifyOptIn, &sess.Mode, &sess.ImageCount); err == nil {
            if fin.Valid { t := fin.Time; sess.FinalizedAt = &t }
            out = append(out, sess)
        }
    }
    return out
}
