package main

import (
    "os"
)

// CleanupExpired removes uploads older than maxAgeHours and unfinalized sessions older than maxAgeHours.
func (s *DBStore) CleanupExpired(maxAgeHours int) error {
    // delete old uploads tmp files and rows
    rows, err := s.db.Query(`SELECT id, tmp_path FROM uploads WHERE now() - created_at > ($1 || ' hours')::interval`, maxAgeHours)
    if err == nil {
        defer rows.Close()
        for rows.Next() {
            var id, tmp string
            if err := rows.Scan(&id, &tmp); err == nil {
                _ = os.Remove(tmp)
            }
        }
        _, _ = s.db.Exec(`DELETE FROM uploads WHERE now() - created_at > ($1 || ' hours')::interval`, maxAgeHours)
    }
    // delete stale sessions (not finalized)
    srows, err := s.db.Query(`SELECT id FROM sessions WHERE finalized_at IS NULL AND now() - created_at > ($1 || ' hours')::interval`, maxAgeHours)
    if err == nil {
        defer srows.Close()
        for srows.Next() {
            var id string
            if err := srows.Scan(&id); err == nil {
                _ = s.DeleteSession(id)
            }
        }
    }
    return nil
}
