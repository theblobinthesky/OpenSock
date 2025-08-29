package main

// DB-backed config + cleanup helpers

// GetConfig returns current runtime caps from the DB config table.
func (s *DBStore) GetConfig() (cap int, minMixed int, minSame int, maxMB int, err error) {
    row := s.db.QueryRow(`SELECT session_image_cap, min_required_mixed, min_required_same, file_size_limit_mb FROM config WHERE id=1`)
    if err = row.Scan(&cap, &minMixed, &minSame, &maxMB); err != nil {
        return 0, 4, 3, 25, err
    }
    return
}

// SetConfig updates config table values.
func (s *DBStore) SetConfig(cap int, minMixed int, minSame int, maxMB int) error {
    _, err := s.db.Exec(`UPDATE config SET session_image_cap=$1, min_required_mixed=$2, min_required_same=$3, file_size_limit_mb=$4 WHERE id=1`, cap, minMixed, minSame, maxMB)
    return err
}
