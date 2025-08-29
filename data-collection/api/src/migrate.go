package main

import (
    "database/sql"
    "embed"
    "fmt"
    "sort"
    "strings"
)

//go:embed migrations/*.sql
var migrationsFS embed.FS

func runMigrations(db *sql.DB) error {
    if _, err := db.Exec(`CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY)`); err != nil {
        return err
    }
    entries, err := migrationsFS.ReadDir("migrations")
    if err != nil { return err }
    names := make([]string, 0, len(entries))
    for _, e := range entries { if !e.IsDir() && strings.HasSuffix(e.Name(), ".sql") { names = append(names, e.Name()) } }
    sort.Strings(names)
    for _, name := range names {
        var exists int
        if err := db.QueryRow(`SELECT 1 FROM schema_migrations WHERE version=$1`, name).Scan(&exists); err == nil {
            continue
        }
        b, err := migrationsFS.ReadFile("migrations/" + name)
        if err != nil { return err }
        if _, err := db.Exec(string(b)); err != nil {
            return fmt.Errorf("apply migration %s: %w", name, err)
        }
        if _, err := db.Exec(`INSERT INTO schema_migrations(version) VALUES($1)`, name); err != nil {
            return err
        }
    }
    return nil
}

