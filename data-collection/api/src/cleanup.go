package main

import (
    "log"
    "time"
)

func startCleanupLoop(store Store) {
    go func() {
        ticker := time.NewTicker(1 * time.Hour)
        defer ticker.Stop()
        // initial run after 10s
        time.Sleep(10 * time.Second)
        _ = store.CleanupExpired(24)
        for range ticker.C {
            if err := store.CleanupExpired(24); err != nil {
                log.Printf("cleanup error: %v", err)
            }
        }
    }()
}
