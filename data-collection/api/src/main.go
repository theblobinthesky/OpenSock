package main

import (
    "fmt"
    "log"
    "net/http"
    "os"
)

func main() {
    cfg := LoadConfig()
    dsn := os.Getenv("DATABASE_URL")
    if dsn == "" { log.Fatalf("DATABASE_URL must be set (Postgres is required)") }
    store, err := NewDBStore(dsn, cfg.StorageDir, cfg.SessionImageCap)
    if err != nil { log.Fatalf("db storage init: %v", err) }
    log.Printf("using Postgres-backed store")

    mux := http.NewServeMux()
    // Register Connect-Web (gRPC-web) handlers only — no REST.
    RegisterConnectHandlers(mux, store, cfg)

    // start periodic cleanup for expired sessions/uploads
    startCleanupLoop(store)

    mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        _, _ = w.Write([]byte("ok"))
    })

    addr := fmt.Sprintf(":%d", cfg.Port)
    log.Printf("listening on %s (storage: %s)", addr, cfg.StorageDir)
    if err := http.ListenAndServe(addr, rateLimit(cors(mux), 600)); err != nil {
        log.Println(err)
        os.Exit(1)
    }
}
