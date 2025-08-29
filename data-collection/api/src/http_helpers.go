package main

import (
    "net/http"
    "strings"
    "sync"
    "time"
)

func methods(method string, h http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        if r.Method != method {
            http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
            return
        }
        h.ServeHTTP(w, r)
    })
}

func withJSON(h http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        h.ServeHTTP(w, r)
    })
}

func cors(h http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        if r.Method == http.MethodOptions { w.WriteHeader(http.StatusNoContent); return }
        h.ServeHTTP(w, r)
    })
}

func pathParam(path, prefix string) (id, tail string) {
    if !strings.HasPrefix(path, prefix) { return "", "" }
    rest := strings.TrimPrefix(path, prefix)
    parts := strings.SplitN(rest, "/", 2)
    id = parts[0]
    if len(parts) == 2 { tail = "/" + parts[1] } else { tail = "" }
    return
}



// very simple in-memory rate limiter per remote IP
type ipLimiter struct {
    mu sync.Mutex
    hits map[string]int
    window time.Time
    limit int
}

func newIPLimiter(limit int) *ipLimiter { return &ipLimiter{hits: map[string]int{}, window: time.Now(), limit: limit} }

func (rl *ipLimiter) allow(ip string) bool {
    rl.mu.Lock(); defer rl.mu.Unlock()
    now := time.Now()
    if now.Sub(rl.window) > time.Minute {
        rl.hits = map[string]int{}; rl.window = now
    }
    rl.hits[ip]++
    return rl.hits[ip] <= rl.limit
}

func rateLimit(h http.Handler, limitPerMin int) http.Handler {
    rl := newIPLimiter(limitPerMin)
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        ip := r.RemoteAddr
        if !rl.allow(ip) {
            http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
            return
        }
        h.ServeHTTP(w, r)
    })
}
