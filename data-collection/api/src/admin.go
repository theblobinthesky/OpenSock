package main

import "net/http"

// adminAuth: simple Basic Auth middleware for AdminService
func adminAuth(password string, next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        if password == "" { http.Error(w, "admin disabled", http.StatusForbidden); return }
        user, pass, ok := r.BasicAuth()
        if !ok || pass != password {
            w.Header().Set("WWW-Authenticate", "Basic realm=Restricted")
            http.Error(w, "unauthorized", http.StatusUnauthorized)
            return
        }
        _ = user // ignored
        next.ServeHTTP(w, r)
    })
}
