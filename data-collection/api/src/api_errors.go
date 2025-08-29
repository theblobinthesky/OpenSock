package main

import (
    "encoding/json"
    "net/http"
)

type MinImagesError struct {
    Required int
    Have     int
}

func (e *MinImagesError) Error() string {
    return "minimum images not met"
}

func writeError(w http.ResponseWriter, status int, key string, params map[string]any) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    _ = json.NewEncoder(w).Encode(map[string]any{
        "error_key": key,
        "params":    params,
    })
}

