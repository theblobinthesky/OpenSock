package main

import (
    "io"
    "net/http"
)

type Store interface {
    CreateSession(req CreateSessionRequest) (*Session, error)
    UploadImages(r *http.Request, sessionID string) (uploaded int, duplicates int, remaining int, err error)
    FinalizeSession(id string) error
    SessionsByEmail(email string) []Session
    SessionsQuery(email, name, mode, status string) []Session
    DeleteSession(id string) error
    DeleteUser(email string) error
    // Chunked uploads
    InitUpload(sessionID, filename, mime string, totalSize int64) (uploadID string, err error)
    UploadChunk(uploadID string, index int, body io.Reader) error
    CompleteUpload(uploadID string) (sessionID string, serverFilename string, err error)
    // Config
    GetConfig() (cap int, minMixed int, minSame int, maxMB int, err error)
    SetConfig(cap int, minMixed int, minSame int, maxMB int) error
    // Expiry cleanup
    CleanupExpired(maxAgeHours int) error
}
