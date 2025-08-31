package main

import "time"

type CreateSessionRequest struct {
    Name         string `json:"name"`
    Email        string `json:"email"`
    NotifyOptIn  bool   `json:"notify_opt_in"`
    Language     string `json:"language"`
    Mode         string `json:"mode"`
}

type Session struct {
    ID           string    `json:"id"`
    CreatedAt    time.Time `json:"created_at"`
    FinalizedAt  *time.Time `json:"finalized_at,omitempty"`
    Name         string    `json:"name,omitempty"`
    Email        string    `json:"email,omitempty"`
    NotifyOptIn  bool      `json:"notify_opt_in"`
    Language     string    `json:"language,omitempty"`
    Mode         string    `json:"mode,omitempty"`
    ImageCount   int       `json:"image_count"`
    Hashes       map[string]struct{} `json:"-"`
}
