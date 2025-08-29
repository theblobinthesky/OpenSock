package main

import (
    "os"
    "strconv"
)

type Config struct {
    Port             int
    StorageDir       string
    SessionImageCap  int
    AdminPassword    string
}

func getenv(key, def string) string {
    if v := os.Getenv(key); v != "" {
        return v
    }
    return def
}

func LoadConfig() Config {
    port, _ := strconv.Atoi(getenv("PORT", "8080"))
    capVal, _ := strconv.Atoi(getenv("SESSION_IMAGE_CAP", "20"))
    return Config{
        Port:            port,
        StorageDir:      getenv("STORAGE_DIR", "./storage"),
        SessionImageCap: capVal,
        AdminPassword:   os.Getenv("ADMIN_PASSWORD"),
    }
}

