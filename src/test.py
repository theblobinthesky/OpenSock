#!/usr/bin/env python3
import requests
import sys
from pathlib import Path

def upload_scan(instance_id: int, file_path: str, host: str = "http://localhost:5000"):
    file_path = Path(file_path)
    url = f"{host}/upload_file/{instance_id}"
    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f, "image/jpeg")}
        resp = requests.post(url, files=files)

    if resp.ok:
        print(f"Upload succeeded (HTTP {resp.status_code})")
    else:
        print(f"Upload failed (HTTP {resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)

def main():
    # Configure these as needed
    instance_id = 0
    host = "http://127.0.0.1:5000"

    upload_scan(instance_id, "../data/scan0.jpg", host)
    upload_scan(instance_id, "../data/scan1.jpg", host)
    upload_scan(instance_id, "../data/scan2.jpg", host)
    upload_scan(instance_id, "../data/scan3.jpg", host)

    resp = requests.get(f"{host}/process_multipart_scan/{instance_id}", files=files)

if __name__ == "__main__":
    main()
