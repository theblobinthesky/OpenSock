#!/usr/bin/env python3
import requests
import sys
from pathlib import Path
from typing import Optional


def upload_scan(multipart_scan_id: Optional[int], file_path: str, host: str = "http://localhost:5000"):
    file_path = Path(file_path)
    
    if multipart_scan_id is None:
        url = f"{host}/upload_file/new"
    else:
        url = f"{host}/upload_file/{multipart_scan_id}"

    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f, "image/jpeg")}
        resp = requests.post(url, files=files)

    if resp.ok:
        print(f"Upload succeeded (HTTP {resp.status_code})")
    else:
        print(f"Upload failed (HTTP {resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)

    return resp.content


def main():
    # Configure these as needed
    host = "http://127.0.0.1:5000"

    multipart_scan_id = int(upload_scan(None, "../data/scan0.jpg", host))
    upload_scan(multipart_scan_id, "../data/scan1.jpg", host)
    upload_scan(multipart_scan_id, "../data/scan2.jpg", host)
    upload_scan(multipart_scan_id, "../data/scan3.jpg", host)

    resp = requests.get(f"{host}/get_multipart_scan/{multipart_scan_id}")

if __name__ == "__main__":
    main()
