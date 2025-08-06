import sqlite3
from flask import g
from typing import List
from pathlib import Path
from .config import InferenceConfig

DATABASE_DIR = Path("../data/backend/")

def get_conn():
    if "db" not in g:
        DATABASE_DIR.mkdir(exist_ok=True)
        conn = sqlite3.connect(DATABASE_DIR / "database.db")
        conn.cursor().executescript("""
            CREATE TABLE IF NOT EXISTS multipart_scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT
            );

            CREATE TABLE IF NOT EXISTS uploads (
                    upload_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    multipart_scan_id INTEGER NOT NULL,
                    FOREIGN KEY (multipart_scan_id) REFERENCES multipart_scans(id)
            );
        """)

        g.db = conn

    return g.db


def create_multipart_scan_if_necessary():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO multipart_scans DEFAULT VALUES")
    conn.commit()
    multipart_scan_id = cursor.lastrowid
    return multipart_scan_id


def insert_scan(multipart_scan_id: int):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
            INSERT INTO uploads (multipart_scan_id)
            VALUES (?)
    """, (multipart_scan_id,))
    conn.commit()
    upload_id = cursor.lastrowid
    return upload_id

# TODO: Duplicate code.
BACKEND_DATA_ROOT = "../data/backend"

def get_multipart_scan_uploads(config: InferenceConfig, multipart_scan_id: int) -> List[str]:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT multipart_scan_id, upload_id FROM uploads u
        INNER JOIN multipart_scans m ON u.multipart_scan_id = m.id
        WHERE m.id = ?;
        """, (multipart_scan_id,)
    )
    
    rows = cursor.fetchall()
    paths = [f"{config.get_multipart_input_dir()}/{row[0]}/{row[1]}.jpg" for row in rows]
    return paths
