import sqlite3
from flask import g
from typing import List

DATABASE_PATH = "../data/database.db"

def get_conn():
    if "db" not in g:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.cursor().executescript("""
            CREATE TABLE IF NOT EXISTS multipart_scans (
                    id INTEGER PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS uploads (
                    upload_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    multipart_scan_id INTEGER NOT NULL,
                    FOREIGN KEY (multipart_scan_id) REFERENCES multipart_scans(id)
            );
        """)

        g.db = conn

    return g.db


def create_multipart_scan_if_necessary(multipart_scan_id: int):
    cursor = get_conn().cursor()
    cursor.execute("""
            INSERT OR IGNORE INTO multipart_scans (id)
            VALUES (?)
    """, (multipart_scan_id,))


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

def get_multipart_scan_uploads(multipart_scan_id: int) -> List[str]:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT multipart_scan_id, upload_id FROM uploads u
        INNER JOIN multipart_scans m ON u.multipart_scan_id = m.id
        WHERE m.id = ?;
        """, (multipart_scan_id,)
    )
    
    rows = cursor.fetchall()
    paths = [f"{BACKEND_DATA_ROOT}/{row[0]}_{row[1]}.jpg" for row in rows]
    return paths
