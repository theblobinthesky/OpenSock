from flask import Flask, g, request, abort, send_file
from inference.persistence import create_multipart_scan_if_necessary, insert_scan
from inference.multipart_processor import process_multipart_scan
import os

BACKEND_DATA_ROOT = "../data/backend"

app = Flask(__name__)

@app.route("/upload_file/<int:multipart_scan_id>", methods=["POST"])
def upload_file(multipart_scan_id: int):
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".jpg"):
        abort(400)

    create_multipart_scan_if_necessary(multipart_scan_id)
    upload_id = insert_scan(multipart_scan_id)

    file_bytes = file.read()
    if not os.path.exists(BACKEND_DATA_ROOT): os.mkdir(BACKEND_DATA_ROOT)
    file_path = f"{BACKEND_DATA_ROOT}/{multipart_scan_id}_{upload_id}.jpg"
    print(file_path)
    with open(file_path, "wb") as out:
        out.write(file_bytes)


    return "", 200


@app.route("/get_multipart_scan/<int:multipart_scan_id>", methods=["GET"])
def get_mulipart_scan(multipart_scan_id: int):
    process_multipart_scan(multipart_scan_id)
    path = "..."
    return send_file(path, mimetype="image/jpeg")


@app.teardown_appcontext
def close_db(exc):
    db = g.pop('db', None)
    if db is not None:
        db.close()
  

if __name__ == "__main__":
    app.run()
