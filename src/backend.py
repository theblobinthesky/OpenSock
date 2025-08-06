from inference.config import InferenceConfig
from flask import Flask, g, request, abort, send_file
from inference.persistence import create_multipart_scan_if_necessary, insert_scan
from inference.multipart_processor import process_multipart_scan
from pathlib import Path


app = Flask(__name__)
config = InferenceConfig()


def upload_file_into_multipart_scan_id(multipart_scan_id: int):
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".jpg"):
        abort(400)

    upload_id = insert_scan(multipart_scan_id)

    file_bytes = file.read()
    multipart_input_dir = config.get_multipart_input_dir()
    multipart_scan_dir = f"{multipart_input_dir}/{multipart_scan_id}"
    Path(multipart_input_dir).mkdir(exist_ok=True)
    Path(multipart_scan_dir).mkdir(exist_ok=True)
    file_path = f"{multipart_scan_dir}/{upload_id}.jpg"
    with open(file_path, "wb") as out:
        out.write(file_bytes)


@app.route("/upload_file/new", methods=["POST"])
def upload_file_without_existing_multipart_scan_id():
    multipart_scan_id = create_multipart_scan_if_necessary()
    upload_file_into_multipart_scan_id(multipart_scan_id)
    return str(multipart_scan_id), 200


@app.route("/upload_file/<int:multipart_scan_id>", methods=["POST"])
def upload_file(multipart_scan_id: int):
    upload_file_into_multipart_scan_id(multipart_scan_id)
    return "", 200


@app.route("/get_multipart_scan/<int:multipart_scan_id>", methods=["GET"])
def get_mulipart_scan(multipart_scan_id: int):
    path = process_multipart_scan(config, multipart_scan_id)
    return send_file(path, mimetype="image/jpeg")


@app.teardown_appcontext
def close_db(exc):
    db = g.pop('db', None)
    if db is not None:
        db.close()
  

if __name__ == "__main__":
    app.run()
