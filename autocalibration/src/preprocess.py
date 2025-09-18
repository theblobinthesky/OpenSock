import argparse
import glob
import os
import shutil
import numpy as np
import torch
import torch.nn.functional as f
from transformers import AutoImageProcessor, AutoModel, SuperPointForKeypointDetection
from PIL import Image
from tqdm import tqdm

dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dino_model = AutoModel.from_pretrained('facebook/dinov2-base', device_map="cuda")
DINO_EMB_DIM = 768
MAX_NUM_POINTS = 1024

def compute_dino_feature(img: Image.Image) -> np.ndarray:
    image = img.resize((518, 518))
    inputs = dino_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = dino_model(**inputs.to("cuda"))
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states[:, 1:].reshape((16, 16, DINO_EMB_DIM))


sp_processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
sp_model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint", device_map="cuda")


def compute_superpoint_like_descriptors(img: Image.Image) -> np.ndarray:
    inputs = sp_processor(img, return_tensors="pt")
    with torch.no_grad():
        outputs = sp_model(**inputs.to("cuda"))

    # Post-process to get keypoints, scores, and descriptors
    image_size = (img.height, img.width)
    processed_outputs = sp_processor.post_process_keypoint_detection(outputs, [image_size])
    return processed_outputs[0]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def bilinear_sample(dino_feat: np.ndarray, keyp: np.ndarray):
    dino_feat = dino_feat.permute((2, 0, 1))
    keyp = 2.0 * keyp - 1.0
    N = keyp.shape[0]
    D = dino_feat.shape[0]
    dino_feat = f.grid_sample(dino_feat[None, :], keyp.reshape((1, 1, N, 2)), mode="bilinear")
    dino_feat = dino_feat.reshape((D, N)).permute((1, 0))
    return dino_feat.cpu().numpy()

def process_directory(input_dir: str, dino_output_dir: str, sp_output_dir, label: str):
    # Collect image files
    exts = ("*.png", "*.jpg", "*.jpeg")
    image_paths = []
    for e in exts:
        image_paths.extend(glob.glob(os.path.join(input_dir, e)))
    image_paths.sort()

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    for path in tqdm(image_paths):
        base = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")

        dino_feat = compute_dino_feature(img)
        sp_desc = compute_superpoint_like_descriptors(img)

        keyp = sp_desc["keypoints"].float()
        keyp[:, 0] /= float(img.size[0]) - 1.0
        keyp[:, 1] /= float(img.size[1]) - 1.0

        desc = sp_desc["descriptors"]
        scores = sp_desc["scores"]

        if len(scores) > MAX_NUM_POINTS:
            indices = torch.argsort(-scores)[:, None] # descending
            keyp = torch.take_along_dim(sp_desc["keypoints"], indices, dim=0)[:MAX_NUM_POINTS]
            desc = torch.take_along_dim(sp_desc["descriptors"], indices, dim=0)[:MAX_NUM_POINTS]

        np.save(os.path.join(dino_output_dir, f"{base}.npy"), bilinear_sample(dino_feat, keyp))
        np.save(os.path.join(sp_output_dir, f"{base}.npy"), {
            "points": keyp.to(torch.float16).cpu().numpy(),
            "features": desc.to(torch.float16).cpu().numpy()
        })

        tqdm.write(f"Processed {label}: {base}")


def run(camera_0_dir: str, camera_1_dir: str, 
        camera_0_dino: str, camera_0_sp: str,
        camera_1_dino: str, camera_1_sp: str):
    # Prepare outputs
    shutil.rmtree(camera_0_dino, ignore_errors=True)
    shutil.rmtree(camera_0_sp, ignore_errors=True)
    shutil.rmtree(camera_1_dino, ignore_errors=True)
    shutil.rmtree(camera_1_sp, ignore_errors=True)
    ensure_dir(camera_0_dino)
    ensure_dir(camera_0_sp)
    ensure_dir(camera_1_dino)
    ensure_dir(camera_1_sp)

    # Process camera_0
    process_directory(camera_0_dir, camera_0_dino, camera_0_sp, "camera_0")

    # Process camera_1
    process_directory(camera_1_dir, camera_1_dino, camera_1_sp, "camera_1")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lightweight feature extraction (offline)")
    ap.add_argument("--camera-0", default="data/dataset/camera_0")
    ap.add_argument("--camera-1", default="data/dataset/camera_1")
    ap.add_argument("--camera-0-dino", default="data/dataset/camera_0_dino")
    ap.add_argument("--camera-0-sp", default="data/dataset/camera_0_sp")
    ap.add_argument("--camera-1-dino", default="data/dataset/camera_1_dino")
    ap.add_argument("--camera-1-sp", default="data/dataset/camera_1_sp")
    args = ap.parse_args(argv)
    run(
        args.camera_0, args.camera_1, 
        args.camera_0_dino, args.camera_0_sp,
        args.camera_1_dino, args.camera_1_sp
    )


if __name__ == "__main__":
    main()
