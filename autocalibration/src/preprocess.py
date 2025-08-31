from dotenv import load_dotenv
load_dotenv()
import os, shutil
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel, SuperPointForKeypointDetection
from PIL import Image
from transformers.image_utils import load_image
import glob

def get_dino(model_id: str):
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    return processor, AutoModel.from_pretrained(model_id, device_map="auto")

def get_superpoint():
    processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
    model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
    return processor, model

def run_dino(processor, model, image_ref: str):
    image = load_image(image_ref)
    image = image.resize((518, 518))  # Resize to 518x518 as this is a multiple of 14x14 image patch size that dino uses.
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)

    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is None:
        raise ValueError("Model output does not have last_hidden_state")
    return last_hidden[:, 1:]

def run_superpoint(processor, model, image_ref: str):
    image = Image.open(image_ref)
    image = image.resize((640, 480))  # Resize to 640x480 for consistency with SuperPoint training
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    image_size = (image.height, image.width)
    processed_outputs = processor.post_process_keypoint_detection(outputs, [image_size])
    return processed_outputs

def main():
    dino_processor, dino_model = get_dino(os.environ['DINO_MODEL'])
    superpoint_processor, superpoint_model = get_superpoint()
    input_dir = "data/inputs"
    dino_output_dir = "data/dino-features"
    superpoint_output_dir = "data/superpoint-features"

    shutil.rmtree(dino_output_dir, ignore_errors=True)
    shutil.rmtree(superpoint_output_dir, ignore_errors=True)
    os.makedirs(dino_output_dir, exist_ok=True)
    os.makedirs(superpoint_output_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(input_dir, "*.png")) + glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.jpeg"))

    for image_path in image_paths:
        # DINO features
        features = run_dino(dino_processor, dino_model, image_path)
        base_name = os.path.basename(image_path).rsplit('.', 1)[0]
        dino_output_path = os.path.join(dino_output_dir, "{}_features.npy".format(base_name))
        np.save(dino_output_path, features.cpu().numpy())
        
        # SuperPoint features
        superpoint_outputs = run_superpoint(superpoint_processor, superpoint_model, image_path)
        superpoint_data = superpoint_outputs[0]
        superpoint_data['keypoints'] = superpoint_data['keypoints'].numpy()
        superpoint_data['scores'] = superpoint_data['scores'].numpy()
        superpoint_data['descriptors'] = superpoint_data['descriptors'].numpy()
        superpoint_output_path = os.path.join(superpoint_output_dir, "{}_superpoint.npy".format(base_name))
        np.save(superpoint_output_path, superpoint_data)

if __name__ == "__main__":
    main()
