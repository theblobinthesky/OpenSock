import os, shutil
import json
import cv2
import tensorflow as tf
from typing import Tuple
from tqdm import tqdm

class SimpleYOLOPreprocessor:
    def __init__(self, videos_dir: str, annotations_dir: str, output_dir: str):
        self.videos_dir = videos_dir
        self.annotations_dir = annotations_dir
        self.output_dir = output_dir
        self._process_all_videos()
        
    def _process_all_videos(self) -> None:
        if os.path.exists(self.output_dir): 
            if 'INVALIDATE_DATASET' in os.environ:
                shutil.rmtree(self.output_dir, ignore_errors=True)
            else:
                return
        
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels"), exist_ok=True)

        video_files = [f for f in os.listdir(self.videos_dir) if f.endswith('.mov')]
        
        for video_file in tqdm(video_files, desc='Processing video files'):
            video_name = os.path.splitext(video_file)[0]
            json_path = os.path.join(self.annotations_dir, f"{video_name}.json")
            
            if not os.path.exists(json_path):
                continue
                
            # Process video and extract frames with annotations
            cap = cv2.VideoCapture(os.path.join(self.videos_dir, video_file))
            
            with open(json_path, 'r') as f:
                annotation_data = json.load(f)
            
            for frame_info in annotation_data['important_frames']:
                frame_idx = frame_info['index']
                frame_data = frame_info['data']
                
                # Extract frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Save image
                img_path = os.path.join(self.output_dir, "images", f"{video_name}_{frame_idx:06d}.jpg")
                cv2.imwrite(img_path, frame)
                
                # Create YOLO label
                label_path = os.path.join(self.output_dir, "labels", f"{video_name}_{frame_idx:06d}.txt")
                
                height, width = frame.shape[:2]
                with open(label_path, 'w') as f:
                    for _, obj_data in frame_data.items():
                        x, y, w, h = obj_data['bbox']
                        
                        # Convert to YOLO format: class_id x_center y_center width height
                        x_center = (x + w/2) / width
                        y_center = (y + h/2) / height
                        w_norm = w / width
                        h_norm = h / height
                        
                        f.write(f"0 {x_center} {y_center} {w_norm} {h_norm}\n")
            
            cap.release()
    
    def create_dataset(self, batch_size: int = 16, img_size: Tuple[int, int] = (640, 640)):
        img_paths = tf.data.Dataset.list_files(os.path.join(self.output_dir, "images", "*.jpg"))
        
        def load_data(img_path):
            # Get corresponding label path
            label_path = tf.strings.regex_replace(img_path, "images", "labels")
            label_path = tf.strings.regex_replace(label_path, ".jpg", ".txt")
            
            # Load and preprocess image
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, img_size)
            img = img / 255.0
            
            # Load and parse label
            label_raw = tf.io.read_file(label_path)
            labels = tf.strings.split(label_raw, '\n')
            labels = tf.strings.split(labels, ' ')
            
            # Convert strings to numbers (excluding empty lines)
            mask = tf.strings.length(labels) > 0
            labels = tf.boolean_mask(labels, mask)
            labels = tf.strings.to_number(labels, out_type=tf.float32)
            labels = tf.reshape(labels, [-1, 5])  # Reshape to [num_objects, 5]
            
            return img, labels
        
        dataset = img_paths.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset