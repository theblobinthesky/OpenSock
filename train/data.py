import os, shutil, json, cv2, numpy as np
# import tensorflow as tf
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
                print(f"Annotation '{json_path}' does not exist. Skipping!")
                continue
                
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
                
                # Save labels as a numpy array with absolute bbox coordinates: [class, x, y, w, h]
                label_path = os.path.join(self.output_dir, "labels", f"{video_name}_{frame_idx:06d}.npy")
                labels = []
                for _, obj_data in frame_data.items():
                    x, y, w, h = obj_data['bbox']
                    labels.append([0, x, y, w, h])
                labels = np.array(labels, dtype=np.float32)
                np.save(label_path, labels)
            
            cap.release()
    
    def _load_data(self, img_size: Tuple[int, int]):
        # def load_data(img_path):
        #     def _load(path):
        #         # Convert the Tensor string to a normal Python string
        #         path = path.numpy().decode('utf-8')
        #         # Read and decode image
        #         img_raw = tf.io.read_file(path).numpy()
        #         img_decoded = tf.image.decode_jpeg(img_raw, channels=3).numpy()
        #         orig_shape = img_decoded.shape[:2]  # (height, width)
        #         # Resize image
        #         img_resized = tf.image.resize(img_decoded, img_size).numpy()
        #         img_resized = img_resized / 255.0

        #         # Load labels from corresponding .npy file
        #         label_path = path.replace("images", "labels").replace(".jpg", ".npy")
        #         labels = np.load(label_path).astype(np.float32)
        #         # Compute scaling factors to adjust bbox coordinates
        #         scale_x = 1.0 / orig_shape[1]
        #         scale_y = 1.0 / orig_shape[0]
        #         labels_scaled = labels.copy()
        #         labels_scaled[:, 1] *= scale_x
        #         labels_scaled[:, 2] *= scale_y
        #         labels_scaled[:, 3] *= scale_x
        #         labels_scaled[:, 4] *= scale_y
        #         bboxes = labels_scaled[:, 1:5]
        #         classes = labels_scaled[:, 0]
        #         return img_resized.astype(np.float32), bboxes.astype(np.float32), classes.astype(np.float32)
            
        #     img, bboxes, classes = tf.py_function(
        #         func=_load, 
        #         inp=[img_path], 
        #         Tout=[tf.float32, tf.float32, tf.float32]
        #     )
        #     # Optionally, set static shapes if you know them.
        #     return img, (bboxes, classes)
        # return load_data
        pass

    def create_dataset(self, batch_size: int = 16, img_size: Tuple[int, int] = (640, 640)):
        # img_paths = tf.data.Dataset.list_files(os.path.join(self.output_dir, "images", "*.jpg"))
        # load_data_fn = self._load_data(img_size)
        # dataset = img_paths.map(load_data_fn, num_parallel_calls=tf.data.AUTOTUNE)
        # dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        # return dataset
        pass

    def create_train_valid_datasets(self, batch_size: int = 16, img_size: Tuple[int, int] = (640, 640),
                                    valid_split: float = 0.1, seed: int = 42):
        return None, None
        # # Gather all image paths
        # all_img_paths = tf.io.gfile.glob(os.path.join(self.output_dir, "images", "*.jpg"))
        # np.random.seed(seed)
        # np.random.shuffle(all_img_paths)
        # split_index = int(len(all_img_paths) * (1 - valid_split))
        # train_paths = all_img_paths[:split_index]
        # valid_paths = all_img_paths[split_index:]
        
        # train_ds = tf.data.Dataset.from_tensor_slices(train_paths)
        # valid_ds = tf.data.Dataset.from_tensor_slices(valid_paths)
        # load_data_fn = self._load_data(img_size)
        # train_ds = train_ds.map(load_data_fn, num_parallel_calls=tf.data.AUTOTUNE)
        # valid_ds = valid_ds.map(load_data_fn, num_parallel_calls=tf.data.AUTOTUNE)
        # train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        # valid_ds = valid_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        # return train_ds, valid_ds
