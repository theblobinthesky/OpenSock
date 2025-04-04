import os, json, cv2, numpy as np
from tqdm import tqdm
from native_dataloader import FileType, Dataset, Head
from config import DataConfig
import jax.numpy as jnp

def get_obj_detection_dataset(config: DataConfig) -> Dataset:

    def init_ds_fn():
        os.makedirs(os.path.join(config.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "bboxes"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "classes"), exist_ok=True)

        video_files = [f for f in os.listdir(config.videos_dir) if f.endswith('.mov')]
        
        for video_file in tqdm(video_files, desc='Processing video files'):
            video_name = os.path.splitext(video_file)[0]
            json_path = os.path.join(config.annotations_dir, f"{video_name}.json")
            
            if not os.path.exists(json_path):
                print(f"Annotation '{json_path}' does not exist. Skipping!")
                continue
                
            cap = cv2.VideoCapture(os.path.join(config.videos_dir, video_file))
            
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
                shape = frame.shape
                file_name = f"{video_name}_{frame_idx:06d}.jpg"
                img_path = os.path.join(config.output_dir, "images", file_name) 
                cv2.imwrite(img_path, frame)

                # Save metadata. 
                labels = []
                for _, obj_data in frame_data.items():
                    x, y, w, h = obj_data['bbox']
                    mask = 0.0 if obj_data['is_occluded'] else 1.0
                    labels.append([x, y, w, h, mask, 0.0])

                for _ in range(config.max_objs_per_image - len(labels)):
                    labels.append([0, 0, 0, 0, 0.0, 0.0])

                labels = np.array(labels)

                sx, sy = 1.0 / shape[1], = 1.0 / shape[0]
                labels[:, :4] *= [sx, sy, sx, sy]

                np.save(os.path.join(config.output_dir, "labels", file_name), labels)


            cap.release()


    def post_process_fn(batch: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        labels = batch['labels']

        return {
            'img': batch['img'],
            'bboxes': labels[:, :4],
            'masks': labels[:, 4],
            'classes': labels[:, 5]
        }


    ds = Dataset.from_subdirs(
        config.output_dir, 
        [
            Head(FileType.JPG, "images", (*config['img_size'], 3)),
            Head(FileType.NPY, "bboxes", (config.max_objs_per_image, 4)),
            Head(FileType.NPY, "masks", (config.max_objs_per_image,)),
            Head(FileType.NPY, "classes", (config.max_objs_per_image,)),
        ], 
        ["images", "bboxes", "masks", "classes"],
        init_ds_fn,
        post_process_fn=post_process_fn,
        is_virtual_dataset=True
    )

    return ds
