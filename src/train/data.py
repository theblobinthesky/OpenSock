import os, json, cv2, numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from native_dataloader import FileType, Dataset, Head
import jax, jax.numpy as jnp
import torch
from datasets import load_dataset
from functools import partial
from .config import DataConfig


dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to('cuda')
dinov2 = torch.compile(dinov2)

# TODO: Unify "codebases"
def apply_homography(image: np.ndarray, homography: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = cv2.warpPerspective(image, homography, image.shape[:2])
    image = image.transpose((1, 0, 2))
    image = cv2.resize(image, size[::-1], interpolation=cv2.INTER_AREA)
    return image


def process_video_files_for_classifier(config: DataConfig, video_files: list[str]):
    num_class_samples = 0

    # Prepare random sample of important frames to process.
    video_and_frame_info = []
    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        json_path = os.path.join(config.annotations_dir, f"{video_name}.json")
            
        if not os.path.exists(json_path):
            print(f"Annotation '{json_path}' does not exist. Skipping!")
            continue
                
        with open(json_path, 'r') as f:
            annotation_data = json.load(f)

        for frame_info in annotation_data['important_frames']:
            video_and_frame_info.append((video_file, frame_info))

    video_and_frame_info = sorted(video_and_frame_info, key=lambda item: item[0])

    # Delete everything after the max number of boxes.
    num_class_samples = 0
    cutoff_video_and_frame_info = []
    for (video_file, frame_info) in video_and_frame_info:
        cutoff_video_and_frame_info.append((video_file, frame_info))

        num_class_samples += len(frame_info['data'].keys())
        if num_class_samples >= config.classifier_max_num_class_samples:
            break

    frame_info_by_video = {}
    for (video_file, frame_info) in video_and_frame_info:
        frame_info_by_video.setdefault(video_file, []).append(frame_info)


    # Process them.
    num_processed_class_samples = 0
    with tqdm(total=num_class_samples, desc='Processing class images') as pbar:
        for video_file, frame_indices in frame_info_by_video.items():
            cap = cv2.VideoCapture(os.path.join(config.videos_dir, video_file))
            
            for frame_info in frame_indices:
                frame_idx = frame_info['index']
                frame_data = frame_info['data']
                homography = np.array(frame_info['stabilizer_homography'], np.float32).reshape((3, 3))
                
                # Extract frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Save metadata. 
                frame = apply_homography(frame, homography, size=config.classifier_target_image_size)
                for obj_id, obj_data in frame_data.items():
                    obj_id = int(obj_id)
                    x, y, w, h = obj_data['bbox']
                    if obj_data['is_occluded']:
                        continue

                    roi = frame[x:x+w, y:y+h].copy()
                    if roi.size == 0:
                        continue

                    roi = cv2.resize(roi, config.classifier_image_size)
                    label = np.array([1.0], np.float32)

                    file_name = f"{video_name}_{frame_idx:06d}_{obj_id:04d}"
                    cv2.imwrite(os.path.join(config.classifier_output_dir, "images", f"{file_name}.jpg"), roi)
                    np.save(os.path.join(config.classifier_output_dir, "labels", f"{file_name}.npy"), label)
                    num_processed_class_samples += 1
                    pbar.update(1) 
                    
                    if num_processed_class_samples >= num_class_samples:
                        return num_class_samples

            cap.release()

    return num_class_samples


def load_imagenet_for_classifier(config: DataConfig, avoided_class: int, num_samples: int):
    dataset_stream = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    sample_idx = 0

    with tqdm(total=num_samples, desc='Loading imagenet samples') as pbar:
        for sample in dataset_stream:
            if sample['label'] != avoided_class:
                roi = np.array(sample['image'])
                roi = cv2.resize(roi, config.classifier_image_size)
                label = np.array([0.0], np.float32)

                file_name = f"imagenet_{sample_idx:06d}"
                cv2.imwrite(os.path.join(config.classifier_output_dir, "images", f"{file_name}.jpg"), roi)
                np.save(os.path.join(config.classifier_output_dir, "labels", f"{file_name}.npy"), label)
                sample_idx += 1
                pbar.update(1)
                
                if sample_idx >= num_samples:
                    break


def get_classifier_dataset(config: DataConfig) -> Dataset:

    def init_ds_fn():
        os.makedirs(os.path.join(config.classifier_output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(config.classifier_output_dir, "labels"), exist_ok=True)

        video_files = [f for f in os.listdir(config.videos_dir) if f.endswith('.mov')]
        num_class_samples = process_video_files_for_classifier(config, video_files)

        total_imagenet_samples = int(num_class_samples / (1.0 - config.classifier_imagenet_percentage))
        num_imagenet_samples = int(config.classifier_imagenet_percentage * total_imagenet_samples)
        load_imagenet_for_classifier(config, config.classifier_avoid_class, num_imagenet_samples)


    def post_process_function(batch: dict[str, jnp.ndarray]) -> jnp.ndarray:
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                images = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(batch['images']))
                images = torch.permute(images, (0, 3, 1, 2))
                features = dinov2(images).contiguous()
                jax_features = jnp.from_dlpack(torch.utils.dlpack.to_dlpack(features))

        return {
            'features': jax_features,
            'labels': batch['labels']
        }


    ds = Dataset.from_subdirs(
        config.classifier_output_dir, 
        [
            Head(FileType.JPG, "images", (*config.classifier_image_size, 3)),
            Head(FileType.NPY, "labels", (1,)),
        ], 
        ["images", "labels"],
        init_ds_fn,
        post_process_fn=post_process_function,
        is_virtual_dataset=True
    )

    return ds


def process_video_file_for_obj_detection(config: DataConfig, video_file: str):
    video_name = os.path.splitext(video_file)[0]
    json_path = os.path.join(config.annotations_dir, f"{video_name}.json")
    
    if not os.path.exists(json_path):
        print(f"Annotation '{json_path}' does not exist. Skipping!")
        return
        
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
        file_name = f"{video_name}_{frame_idx:06d}"
        img_path = os.path.join(config.output_dir, "images", f"{file_name}.jpg") 
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

        sx, sy = 1.0 / shape[1], 1.0 / shape[0]
        labels[:, :4] *= [sx, sy, sx, sy]

        np.save(os.path.join(config.output_dir, "labels", f"{file_name}.npy"), labels.astype(np.float32))

    cap.release()


def get_obj_detection_dataset(config: DataConfig) -> Dataset:

    def init_ds_fn():
        os.makedirs(os.path.join(config.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "labels"), exist_ok=True)

        video_files = [f for f in os.listdir(config.videos_dir) if f.endswith('.mov')]
        partial_func = partial(process_video_file_for_obj_detection, config)

        # More than two workers don't improve performance, 
        # as this is just to improve cpu utilization of the already multithreaded code.
        process_map(partial_func, video_files, max_workers=2, desc='Processing video files')


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
            Head(FileType.JPG, "images", (*config.img_size, 3)),
            Head(FileType.NPY, "labels", (config.max_objs_per_image, 6)),
        ], 
        ["images", "labels"],
        init_ds_fn,
        post_process_fn=post_process_fn,
        is_virtual_dataset=True
    )

    return ds
