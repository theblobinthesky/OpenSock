import os, json, cv2, numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from native_dataloader import FileType, Dataset, Head
import jax, jax.numpy as jnp
import torch
from datasets import load_dataset
from functools import partial
from .config import DataConfig
from .utils import compute_iou_with_sides


dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to('cuda')
dinov2 = torch.compile(dinov2)

def get_num_samples_from_class_samples(num_class_samples: int, other_total_perc: float, other_part_perc: float) -> int:
    total_samples = int(num_class_samples / (1.0 - other_total_perc))
    return int(other_part_perc * total_samples)


# TODO: Unify "codebases"
def apply_homography(image: np.ndarray, homography: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = cv2.warpPerspective(image, homography, image.shape[:2])
    image = image.transpose((1, 0, 2))
    image = cv2.resize(image, size[::-1], interpolation=cv2.INTER_AREA)
    return image

def cut_down_samples_per_scope(neg_samples_per_scope: list[int], total_neg_samples: int):
    neg_samples_per_scope = [i for i in neg_samples_per_scope]
    current_total = sum(neg_samples_per_scope)
    while current_total > total_neg_samples:
        i = np.argmax(neg_samples_per_scope)
        if neg_samples_per_scope[i] > 0:
            neg_samples_per_scope[i] -= 1
            current_total -= 1
    
    return neg_samples_per_scope


def process_video_files_for_classifier(config: DataConfig, video_files: list[str]):
    # Prepare random sample of important frames to process.
    is_class_instance_per_video = {}
    video_and_frame_info = []
    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        json_path = os.path.join(config.annotations_dir, f"{video_name}.json")
            
        if not os.path.exists(json_path):
            print(f"Annotation '{json_path}' does not exist. Skipping!")
            continue
                
        with open(json_path, 'r') as f:
            annotation_data = json.load(f)

        is_class_instance_per_video[video_file] = {id: instance['is_class_instance'] for id, instance in annotation_data['instances'].items()}

        for frame_info in annotation_data['important_frames']:
            video_and_frame_info.append((video_file, frame_info))

    video_and_frame_info = sorted(video_and_frame_info, key=lambda item: item[0])

    def count_class_samples(is_class_instance: dict[int, bool], frame_info):
        ids = list(frame_info['data'].keys())
        total = len(ids)
        ids = [id for id in ids if is_class_instance[id]]
        return len(ids), total - len(ids)

    # First deal with the positive boxes.
    # Delete everything after the max number of boxes.
    total_class_samples = 0
    cutoff_video_and_frame_info = []
    for (video_file, frame_info) in video_and_frame_info:
        cutoff_video_and_frame_info.append((video_file, frame_info))

        num_class_samples, _ = count_class_samples(is_class_instance_per_video[video_file], frame_info)
        total_class_samples += num_class_samples

        if num_class_samples >= config.classifier_max_num_class_samples:
            break

    total_class_samples = min(total_class_samples, config.classifier_max_num_class_samples)

    frame_info_by_video = {}
    for (video_file, frame_info) in cutoff_video_and_frame_info:
        frame_info_by_video.setdefault(video_file, []).append(frame_info)
    frame_info_by_video = list(frame_info_by_video.items())


    # Calculate the number of negative boxes based on the available.
    total_negative_samples = get_num_samples_from_class_samples(total_class_samples, 
                                       config.classifier_negative_percentage + config.classifier_imagenet_percentage,
                                       config.classifier_negative_percentage)
    class_samples_per_video = [sum([len(frame_info['data']) for frame_info in frame_infos]) for _, frame_infos in frame_info_by_video]
    neg_samples_per_video = cut_down_samples_per_scope(class_samples_per_video, total_negative_samples)
    assert total_negative_samples == sum(neg_samples_per_video)


    # Define anchor boxes
    height, width = config.classifier_target_image_size
    anchor_scales = np.linspace(
        config.classifier_neg_anchors['min_size'] * min(height, width),
        config.classifier_neg_anchors['max_size'] * min(height, width),
        config.classifier_neg_anchors['steps']
    )
    anchors = []
    step_size = min(height, width) // 10  # Grid step for anchor placement
    for y in range(0, height - step_size, step_size):
        for x in range(0, width - step_size, step_size):
            for scale in anchor_scales:
                w = h = int(scale)
                if x + w <= width and y + h <= height:
                    anchors.append([x, y, x + w, y + h])


    # Process them.
    num_processed_class_samples = 0
    with tqdm(total=total_class_samples + total_negative_samples, desc='Processing class and negative samples') as pbar:
        for i, (video_file, frame_infos) in enumerate(frame_info_by_video):
            cap = cv2.VideoCapture(os.path.join(config.videos_dir, video_file))
            is_class_instance = is_class_instance_per_video[video_file]

            # Compute the number of negative samples per frame.
            neg_samples = neg_samples_per_video[i]
            num_neg_per_frame = [neg_samples for _ in range(len(frame_infos))]
            num_neg_per_frame = cut_down_samples_per_scope(num_neg_per_frame, neg_samples)
            assert neg_samples == sum(num_neg_per_frame)


            for frame_info, num_neg in zip(frame_infos, num_neg_per_frame):
                frame_idx = frame_info['index']
                frame_data = frame_info['data']
                homography = np.array(frame_info['stabilizer_homography'], np.float32).reshape((3, 3))

                # Extract frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = apply_homography(frame, homography, size=config.classifier_target_image_size)

                # Extract the positive samples.            
                for obj_id, obj_data in frame_data.items():
                    if num_processed_class_samples >= total_class_samples:
                        break 

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


                # Extract the negative samples
                if num_neg == 0:
                    continue

                height, width = frame.shape[:2]
                positive_bboxes = []
                for obj_id, obj_data in frame_data.items():
                    if not obj_data['is_occluded'] and is_class_instance[obj_id]:
                        y, x, h, w = obj_data['bbox']
                        positive_bboxes.append([x, y, x + w, y + h])


                # Filter negative anchors based on IoU
                negative_anchors = []
                np.random.shuffle(anchors)
                for anchor in anchors:
                    max_iou = 0
                    for pos_bbox in positive_bboxes:
                        iou = compute_iou_with_sides(anchor, pos_bbox)  # Implement this function
                        max_iou = max(max_iou, iou)

                    if max_iou < config.classifier_neg_class_iou_thresh:
                        negative_anchors.append(anchor)
                        if len(negative_anchors) >= num_neg: break

                if len(negative_anchors) < num_neg:
                    raise ValueError("There are too few negative anchors per frame.")


                # Extract and save negative samples
                for anchor_idx, anchor in enumerate(negative_anchors):
                    x_min, y_min, x_max, y_max = anchor
                    roi = frame[y_min:y_max, x_min:x_max].copy()
                    if roi.size == 0:
                        continue

                    roi = cv2.resize(roi, config.classifier_image_size)
                    label = np.array([0.0], np.float32)  # Negative label

                    file_name = f"{video_name}_{frame_idx:06d}_neg_{anchor_idx:04d}"
                    cv2.imwrite(os.path.join(config.classifier_output_dir, "images", f"{file_name}.jpg"), roi)
                    np.save(os.path.join(config.classifier_output_dir, "labels", f"{file_name}.npy"), label)
                    pbar.update(1)


    return total_class_samples


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

def pp_encode_images_into_features(batch: dict[str, jnp.ndarray]) -> jnp.ndarray:
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


def get_classifier_dataset(config: DataConfig) -> Dataset:

    def init_ds_fn():
        os.makedirs(os.path.join(config.classifier_output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(config.classifier_output_dir, "labels"), exist_ok=True)

        video_files = [f for f in os.listdir(config.videos_dir) if f.endswith('.mov')]
        num_class_samples = process_video_files_for_classifier(config, video_files)

        num_imagenet_samples = get_num_samples_from_class_samples(num_class_samples, 
                                                                  config.classifier_negative_percentage + config.classifier_imagenet_percentage, 
                                                                  config.classifier_imagenet_percentage)
        load_imagenet_for_classifier(config, config.classifier_avoid_class, num_imagenet_samples)


    ds = Dataset.from_subdirs(
        config.classifier_output_dir, 
        [
            Head(FileType.JPG, "images", (*config.classifier_image_size, 3)),
            Head(FileType.NPY, "labels", (1,)),
        ], 
        ["images", "labels"],
        init_ds_fn,
        post_process_fn=pp_encode_images_into_features,
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

#
# foreach video, instance
#   extract all class instances
#   find positive_examples for instance
#   find negative_examples for instance
#   data[video, instance] = pos, neg
#
#
#
#
#
#

def get_instance_tracker_dataset(config: DataConfig) -> Dataset:

    def init_ds_fn():
        os.makedirs(os.path.join(config.classifier_output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(config.classifier_output_dir, "labels"), exist_ok=True)

        video_files = [f for f in os.listdir(config.videos_dir) if f.endswith('.mov')]
        num_class_samples = process_video_files_for_classifier(config, video_files)

        num_imagenet_samples = get_num_samples_from_class_samples(num_class_samples, 
                                                                  config.classifier_negative_percentage + config.classifier_imagenet_percentage, 
                                                                  config.classifier_imagenet_percentage)
        load_imagenet_for_classifier(config, config.classifier_avoid_class, num_imagenet_samples)


    ds = Dataset.from_subdirs(
        config.classifier_output_dir, 
        [
            Head(FileType.JPG, "images", (*config.classifier_image_size, 3)),
            Head(FileType.NPY, "labels", (1,)),
        ], 
        ["images", "labels"],
        init_ds_fn,
        post_process_fn=pp_encode_images_into_features,
        is_virtual_dataset=True
    )

    return ds
