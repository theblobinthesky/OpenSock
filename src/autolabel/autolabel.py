import os
os.environ['TQDM_DISABLE'] = '1'
import cv2
import numpy as np
import logging
import time
from .config import BaseConfig
from .trackers import Stabilizer, ImageTracker, VideoTracker, apply_homography


def get_interesting_frames(cap, skip_frames: int, output_size: tuple[int, int],
                        diff_threshold: float,
                        max_interesting_frames: int,
                        no_change_threshold: float = 0.01,
                        consecutive_static_required: int = 3) -> list[int]:
    def _process(frame: np.ndarray):
        frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 1
    frames = [0]
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to retrieve frame.")
        return []
    
    original_size = prev_frame.shape[:-1]
    prev_frame = _process(prev_frame)
    
    static_count = 0

    while frame_idx < total_frames:
        new_frame_idx = int(min(total_frames - 1, cap.get(cv2.CAP_PROP_POS_FRAMES) + skip_frames))
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_idx)
        frame_idx = new_frame_idx

        ret, frame = cap.read()
        if not ret:
            print("Failed to retrieve frame.")
            return []

        size = frame.shape[:-1]
        if original_size != size:
            raise ValueError("The frame sizes have to be consistent.")

        gray = _process(frame)
        diff = cv2.absdiff(prev_frame, gray)
        diff_norm = np.sum(diff) / (diff.shape[0] * diff.shape[1] * 255)

        if diff_norm > diff_threshold:
            frames.append(frame_idx)
            prev_frame = gray
            static_count = 0  # reset static counter on movement
        elif diff_norm < no_change_threshold:
            static_count += 1
            if static_count >= consecutive_static_required:
                frames.append(frame_idx)
                static_count = 0  # reset after flagging static sequence
        else:
            static_count = 0  # reset if difference is in-between

        frame_idx += 1
    
    if len(frames) > max_interesting_frames:
        np.random.shuffle(frames)
        frames = frames[:max_interesting_frames]
        frames = sorted(frames)

    return frames, np.array(original_size)

def extract_frames(temp_output_dir, cap, frame_tuples, output_size):
    for sub_path in os.listdir(temp_output_dir):
        os.remove(f"{temp_output_dir}/{sub_path}")

    if not os.path.exists(temp_output_dir):
        os.mkdir(temp_output_dir)

    for i, (frame_idx, homography) in enumerate(frame_tuples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logging.info("Failed to retrieve frame.")
            return []
        frame = apply_homography(frame, homography, output_size)
        cv2.imwrite(f"{temp_output_dir}/{i}.jpeg", frame)

def process_video(config: BaseConfig, 
                  video_path, output_dir, video_tracker, diff_threshold, skip_frames, track_skip, max_interesting_frames, iou_thresh, large_size):
    image_tracker = video_tracker.image_tracker
    stabilizer = image_tracker.stabilizer
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.info(f"Error: Could not open video file {video_path}")
        return

    begin = time.time()
    interesting_frames, original_size = get_interesting_frames(cap, skip_frames, image_tracker.target_size, diff_threshold, max_interesting_frames)
    if np.any(original_size != large_size):
        raise ValueError(f"The image size {original_size} is not the expected large size {large_size}.")
    end = time.time()
    logging.info(f"Found {len(interesting_frames)} interesting frames. {end - begin:.4f}s")

    # TODO: Make sure the aspect ratio doesn't change.

    begin = time.time()
    homographies = stabilizer.stabilize_frames(cap, interesting_frames)
    end = time.time()
    logging.info(f"Stabilized frames. {end - begin:.4f}s")

    tracks = []
    obj_id_start = 0

    def _get_obj_ids():
        nonlocal obj_id_start
        ids = list(range(obj_id_start, obj_id_start + len(interesting_frames)))
        return ids

    num_track_frames = len(interesting_frames) // track_skip
    track_frame_idx = 0
    frame_hom_pairs = list(zip(interesting_frames, homographies))
    np.random.shuffle(frame_hom_pairs)
    for frame_idx, homography in frame_hom_pairs:
        if track_frame_idx >= num_track_frames: break

        begin = time.time()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logging.info("Failed to read frame.")
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        masks = image_tracker.process_image(frame, homography)
        if len(masks) == 0:
            logging.info("Failed to find socks. Skipping the frame.")
            continue

        frame_indices = list(reversed([(i, H) for (i, H) in zip(interesting_frames, homographies) if i <= frame_idx]))
        extract_frames(config.temp_fs_dir, cap, frame_indices, image_tracker.target_size)
        track_bw = video_tracker.track_forward(config.temp_fs_dir, frame_indices, masks, _get_obj_ids())

        frame_indices = [(i, H) for (i, H) in zip(interesting_frames, homographies) if i >= frame_idx]
        extract_frames(config.temp_fs_dir, cap, frame_indices, image_tracker.target_size)
        track_fw = video_tracker.track_forward(config.temp_fs_dir, frame_indices, masks, _get_obj_ids())

        track = [*reversed(track_bw[1:]), *track_fw]
        tracks.append(track)
        obj_id_start += len(interesting_frames)

        end = time.time()
        track_frame_idx += 1
        logging.info(f"Tracked frame {track_frame_idx}/{num_track_frames}. {end - begin:.4f}s")


    cap.release()

    begin = time.time()
    master_track, num_objs = video_tracker.merge_into_master_track(len(interesting_frames), 4, obj_id_start, iou_thresh, tracks)
    end = time.time()
    logging.info(f"Merged and deduplicated tracks. {end - begin:.4f}s")

    begin = time.time()
    video_tracker.mark_occlusions(master_track)
    end = time.time()
    logging.info(f". {end - begin:.4f}s")

    begin = time.time()
    video_tracker.export_master_track(interesting_frames, homographies, master_track, num_objs, output_dir, video_name)
    end = time.time()
    logging.info(f"Exported master track. {end - begin:.4f}s")


def _setup_sam2_model(sam2_checkpoint: str, sam2_config: str, device):
    from sam2.build_sam import build_sam2
    from sam2.build_sam import build_sam2_video_predictor
    import requests

    if not os.path.exists(sam2_checkpoint):
        print(f"Downloading {sam2_checkpoint}...")
        url = f"https://dl.fbaipublicfiles.com/segment_anything_2/{sam2_checkpoint}"
        with open(sam2_checkpoint, "wb") as f:
            f.write(requests.get(url).content)
    
    if not os.path.exists(sam2_config):
        print(f"Downloading {sam2_config}...")
        url = f"https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/configs/sam2.1/{sam2_config}"
        with open(sam2_config, "wb") as f:
            f.write(requests.get(url).content)

    sam2 = build_sam2("/" + os.path.abspath(sam2_config), sam2_checkpoint, device=device, apply_postprocessing=False)
    sam2_video = build_sam2_video_predictor("/" + os.path.abspath(sam2_config), sam2_checkpoint, vos_optimized=False) # TODO: Bug. vos_optimized=True seems to not work for now.
    return sam2, sam2_video


def label_automatically(
    classifier,
    classifier_transform,
    config: BaseConfig
):
    if not os.path.exists(config.input_dir):
        os.makedirs(config.input_dir)
        logging.info(f"Created directory {config.input_dir} - please add your videos there and run again")
        return
    os.makedirs(config.output_dir, exist_ok=True)

    image_shape_gcd = np.gcd(config.image_size[0], config.image_size[1])
    target_shape_gcd = np.gcd(config.output_warped_size[0], config.output_warped_size[1])
    norm_image_shape = np.array(config.image_size, np.uint32) / image_shape_gcd
    norm_target_shape = np.array(config.output_warped_size, np.uint32) / target_shape_gcd

    if np.any(norm_image_shape != norm_target_shape):
        raise ValueError("The image shape does not have the same aspect ratio as the target shape.")


    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam2, sam2_video = _setup_sam2_model(config.sam2_checkpoint, config.sam2_config, device)

    stabilizer = Stabilizer(config)
    image_tracker = ImageTracker(config, stabilizer, sam2, classifier, classifier_transform)
    video_tracker = VideoTracker(config, image_tracker, sam2_video)

    video_files = sorted(
        [f for f in os.listdir(config.input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    )

    # video_files = video_files[:2] # TODO: revert

    if not video_files:
        logging.info(f"No videos found in {config.input_dir}")
        return

    for video_file in video_files:
        video_path = os.path.join(config.input_dir, video_file)
        logging.info(f"Processing video: {video_file}")
        process_video(
            config,
            video_path, config.output_dir, video_tracker,
            config.diff_threshold, config.skip_frames, config.track_skip, config.max_interesting_frames, config.iou_thresh,
            config.image_size
        )
    logging.info("All operations completed successfully!")
