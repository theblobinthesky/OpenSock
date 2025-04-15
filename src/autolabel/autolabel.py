import os
os.environ['TQDM_DISABLE'] = '1'
import cv2
import numpy as np
import logging
import time
from .config import BaseConfig
from .trackers import Stabilizer, ImageTracker, VideoTracker, apply_homography


def get_interesting_frames(cap, config: BaseConfig) -> list[int]:
    def _process(frame: np.ndarray):
        work_size = tuple(np.array(config.image_size) // config.performance_downscale_factor)
        frame = cv2.resize(frame, work_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to retrieve frame.")
        return [], None, None
    
    original_size = prev_frame.shape[:-1]
    prev_frame = _process(prev_frame)
    prev_proc_frame = prev_frame

    frame_idx = 0 
    static_count = 0
    frames = []
    all_errors = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        size = frame.shape[:-1]
        if original_size != size:
            raise ValueError("The frame sizes have to be consistent.")

        gray = _process(frame)

        diff = cv2.absdiff(prev_frame, gray)
        diff_norm = np.sum(diff) / (diff.shape[0] * diff.shape[1] * 255)
        prev_frame = gray
        all_errors.append(diff_norm)

        proc_diff = cv2.absdiff(prev_proc_frame, gray)
        proc_diff_norm = np.sum(proc_diff) / (proc_diff.shape[0] * proc_diff.shape[1] * 255)

        if proc_diff_norm > config.diff_threshold:
            frames.append(frame_idx)
            prev_proc_frame = gray
            static_count = 0
        elif proc_diff_norm < config.no_change_threshold:
            static_count += 1
            if static_count >= config.consecutive_static_required:
                frames.append(frame_idx)
                static_count = 0
        else:
            static_count = 0

        frame_idx += 1

    if total_num_frames <= 1:
        raise ValueError("Video has too few frames.")

    all_errors = [all_errors[0]] + all_errors + [all_errors[-1]]
    bidir_errors = [all_errors[i] + all_errors[i + 1] for i in range(total_num_frames)]


    # import pickle 
    # with open('test.pickle', 'wb') as file:
    #     pickle.dump([frames, bidir_errors, total_num_frames], file)

    # import pickle 
    # with open('test.pickle', 'rb') as file:
    #     [frames, bidir_errors, total_num_frames] = pickle.load(file)

    # Compute the best partitioning into config.num_track_frames partitions
    # such that the sum over the minimums is minimal using dynamic programming.

    # Let RMC(i, k) be a O(1) way to get the minimum over the last k of the first i items.
    # Let i be the number of items and j be the number of partitions. The recurrence is as follows:
    # min_way_to_partition_i_items_into_j_partitions = d(j, i) = min_(1 <= k < i){d(j - 1, k) + RMC(i, i - k)}
    # Backtrack through previous_matrix to find what the partitioning is. Simple enough?

    # Calculate RMC (rank minimum cache).
    RMC = np.zeros((total_num_frames, total_num_frames))
    center_window_width = int(round(config.center_window_width_perc * total_num_frames))
    for i in range(total_num_frames):
        last_err = float('inf')
        window_size = min(i + 1, center_window_width)
        for k in range(window_size):
            last_err = min(last_err, bidir_errors[i - k])
            RMC[i, k] = last_err


    # Initialize dynamic matrix.
    dynamic_matrix = np.full((config.num_track_frames, total_num_frames), np.inf)
    last_error = float('inf')
    for i in range(total_num_frames):
        last_error = dynamic_matrix[0, i] = min(last_error, bidir_errors[i])

    # Calculate dynamic matrix.
    previous_matrix = np.zeros((config.num_track_frames, total_num_frames), np.uint32)
    min_partition_size = int(round(config.min_partition_size_perc * total_num_frames))

    for j in range(1, config.num_track_frames):
        for i in range(max(j, 2 * min_partition_size), total_num_frames):
            min_total = float('inf')
            min_idx = -1
            for k in range(min_partition_size - 1, i + 1 - min_partition_size):
                total = dynamic_matrix[j - 1, k] + RMC[i, i - k - 1]
                if total < min_total:
                    min_total = total
                    min_idx = k

            if min_idx >= 0: 
                dynamic_matrix[j, i] = min_total
                previous_matrix[j, i] = min_idx

    # Backtrack to find the solution.
    j = config.num_track_frames - 1
    i = total_num_frames - 1
    partitions = []

    while j >= 0:
        k = int(previous_matrix[j, i])
        partitions.append((k + 1, i + 1))
        j, i = j - 1, k
    
    partitions.reverse()

    # min_value = dynamic_matrix[config.num_track_frames - 1, total_num_frames - 1]

    # Find the track frames.
    track_frames = []
    for (begin, end) in partitions:
        idx = begin + int(np.argmin(bidir_errors[begin:end]))
        track_frames.append(idx)

    interesting_frames = list(set(frames) | set(track_frames))
    interesting_frames.sort()

    track_frame_indices = [interesting_frames.index(f) for f in track_frames]

    return interesting_frames, track_frame_indices, np.array(original_size)


# config = BaseConfig(
#     imagenet_class = 806,
#     input_dir = "../data/sock_videos",
#     output_dir = "../data/sock_video_master_tracks",
#     image_size = (1080, 1920),
#     output_warped_size = (540, 960),   
#     classifier_confidence_threshold = 0.01
# )
# get_interesting_frames(None, config)

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

def process_video(config: BaseConfig, video_path, video_tracker):
    image_tracker = video_tracker.image_tracker
    stabilizer = image_tracker.stabilizer
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(config.output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.info(f"Error: Could not open video file {video_path}")
        return

    begin = time.time()
    interesting_frames, track_frame_indices, original_size = get_interesting_frames(cap, config)
    if np.any(original_size != config.image_size):
        raise ValueError(f"The image size {original_size} is not the expected large size {config.image_size}.")
    end = time.time()
    logging.info(f"Found {len(interesting_frames)} interesting frames and {len(track_frame_indices)} track frames. {end - begin:.4f}s")

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(len(track_frame_indices), 1)
    # for i, track_frame_idx in enumerate(track_frame_indices):
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, interesting_frames[track_frame_idx])
    #     ret, frame = cap.read()
    #     if not ret:
    #         logging.info("Failed to read frame.")
    #         return

    #     axs[i].imshow(frame)
    # plt.show()

    begin = time.time()
    homographies = stabilizer.stabilize_frames(cap, interesting_frames)
    end = time.time()
    logging.info(f"Stabilized frames. {end - begin:.4f}s")

    instances = []
    tracks = []
    obj_id_start = 0

    def _get_obj_ids(masks):
        nonlocal obj_id_start
        ids = list(range(obj_id_start, obj_id_start + len(masks)))
        return ids

    track_frame_hom_pairs = [(interesting_frames[track_frame_idx], homographies[track_frame_idx]) for track_frame_idx in track_frame_indices]
    for frame_idx, homography in track_frame_hom_pairs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logging.info("Failed to read frame.")
            return

        begin = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        masks = image_tracker.process_image(frame, homography)
        if len(masks) == 0:
            logging.info("Failed to find socks. Skipping the frame.")
            continue

        instances.extend(masks)
        end = time.time()
        logging.info(f"Found {len(masks)} class instances. {end - begin:.4f}s")

        begin = time.time()
        frame_indices = list(reversed([(i, H) for (i, H) in zip(interesting_frames, homographies) if i <= frame_idx]))
        extract_frames(config.temp_fs_dir, cap, frame_indices, image_tracker.target_size)
        track_bw = video_tracker.track_forward(config.temp_fs_dir, frame_indices, masks, _get_obj_ids(masks))

        frame_indices = [(i, H) for (i, H) in zip(interesting_frames, homographies) if i >= frame_idx]
        extract_frames(config.temp_fs_dir, cap, frame_indices, image_tracker.target_size)
        track_fw = video_tracker.track_forward(config.temp_fs_dir, frame_indices, masks, _get_obj_ids(masks))

        track = [*reversed(track_bw[1:]), *track_fw]
        tracks.append(track)
        obj_id_start += len(masks)

        end = time.time()
        logging.info(f"Tracked all frames. {end - begin:.4f}s")


        track_frame_idx += 1
        logging.info(f"Tracked frame {track_frame_idx}/{len(track_frame_indices)}.")


    cap.release()

    instances = [
        {
            'class_confidence': instance['class_confidence'],
            'is_class_instance': bool(instance['is_class_instance'])
        }
        for instance in instances
    ]


    # import pickle
    # with open("metadata.pickle", "wb") as f:
    #     pickle.dump([instances, tracks, interesting_frames, homographies, video_name], f)

    # import pickle
    # with open("metadata.pickle", "rb") as f:
    #     [instances, tracks, interesting_frames, homographies, video_name] = pickle.load(f)


    begin = time.time()
    instances, master_track = video_tracker.merge_into_master_track(len(interesting_frames), 4, instances, tracks)
    end = time.time()
    logging.info(f"Merged and deduplicated tracks. {end - begin:.4f}s")

    video_tracker.mark_occlusions(master_track)

    begin = time.time()
    video_tracker.export_master_track(interesting_frames, instances, homographies, master_track, config.output_dir, video_name)
    end = time.time()
    logging.info(f"Exported master track. {end - begin:.4f}s")


def _setup_sam2_model(config: BaseConfig, device):
    from sam2.build_sam import build_sam2
    from sam2.build_sam import build_sam2_video_predictor
    import requests

    if not os.path.exists(config.sam2_checkpoint):
        with open(config.sam2_checkpoint, "wb") as f:
            f.write(requests.get(config.sam2_checkpoint_url).content)
        logging.info(f"Downloaded {config.sam2_checkpoint}.")
    
    if not os.path.exists(config.sam2_config):
        with open(config.sam2_config, "wb") as f:
            f.write(requests.get(config.sam2_config_url).content)
        logging.info(f"Downloaded {config.sam2_config}.")

    sam2 = build_sam2("/" + os.path.abspath(config.sam2_config), config.sam2_checkpoint, device=device, apply_postprocessing=False)
    sam2_video = build_sam2_video_predictor("/" + os.path.abspath(config.sam2_config), config.sam2_checkpoint, vos_optimized=True)
    return sam2, sam2_video


def label_automatically(
    classifier,
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
    sam2, sam2_video = _setup_sam2_model(config, device)

    stabilizer = Stabilizer(config)
    image_tracker = ImageTracker(config, stabilizer, sam2, classifier)
    video_tracker = VideoTracker(config, image_tracker, sam2_video)

    video_files = sorted(
        [f for f in os.listdir(config.input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    )

    # video_files = video_files[:1] # TODO: revert

    if not video_files:
        logging.info(f"No videos found in {config.input_dir}")
        return

    for video_file in video_files:
        video_path = os.path.join(config.input_dir, video_file)
        logging.info(f"Processing video: {video_file}")
        begin = time.time()
        process_video(config, video_path, video_tracker)
        end = time.time()
        logging.info(f"Processed video: {video_file}. {end - begin}s")


    logging.info("All operations completed successfully!")
