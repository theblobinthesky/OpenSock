import os, json
os.environ['TQDM_DISABLE'] = '1'
import numpy as np
import logging
import torch
from .config import BaseConfig
from .timing import timed
from .video_capture import VideoContext
from .calibration import calibrate_and_save
from .preprocessor import preprocess
from .singleframe_processor import process_single_frames
from .multiframe_processor import process_multiple_frames


def export_master_track(important_frames: list[int], instances: list, homographies: list[np.ndarray], master_track: list, path: str):
    important_frames = []
    for frame_idx, frame, homography in zip(important_frames, master_track, homographies):
        frame_data = {}
        for track_id, data in frame.items():
            frame_data[str(track_id)] = {
                "segmentation": [data["segmentation"]],
                "bbox": data["bbox"],
                "area": data["area"],
                "is_occluded": data["is_occluded"]
            }
        important_frames.append({
            'index': frame_idx,
            'data': frame_data,
            'stabilizer_homography': list(homography.flatten())
        })

    export_data = {
        'instances': instances,
        'important_frames': important_frames,
        'variant_frames': None
    }
    
    with open(path, 'w') as f:
        json.dump(export_data, f, indent=2)
        
    logging.info(f"Master track exported to {path}.")


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
    sam2_video = build_sam2_video_predictor("/" + os.path.abspath(config.sam2_config), config.sam2_checkpoint, vos_optimized=True, points_per_batch=64)
    return sam2, sam2_video


@timed
def process_video(config: BaseConfig, video_ctx: VideoContext, video_path, sam2, sam2_video, classifier):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(config.output_dir, exist_ok=True)

    # interesting_frames, track_frames, homographies = preprocess(video_ctx, config)
    # masks_per_frame = process_single_frames(video_ctx, track_frames, config, sam2, classifier)

    # import pickle
    # with open("../data/temp.pickle", "wb") as file:
    #     pickle.dump([interesting_frames, track_frames, masks_per_frame, homographies, video_ctx], file)

    import pickle
    with open("../data/temp.pickle", "rb") as file:
        [interesting_frames, track_frames, masks_per_frame, homographies, video_ctx] = pickle.load(file)

    instances, master_track = process_multiple_frames(interesting_frames, track_frames, masks_per_frame, video_ctx, config, sam2_video)
    export_master_track(interesting_frames, instances, homographies, master_track, f"{config.output_dir}/{video_name}.json")


def label_automatically(classifier, config: BaseConfig):
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

    sam2, sam2_video = _setup_sam2_model(config, torch.device("cuda"))
    video_ctxs = sorted(os.listdir(config.input_dir))

    for video_ctx in video_ctxs:
        output_ctx_dir = f"{config.output_dir}/{video_ctx}"
        if not os.path.exists(output_ctx_dir):
            os.makedirs(output_ctx_dir)

        # calibrate_and_save(f"{config.input_dir}/{video_ctx}/calibration", output_ctx_dir)

        with open(f"{output_ctx_dir}/calib.json", "rb") as file:
            calib_config = json.load(file)

        video_files = sorted(os.listdir(f"{config.input_dir}/{video_ctx}/videos"))
        for video_file in video_files:
            video_path = f"{config.input_dir}/{video_ctx}/videos/{video_file}"
            video_ctx = VideoContext(video_path, config)
            video_ctx.calib_config = calib_config
            process_video(config, video_ctx, video_path, sam2, sam2_video, classifier)

    logging.info("All operations completed successfully!")
