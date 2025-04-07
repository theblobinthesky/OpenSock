from dataclasses import dataclass
import cv2

@dataclass
class BaseConfig:
    imagenet_class: int
    input_dir: str
    output_dir: str
    image_size: tuple
    output_warped_size: tuple
    aruco_dict_type: int = cv2.aruco.DICT_6X6_250
    aruco_marker_id: int = 5
    secondary_aruco_marker_id: int = 10
    marker_size_mm: float = 80.0

    sam2_checkpoint_url: str = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    sam2_checkpoint: str = "../data/sam2.1_hiera_large.pt"
    sam2_config_url: str = "https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/sam2/configs/sam2/sam2_hiera_l.yaml"
    sam2_config: str = "../data/sam2.1_hiera_l.yaml"

    diff_threshold: float = 0.05
    track_skip: int = 38
    max_interesting_frames: int = 180
    instance_merge_iou_thresh: float = 0.9
    no_change_threshold: float = 0.01
    consecutive_static_required: int = 10
    performance_downscale_factor: int = 4

    classifier_confidence_threshold: float = 0.04
    mask_contrast_control = 1.3     # contrast control (1.0-3.0)
    mask_brightness_control = 10    # brightness control (0-100)
    video_tracker_inside_threshold = 1.0 # 0.99
    temp_fs_dir = "../data/temp_fs"
