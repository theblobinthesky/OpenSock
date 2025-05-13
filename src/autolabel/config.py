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
    automatic_mask_points_per_side = 64
    automatic_mask_quality_thresh = 0.75

    diff_threshold: float = 0.1
    instance_merge_iou_thresh: float = 0.9
    no_change_threshold: float = 0.01
    consecutive_static_required: int = 10
    performance_downscale_factor: int = 4

    max_num_track_frames: int = 6
    worst_allowed_quality: float = 3.0
    error_window_radius: int = 8
    min_partition_size_perc: float = 0.15
    center_window_width_perc: float = 0.08

    classifier_confidence_threshold: float = 0.04
    lut_band_size = 32
    video_tracker_inside_threshold = 1.0 # 0.99
    temp_fs_dir = "../data/temp_fs"
