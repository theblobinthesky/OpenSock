from dataclasses import dataclass
import cv2

@dataclass
class BaseConfig:
    imagenet_class: int
    input_dir: str
    output_dir: str
    image_size: tuple
    output_warped_size: tuple
    diff_threshold: float = 0.04
    skip_frames: int = 5
    aruco_dict_type: int = cv2.aruco.DICT_6X6_250
    aruco_marker_id: int = 5
    secondary_aruco_marker_id: int = 10
    marker_size_mm: float = 80.0
    sam2_checkpoint: str = "sam2.1_hiera_large.pt"
    sam2_config: str = "sam2.1_hiera_l.yaml"
    track_skip: int = 40
    max_interesting_frames: int = 180
    iou_thresh: float = 0.9
    classifier_confidence_threshold: float = 0.01
    mask_contrast_control = 1.5     # contrast control (1.0-3.0)
    mask_brightness_control = 20    # brightness control (0-100)