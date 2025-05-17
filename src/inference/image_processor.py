from autolabel.config import BaseConfig
from autolabel.preprocessor import StabilizerContext
from autolabel.video_capture import apply_homography
import numpy as np

def process_single_image(frame: np.ndarray) -> np.ndarray:
    config = BaseConfig(
        imagenet_class = 806,
        input_dir = "../data/data_input",
        output_dir = "../data/data_output",
        image_size = (1080, 1920),
        output_warped_size = (540, 960),   
        classifier_confidence_threshold = 0.01
    )

    ctx = StabilizerContext(config)
    # TODO: Apply calibration tooo.....
    H, trans_corners_per_frame = ctx.stabilize_frame(frame)
    frame = apply_homography(frame, H, frame.shape[:2])
    return frame, trans_corners_per_frame

