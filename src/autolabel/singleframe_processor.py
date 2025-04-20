from .config import BaseConfig
from .timing import timed
from .video_capture import VideoContext, open_video_capture
from .colormap import calculate_luts
import numpy as np, cv2 
from typing import List, Dict
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def _generate_masks(image: np.ndarray, config: BaseConfig, sam2) -> List[Dict]:
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2, 
        points_per_side=config.automatic_mask_points_per_side,
        pred_iou_thresh=config.automatic_mask_quality_thresh
    )

    masks = mask_generator.generate(image)
    return sorted(masks, key=lambda x: x['area'], reverse=True)

def _filter_masks(image: np.ndarray, masks: List[Dict]) -> List[Dict]:
    sock_masks = []
    image_height, image_width = image.shape[:2]
    min_area = 0.005 * image_height * image_width
    max_area = 0.1 * image_height * image_width
    
    for mask in masks:
        m = mask['segmentation']
        area = mask['area']
        
        if min_area <= area <= max_area:
            mask_binary = m.astype(np.uint8)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                
                if hull_area > 0:
                    sock_masks.append(mask)
    
    return sock_masks

def _classify_instances(image: np.ndarray, masks: List[Dict], config: BaseConfig, classifier) -> List[Dict]:
    class_instances = []
    
    for mask in masks:
        # Extract the masked region
        mask_binary = mask['segmentation'].astype(np.uint8)
        y_indices, x_indices = np.where(mask_binary > 0)
        
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue
            
        # Extract and pad the bounding box
        padding = 10
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        min_y, min_x = max(0, min_y - padding), max(0, min_x - padding)
        max_y, max_x = min(image.shape[0], max_y + padding), min(image.shape[1], max_x + padding)
        
        image_roi = image[min_y:max_y, min_x:max_x].copy()
        class_probability = classifier(image_roi, config.imagenet_class)
        mask['class_confidence'] = float(class_probability)
        mask['is_class_instance'] = bool(class_probability >= config.classifier_confidence_threshold)
        class_instances.append(mask)

    return class_instances


@timed
def process_single_frames(video_ctx: VideoContext, frames: list[int], config: BaseConfig, sam2, classifier):
    masks_per_frame = []

    for _, frame in open_video_capture(video_ctx, frames, load_in_8bit_mode=True):
        masks = _generate_masks(frame, config, sam2)
        sock_masks = _filter_masks(frame, masks)
        sock_masks = _classify_instances(frame, sock_masks, config, classifier)
        masks_per_frame.append(sock_masks)

    opt_frame = np.argmax([len(masks) for masks in masks_per_frame])
    cap = open_video_capture(video_ctx, [frames[opt_frame]])
    if video_ctx.luts is None:
        masks = masks_per_frame[opt_frame]
        masks = [mask['segmentation'] for mask in masks]
        video_ctx.luts = calculate_luts(next(cap)[1], masks, const_band_size=config.lut_band_size)

    return masks_per_frame
