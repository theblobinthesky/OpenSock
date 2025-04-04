import numpy as np
import torch
import cv2
import json
from typing import Dict, List, Tuple, Optional
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from config import BaseConfig
import logging


class Stabilizer:
    def __init__(
        self,
        aruco_dict_type: int,
        aruco_marker_id: int,
        secondary_aruco_marker_id: int,
        output_warped_size: Tuple[int, int],
        marker_size_mm: float
    ):
        self.output_warped_size = output_warped_size
        self.marker_size_mm = marker_size_mm
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.aruco_marker_id = aruco_marker_id
        self.secondary_aruco_marker_id = secondary_aruco_marker_id

    def detect_aruco_marker(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        # Convert to grayscale for marker detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect markers
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            marker_corners = None
            secondary_marker_corners = None

            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == self.aruco_marker_id:
                    marker_corners = corners[i][0]
                elif marker_id == self.secondary_aruco_marker_id:
                    secondary_marker_corners = corners[i][0]

            return marker_corners, secondary_marker_corners
        
        return None, None
    
    def compute_floor_plane_transform(self, marker_corners: np.ndarray, from_pt: np.ndarray=None, to_pt: np.ndarray=None) -> np.ndarray:
        w, h = self.output_warped_size
        center_x, center_y = w // 8, h // 8
        
        # Calculate the vectors between opposite corners to find marker orientation
        vec1 = marker_corners[2] - marker_corners[0]  # Diagonal from top-left to bottom-right
        vec2 = marker_corners[3] - marker_corners[1]  # Diagonal from top-right to bottom-left
        
        # Estimate marker size in pixels (average of two diagonal lengths)
        diag1_length = np.linalg.norm(vec1)
        diag2_length = np.linalg.norm(vec2)
        marker_size_px = (diag1_length + diag2_length) / (2 * np.sqrt(2))  # Divide by sqrt(2) to get side length
        mm_per_pixel = self.marker_size_mm / marker_size_px
        
        # We will create a scaled square in the output image
        # The size preserves the real-world scale based on the marker
        output_marker_size_px = self.marker_size_mm / mm_per_pixel
        
        # Calculate the destination points for the marker in the output image
        # Centered in the output image with proper scaling
        half_size = output_marker_size_px / 1.5
        dst_points = np.array([
            [center_x - half_size, center_y - half_size],  # Top-left
            [center_x + half_size, center_y - half_size],  # Top-right
            [center_x + half_size, center_y + half_size],  # Bottom-right
            [center_x - half_size, center_y + half_size]   # Bottom-left
        ], dtype=np.float32)
        
        # Compute homography matrix that maps marker corners to the destination square
        from_pts = marker_corners.astype(np.float32)
        to_pts = dst_points
        if from_pt is not None:
            from_pts = np.concatenate([from_pts, from_pt[None, :]], axis=0)
            to_pts = np.concatenate([to_pts, to_pt[None, :]], axis=0)
            H, _ = cv2.findHomography(from_pts, to_pts, method=0)
        else:
            H = cv2.getPerspectiveTransform(from_pts, to_pts)

        return H
        
    def stabilize_frames(self, cap, interesting_frames: list[int]):
        primary_homographies = {}
        primary_corners_dict = {}
        secondary_midpoints = {}

        for frame_idx in interesting_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print("Failed to retrieve frame.")
                return []

            primary_corners, secondary_corners = self.detect_aruco_marker(frame)

            if primary_corners is not None:
                H = self.compute_floor_plane_transform(primary_corners)
                primary_homographies[frame_idx] = H
                primary_corners_dict[frame_idx] = primary_corners

            if secondary_corners is not None:
                secondary_midpoints[frame_idx] = np.mean(secondary_corners, axis=0)

        def interpolate_all(existing_dict: dict[int, np.ndarray]):
            found_frames = sorted(existing_dict.keys())
            for curr, nxt in zip(found_frames[:-1], found_frames[1:]):
                ten_curr = existing_dict[curr]
                ten_next = existing_dict[nxt]
                for idx in interesting_frames:
                    if curr <= idx <= nxt:
                        t = float(idx - curr) / (nxt - curr)
                        existing_dict[idx] = (1.0 - t) * ten_curr + t * ten_next

            min_found_frame = np.min(np.array(found_frames))
            max_found_frame = np.max(np.array(found_frames))
            for idx in interesting_frames: 
                if idx < min_found_frame:
                    existing_dict[idx] = existing_dict[min_found_frame]
                elif idx > max_found_frame:
                    existing_dict[idx] = existing_dict[max_found_frame]
                
        
        interpolate_all(primary_homographies)
        interpolate_all(primary_corners_dict)

        if len(list(secondary_midpoints.values())) == 0:
            print("No secondary marker detected.")
            return [primary_homographies[idx] for idx in sorted(primary_homographies.keys())]

        interpolate_all(secondary_midpoints)
 
        avg_midpoint = np.mean(list(secondary_midpoints.values()), axis=0)
        avg_midpoint = np.array([avg_midpoint[0], avg_midpoint[1], 1.0])

        corrected_homographies = {}
        for frame_idx in interesting_frames:
            primary_homography = primary_homographies[frame_idx]
            primary_corners = primary_corners_dict[frame_idx]
            secondary_midpoint = secondary_midpoints[frame_idx]

            transformed_avg_midpoint = primary_homography @ avg_midpoint
            transformed_avg_midpoint /= transformed_avg_midpoint[2]
            transformed_avg_midpoint = transformed_avg_midpoint[:2]

            H = self.compute_floor_plane_transform(primary_corners, secondary_midpoint, transformed_avg_midpoint)
            corrected_homographies[frame_idx] = H

        # This quick fix assumes the camera is stationary.
        avg_homography = np.zeros((3, 3))
        for H in corrected_homographies.values():
            avg_homography += H
        avg_homography /= len(corrected_homographies)

        # [corrected_homographies[idx] for idx in sorted(corrected_homographies.keys())]
        return [avg_homography for _ in corrected_homographies.keys()]


def apply_homography(image: np.ndarray, homography: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = cv2.warpPerspective(image, homography, image.shape[:2])
    image = image.transpose((1, 0, 2))
    image = cv2.resize(image, size[::-1], interpolation=cv2.INTER_AREA)
    return image


class ImageTracker:
    def __init__(
        self, 
        target_size: Tuple[int, int],
        stabilizer: Stabilizer,
        sam2,
        imagenet_class: int,
        classifier,
        classifier_transform,
        config: BaseConfig
    ):
        self.target_size = target_size
        self.stabilizer = stabilizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = sam2
        self.imagenet_class = imagenet_class
        self.classifier = classifier
        self.transform = classifier_transform
        self.config = config
        
    def _load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _generate_masks(self, image: np.ndarray) -> List[Dict]:
        # Use SAM2AutomaticMaskGenerator for generating masks
        mask_generator = SAM2AutomaticMaskGenerator(self.sam)
        masks = mask_generator.generate(image)
        return sorted(masks, key=lambda x: x['area'], reverse=True)

    def _filter_masks(self, image: np.ndarray, masks: List[Dict]) -> List[Dict]:
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
    
    def _classify_socks(self, image: np.ndarray, masks: List[Dict]) -> List[Dict]:
        confirmed_socks = []
        
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
            mask_roi = mask_binary[min_y:max_y, min_x:max_x].copy()
            image_roi = cv2.convertScaleAbs(image_roi, alpha=self.config.mask_contrast_control, beta=self.config.mask_brightness_control)
            roi = image_roi * mask_roi[..., None]

            roi = cv2.resize(roi, (448, 448)) * 255
            roi = Image.fromarray(roi)
            input_tensor = self.transform(roi).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                output = self.classifier(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)

            # Check if it's classified as a sock with enough confidence
            if probabilities[self.imagenet_class] >= self.config.classifier_confidence_threshold:
                mask['class_confidence'] = float(probabilities[self.imagenet_class])
                confirmed_socks.append(mask)

        return confirmed_socks


    def process_image(self, image: np.ndarray, homography: np.ndarray) -> np.ndarray:
        image = apply_homography(image, homography, self.target_size)
        masks = self._generate_masks(image)
        sock_masks = self._filter_masks(image, masks)
        sock_masks = self._classify_socks(image, sock_masks)
        return sock_masks
    

class VideoTracker:
    def __init__(self, image_tracker: ImageTracker, sam2_video):
        self.image_tracker = image_tracker
        self.device = self.image_tracker.device
        self.sam = sam2_video
    
    def track_forward(self, input_dir: str, frame_indices: list[int], masks: list, obj_ids: list):
        track = list(range(len(frame_indices)))
               
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.sam.init_state(input_dir)

            for mask, obj_id in zip(masks, obj_ids):
                self.sam.add_new_mask(state, 0, obj_id, mask['segmentation'])
           
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam.propagate_in_video(state):
                track[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                } 

        return track

    def merge_into_master_track(self, num_interesting_frames: int, min_num_occ_for_merge: int, max_num_objs: int, iou_thresh: float, tracks: list):
        master_track = []
        for frame_idx in range(num_interesting_frames):
            merged = {}
            for track in tracks:
                merged.update(track[frame_idx])
            
            master_track.append(merged) 

        # Based on intersection over union metric, merge the object ids 
        # that seem to track the same object in a majority of frames.
        to_merge = {}
        for i, frame in enumerate(master_track):
            for obj_id, mask in frame.items():
                for obj_id_2, mask_2 in frame.items():
                    if obj_id >= obj_id_2: continue
                    
                    inter = (mask & mask_2).sum()
                    union = (mask | mask_2).sum()
                    
                    if union > 0:
                        iou = inter / union
                        if iou >= iou_thresh:
                            to_merge[(obj_id, obj_id_2)] = to_merge.get((obj_id, obj_id_2), 0) + 1

            maj_votes = [ids for ids, num_votes in to_merge.items() if num_votes >= min_num_occ_for_merge]
            for obj_id_1, obj_id_2 in maj_votes:
                new_obj_id = max_num_objs
                max_num_objs += 1
                del to_merge[(obj_id_1, obj_id_2)]

                for frame in master_track:

                    if obj_id_1 in frame: 
                        frame[new_obj_id] = frame[obj_id_1]
                        del frame[obj_id_1]
                        
                    if obj_id_2 in frame: 
                        frame[new_obj_id] = frame[obj_id_2]
                        del frame[obj_id_2]
                
                            
        # Remap object ids to continuous integers starting from 0.
        unique_ids = set()
        for frame in master_track:
            unique_ids.update(frame.keys())

        sorted_ids = sorted(unique_ids)
        id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}
        master_track = [{id_map[old_id]: mask for old_id, mask in frame.items()} for frame in master_track]

        # Convert to contours.
        for frame in master_track:
            for track_id, mask in list(frame.items()):
                mask_uint8 = mask.astype(np.uint8).squeeze(0)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    largest_contour = cv2.approxPolyDP(largest_contour, 1e-3 * cv2.arcLength(largest_contour, True), True)
                    segmentation = largest_contour.reshape(-1, 2).flatten().tolist()
                    x, y, w, h = cv2.boundingRect(largest_contour)
                else:
                    area = 0.0
                    segmentation = []
                    x, y, w, h = 0, 0, 0, 0


                frame[track_id] = {
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": [x, y, w, h],
                    "is_occluded": False
                }

                    
        return master_track, len(sorted_ids)

    def mark_occlusions(self, master_track: list):
        for frame in master_track:
            for track_id, data in list(frame.items()):
                if len(data["segmentation"]) == 0:
                    data["is_occluded"] = True

        return master_track

    def _get_json_file_path(self, dir: str, filename: str):
        return f"{dir}/{filename}.json"

    def export_master_track(self, important_frame_indices: list[int], homographies: list[np.ndarray], master_track: list, num_objs: int, dir: str, filename: str):
        important_frames = []
        for frame_idx, frame, homography in zip(important_frame_indices, master_track, homographies):
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
            'num_objects': num_objs,
            'important_frames': important_frames,
            'variant_frames': None
        }
        
        with open(self._get_json_file_path(dir, filename), 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logging.debug(f"Master track exported to {filename}")


    def import_master_track(self, dir: str, filename: str):
        with open(self._get_json_file_path(dir, filename), 'r') as f:
            data = json.load(f)

        logging.debug(f"Master track imported from {filename}")

        return data
