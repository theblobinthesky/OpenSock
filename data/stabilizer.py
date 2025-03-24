import numpy as np
import cv2
from typing import Tuple, Optional

class Stabilizer:
    def __init__(
        self,
        aruco_dict_type: int,
        aruco_marker_id: int,
        output_warped_size: Tuple[int, int],
        marker_size_mm: float
    ):
        self.output_warped_size = output_warped_size
        self.marker_size_mm = marker_size_mm
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.aruco_marker_id = aruco_marker_id

    def detect_aruco_marker(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        # Convert to grayscale for marker detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect markers
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            # Find our specific marker ID
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == self.aruco_marker_id:
                    return True, corners[i][0]
        
        return False, None
    
    def compute_floor_plane_transform(
        self, 
        image: np.ndarray, 
        marker_corners: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        w, h = self.output_warped_size
        center_x, center_y = w // 8, h // 8
        
        # Calculate the vectors between opposite corners to find marker orientation
        vec1 = marker_corners[2] - marker_corners[0]  # Diagonal from top-left to bottom-right
        vec2 = marker_corners[3] - marker_corners[1]  # Diagonal from top-right to bottom-left
        
        # Estimate marker size in pixels (average of two diagonal lengths)
        diag1_length = np.linalg.norm(vec1)
        diag2_length = np.linalg.norm(vec2)
        marker_size_px = (diag1_length + diag2_length) / (2 * np.sqrt(2))  # Divide by sqrt(2) to get side length
        
        # Calculate mm per pixel ratio
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
        H = cv2.getPerspectiveTransform(marker_corners.astype(np.float32), dst_points)
        
        # Apply the homography to get the warped image
        warped = cv2.warpPerspective(image, H, (w, h))
        
        return H, warped, mm_per_pixel
        
    def stabilize_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, Optional[np.ndarray], Optional[float]]:
        # Detect ArUco marker
        marker_found, marker_corners = self.detect_aruco_marker(frame)
        
        if marker_found:
            # Compute floor plane transform that preserves scale
            transform_matrix, warped_image, mm_per_pixel = self.compute_floor_plane_transform(frame, marker_corners)
            if warped_image is not None:
                return warped_image, True, transform_matrix, mm_per_pixel
                
        # Return the original frame if no stabilization was done
        return frame, False, None, None

    def get_interesting_frames(self, cap, skip_frames: int, output_size: Tuple[int, int],
                            diff_threshold: float, no_change_threshold: float = 0.01,
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

        return frames
