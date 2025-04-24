from .config import BaseConfig
from .timing import timed
from .video_capture import VideoContext, open_video_capture
import numpy as np, cv2
from typing import Tuple, Optional
from tqdm import tqdm


def get_interesting_frames(video_ctx: VideoContext, config: BaseConfig) -> list[int]:
    def _process(frame: np.ndarray):
        work_size = tuple(np.array(config.image_size) // config.performance_downscale_factor)
        frame = cv2.resize(frame, work_size, interpolation=cv2.INTER_AREA)
        return frame

    cap = open_video_capture(video_ctx, load_in_8bit_mode=True)
    total_num_frames = len(cap)
    _, prev_frame = next(cap)
    original_size = prev_frame.shape[:-1]
    prev_frame = _process(prev_frame)
    prev_proc_frame = prev_frame

    frame_idx = 0 
    frames = []
    all_errors = []

    for idx, frame in tqdm(cap):
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

        frame_idx += 1

    if total_num_frames <= 1:
        raise ValueError("Video has too few frames.")

    all_errors = [all_errors[0]] + all_errors + [all_errors[-1]]
    bidir_errors = [all_errors[i] + all_errors[i + 1] for i in range(total_num_frames)]


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


    # Find the track frames.
    track_frames = []
    for (begin, end) in partitions:
        idx = begin + int(np.argmin(bidir_errors[begin:end]))
        track_frames.append(idx)

    interesting_frames = list(set(frames) | set(track_frames))
    interesting_frames.sort()

    track_frame_indices = [interesting_frames.index(f) for f in track_frames]

    return interesting_frames, track_frame_indices, np.array(original_size)


class StabilizerContext:
    def __init__(self, config: BaseConfig):
        self.output_warped_size = config.output_warped_size
        self.marker_size_mm = config.marker_size_mm
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(config.aruco_dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.aruco_marker_id = config.aruco_marker_id
        self.secondary_aruco_marker_id = config.secondary_aruco_marker_id

    def detect_aruco_marker(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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


@timed
def stabilize_frames(interesting_frames: list[int], config: BaseConfig, video_ctx: VideoContext):
    ctx = StabilizerContext(config)
    primary_homographies = {}
    primary_corners_dict = {}
    secondary_midpoints = {}

    for frame_idx, frame in tqdm(open_video_capture(video_ctx, interesting_frames, load_in_8bit_mode=True)):
        primary_corners, secondary_corners = ctx.detect_aruco_marker(frame)

        if primary_corners is not None:
            H = ctx.compute_floor_plane_transform(primary_corners)
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

        H = ctx.compute_floor_plane_transform(primary_corners, secondary_midpoint, transformed_avg_midpoint)
        corrected_homographies[frame_idx] = H

    # This quick fix assumes the camera is stationary.
    avg_homography = np.zeros((3, 3))
    for H in corrected_homographies.values():
        avg_homography += H
    avg_homography /= len(corrected_homographies)

    # [corrected_homographies[idx] for idx in sorted(corrected_homographies.keys())]
    return [avg_homography for _ in corrected_homographies.keys()]


@timed
def preprocess(video_ctx: VideoContext, config: BaseConfig):
    interesting_frames, track_frame_indices, original_size = get_interesting_frames(video_ctx, config)
    if np.any(original_size != config.image_size):
        raise ValueError(f"The image size {original_size} is not the expected large size {config.image_size}.")
    track_frames = [interesting_frames[track_frame_idx] for track_frame_idx in track_frame_indices]

    homographies = stabilize_frames(interesting_frames, config, video_ctx)
    video_ctx.homographies = {frame_idx: H for frame_idx, H in zip(track_frame_indices, homographies)}

    return interesting_frames, track_frames, homographies
