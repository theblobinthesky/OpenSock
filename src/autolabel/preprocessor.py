from .config import BaseConfig
from .utils import embed_into_projective, unembed_into_euclidean, subtract_projective_into_euclidean, get_similarity_transform_matrix
from .timing import timed
from .video_capture import VideoContext, open_video_capture
import numpy as np, cv2, scipy
from typing import Tuple, Optional
from tqdm import tqdm
import logging


@timed
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


    w = config.error_window_radius
    all_errors = np.pad(all_errors, (w, w), mode='edge')
    kernel = np.ones(2 * w - 1) / (2 * w - 1)
    all_errors = np.convolve(all_errors, kernel, mode='valid')


    def get_optimal_track_frames(num_track_frames: int):
        # Compute the best partitioning into num_track_frames partitions
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
                last_err = min(last_err, all_errors[i - k])
                RMC[i, k] = last_err


        # Initialize dynamic matrix.
        dynamic_matrix = np.full((num_track_frames, total_num_frames), np.inf)
        last_error = float('inf')
        for i in range(total_num_frames):
            last_error = dynamic_matrix[0, i] = min(last_error, all_errors[i])

        # Calculate dynamic matrix.
        previous_matrix = np.zeros((num_track_frames, total_num_frames), np.uint32)
        min_partition_size = int(round(config.min_partition_size_perc * total_num_frames))

        for j in range(1, num_track_frames):
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
        j = num_track_frames - 1
        i = total_num_frames - 1
        partitions = []

        while j >= 0:
            k = int(previous_matrix[j, i])
            partitions.append((k + 1, i + 1))
            j, i = j - 1, k
        
        partitions.reverse()


        # Find the track frames.
        track_frames = []
        track_frame_errors = []
        for (begin, end) in partitions:
            idx = begin + int(np.argmin(all_errors[begin:end]))
            track_frames.append(idx)
            track_frame_errors.append(all_errors[idx])

        interesting_frames = list(set(frames) | set(track_frames))
        interesting_frames.sort()

        quality_factor = np.max(track_frame_errors) / np.mean(track_frame_errors)

        return interesting_frames, track_frames, quality_factor

    interesting_frames = None
    track_frames = None
    quality_factor = None
    for num_track_frames in range(1, config.max_num_track_frames + 1):
        _interesting_frames, _track_frames, _quality_factor = get_optimal_track_frames(num_track_frames)

        if _quality_factor <= config.worst_allowed_quality:
            interesting_frames = _interesting_frames
            track_frames = _track_frames
            quality_factor = _quality_factor

    logging.info(f"Found {len(track_frames)} track frames with quality {quality_factor:.2f} (<= {config.worst_allowed_quality:.2f}).")

    return interesting_frames, track_frames, np.array(original_size)


def get_null_space_vector(M: np.ndarray) -> np.ndarray:
    _, _, Vh = np.linalg.svd(M)
    return Vh[-1]

class StabilizerContext:
    def __init__(self, config: BaseConfig):
        self.output_warped_size = config.output_warped_size
        self.marker_size_mm = config.marker_size_mm
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(config.aruco_dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.config = config

    def detect_rectangles(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        list_of_corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        list_of_corners = np.array(list_of_corners).squeeze(axis=1)
        corners_by_id = {id: corner for id, corner in zip(ids.flatten(), list_of_corners)}
        return corners_by_id

    def compute_orthographic_birds_eye_view(self, frame, corners_by_id) -> np.ndarray:
        # Normalize corner coordinates for better conditioning.
        all_pts = np.vstack(list(corners_by_id.values()))  # shape (4*M, 2)
        mean_xy = all_pts.mean(axis=0)
        std_xy  = all_pts.std(axis=0)
        s = np.mean(std_xy)
        T_norm = np.array([
            [1.0/s,    0,     -mean_xy[0]/s],
            [   0,   1.0/s,   -mean_xy[1]/s],
            [   0,      0,               1]
        ], dtype=float)

        # Apply T_norm to each corner set
        norm_corners = {}
        for mid, corners in corners_by_id.items():
            h = embed_into_projective(corners)
            h = (T_norm @ h.T).T
            norm_corners[mid] = (h[:, :2] / h[:, 2:3])


        # Compute linear initialization for 2d rectification.
        def get_constraint(l: np.ndarray, m: np.ndarray) -> np.ndarray:
            return np.array([
                l[0] * m[0], 
                (l[0] * m[1] + l[1] * m[0]) * 0.5, 
                l[1] * m[1],
                (l[0] * m[2] + l[2] * m[0]) * 0.5, 
                (l[1] * m[2] + l[2] * m[1]) * 0.5,
                l[2] * m[2]
            ])

        constraints = []
        for marker_corners in corners_by_id.values():
            marker_corners = embed_into_projective(marker_corners)
            l0 = np.cross(marker_corners[0], marker_corners[1])
            m0 = np.cross(marker_corners[1], marker_corners[2])
            l1 = np.cross(marker_corners[0], marker_corners[2])
            m1 = np.cross(marker_corners[1], marker_corners[3])

            # It's unclear to me how many constraints i should add.
            # Technically only one since parallel lines don't add any new information.
            # But they are not exactly parallel in my noisy measurements. So idk.
            constraints.append(get_constraint(l0, m0))
            constraints.append(get_constraint(l1, m1))


        C = np.stack(constraints, axis=0)
        v = get_null_space_vector(C)
        v *= np.sign(v[0])
        [a, b, c, d, e, f] = v
        transformed_abs_conic = np.array([
            [a, b / 2, d / 2],
            [b / 2, c, e / 2],
            [d / 2, e / 2, f]
        ])

        K = np.linalg.cholesky(transformed_abs_conic[:2, :2])
        detK = np.linalg.det(K)
        K /= np.sqrt(detK)
        v = scipy.linalg.cho_solve((K, False), transformed_abs_conic[:2, 2]) * detK
        scale = transformed_abs_conic[2, 2] / (v.T @ K @ K.T @ v / detK)

        # TODO: Remove normalization for better conditioning.

        # Refine using gradient descent.
        def pack_params(K: np.ndarray, v: np.ndarray, scale: np.ndarray):
            params = np.array([K[0, 0], K[0, 1], K[1, 1], v[0], v[1]]) / scale
            return params
        
        def unpack_params(params: np.ndarray):
            [k00, k01, k11, v0, v1] = params

            K = np.array([
                [k00, k01],
                [k01, k11]
            ])
            v = np.array([v0, v1])

            H = np.zeros((3, 3), np.float32)
            H[:2, :2] = K / np.linalg.det(K)
            # print(np.linalg.det(K))
            H[2, :2] = v
            H[2, 2] = 1.0


            # Normalise wrt. similarity transformations.
            from_pts = unembed_into_euclidean(embed_into_projective(corners_by_id[1]) @ H.T)
            to_pts = np.array([
                [0, 0],
                [0, 1],
                [1, 1],
                [1, 0.0],
            ])

            E = get_similarity_transform_matrix(from_pts, to_pts)
            H = E @ H

            return H


        def get_residuals(corners, H):
            proj_pts = embed_into_projective(corners) @ H.T
            rolled_proj_pts = np.roll(proj_pts, shift=1, axis=0)
            sides = np.cross(proj_pts, rolled_proj_pts, axis=1)

            rolled_proj_pts = np.roll(proj_pts, shift=2, axis=0)
            sides2 = np.cross(proj_pts, rolled_proj_pts, axis=1)

            def get_dots(sides, other_sides):
                cutoff_dots = np.sum(sides[:, :2] * other_sides[:, :2], axis=1)
                cutoff_norms = np.linalg.norm(sides[:, :2], axis=1)
                cutoff_prev_norms = np.linalg.norm(other_sides[:, :2], axis=1)
                return cutoff_dots / (cutoff_norms * cutoff_prev_norms)

            def get_angles(sides, other_sides):
                dots = get_dots(sides, other_sides)
                angles = np.acos(dots)
                return angles

            def loss_ortho(sides):
                ortho_sides = np.roll(sides, shift=1, axis=0)
                return get_angles(sides, ortho_sides) - np.pi / 2

            def loss_parallel(sides):
                parallel_sides = np.roll(sides, shift=2, axis=0)
                return np.abs(get_dots(sides, parallel_sides)) - 1

            def loss_lengths(pts, length: float, shift: int, cutoff: int):
                prev_pts = np.roll(pts, shift=shift, axis=0)
                diff = subtract_projective_into_euclidean(pts, prev_pts)
                norms = np.linalg.norm(diff[:cutoff], axis=1)
                loss = norms - length
                return loss

            return np.hstack([
                loss_ortho(sides), loss_ortho(sides2), 
                loss_parallel(sides), 
                loss_lengths(proj_pts, length=1.0, shift=1, cutoff=4),
                loss_lengths(proj_pts, length=np.sqrt(2), shift=2, cutoff=2)
            ])


        def reprojection_residuals(params, list_of_corners):
            H = unpack_params(params)

            residuals = []
            for corners in list_of_corners:
                residuals.append(get_residuals(corners, H))
            
            return np.hstack(residuals)

        try:
            result = scipy.optimize.least_squares(
                fun=reprojection_residuals,
                x0=pack_params(K, v, scale),
                args=(np.array(list(corners_by_id.values())),),
                method='trf',
                max_nfev=15000,
                ftol = 1e-12,
                xtol = 1e-12,
                gtol = 1e-12
            )
            H = unpack_params(result.x)
        except ValueError as e:
            H = unpack_params(pack_params(K, v, scale))


        T = np.array([
            [1, 0, 100.0],
            [0, 1, 100.0],
            [0, 0, 1]
        ]) @ np.array([
            [100.0, 0, 0],
            [0, 100.0, 0],
            [0, 0, 1]
        ])


        return T @ H


@timed
def stabilize_frames(interesting_frames: list[int], config: BaseConfig, video_ctx: VideoContext):
    ctx = StabilizerContext(config)
    homographies = []

    for frame_idx, frame in tqdm(open_video_capture(video_ctx, interesting_frames, load_in_8bit_mode=True)):
        corners_by_id = ctx.detect_rectangles(frame)
        H = ctx.compute_orthographic_birds_eye_view(frame, corners_by_id)
        homographies.append(H)

    return homographies

    #     if primary_corners is not None: 
    #         primary_corners_dict[frame_idx] = primary_corners

    #     if secondary_corners is not None:
    #         secondary_corners_dict[frame_idx] = secondary_corners

    # def interpolate_all(existing_dict: dict[int, np.ndarray]):
    #     found_frames = sorted(existing_dict.keys())
    #     for curr, nxt in zip(found_frames[:-1], found_frames[1:]):
    #         ten_curr = existing_dict[curr]
    #         ten_next = existing_dict[nxt]
    #         for idx in interesting_frames:
    #             if curr <= idx <= nxt:
    #                 t = float(idx - curr) / (nxt - curr)
    #                 existing_dict[idx] = (1.0 - t) * ten_curr + t * ten_next

    #     min_found_frame = np.min(np.array(found_frames))
    #     max_found_frame = np.max(np.array(found_frames))
    #     for idx in interesting_frames: 
    #         if idx < min_found_frame:
    #             existing_dict[idx] = existing_dict[min_found_frame]
    #         elif idx > max_found_frame:
    #             existing_dict[idx] = existing_dict[max_found_frame]

    # interpolate_all(primary_corners_dict)
    # interpolate_all(secondary_corners_dict)

    # mean = np.mean(np.array(list(primary_corners_dict.values())), axis=0)
    # TODO: Hotfix.

    # import pickle
    # with open("../data/temp3.pickle", "wb") as file:
    #     pickle.dump([interesting_frames, primary_corners_dict, secondary_corners_dict], file)

    # import pickle
    # with open("../data/temp3.pickle", "rb") as file:
    #     [interesting_frames, primary_corners_dict, secondary_corners_dict] = pickle.load(file)

    return homographies


@timed
def preprocess(video_ctx: VideoContext, config: BaseConfig):
    # interesting_frames, track_frames, original_size = get_interesting_frames(video_ctx, config)

    # import pickle
    # with open("../data/temp.pickle", "wb") as file:
    #     pickle.dump([video_ctx, interesting_frames, track_frames, original_size], file)

    import pickle
    with open("../data/temp.pickle", "rb") as file:
        [video_ctx, interesting_frames, track_frames, original_size] = pickle.load(file)

    if np.any(original_size != config.image_size):
        raise ValueError(f"The image size {original_size} is not the expected large size {config.image_size}.")

    homographies = stabilize_frames(interesting_frames, config, video_ctx)
    video_ctx.homographies = {frame_idx: H for frame_idx, H in zip(interesting_frames, homographies)}

    for idx, frame in open_video_capture(video_ctx, interesting_frames):
        import matplotlib.pyplot as plt
        plt.imshow(frame)
        plt.show()

    exit(0)

    return interesting_frames, track_frames, homographies
