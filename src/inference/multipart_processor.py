from .persistence import get_multipart_scan_uploads
from .image_processor import process_single_image
from .config import InferenceConfig
from shared.utils import unembed_into_euclidean, get_euclidean_transform_matrix, invert_euclidean_transform_matrix
from typing import List, Dict, Set, Tuple
from queue import PriorityQueue
import numpy as np, cv2
from tqdm import tqdm
from pathlib import Path


# TODO: Avoid zero edges explicitly?
def calc_max_sp_tree_without_zero_edges(ids: Set[int], nbs_and_weights_by_id: Dict[int, List[Tuple[int, int]]]):
    # Prim's algorithm.
    first = list(ids)[0]

    queue = PriorityQueue()
    queue.put((0, (-1, first)))
    max_sp_tree = set()
    visited = set([first])

    while len(visited) < len(ids):
        last, curr = queue.get()[1] # Discard weight parameter.
        # Push children.
        for (nb, weight) in nbs_and_weights_by_id[curr]:
            if nb not in visited: queue.put((-weight, (curr, nb)))

        if curr in visited: continue
        visited.add(curr)
        max_sp_tree.add((last, curr))

    return max_sp_tree


def collect_marker_data(paths: List[str]):
    img_size = None 
    img_to_marker_ids = []
    img_to_marker_centers = []

    for path in tqdm(paths):
        img = cv2.imread(path)
        proc, trans_corners_per_frame = process_single_image(img)
        img_size = proc.shape[:2][::-1]
        cv2.imwrite(path, proc)

        img_to_marker_ids.append(list(trans_corners_per_frame.keys()))
        img_to_marker_centers.append(np.mean(np.array(list(trans_corners_per_frame.values())), axis=1))
    
    return img_size, img_to_marker_ids, img_to_marker_centers


def get_max_overlap_tree(img_ids: List[int], img_to_marker_ids):
    marker_inters_by_ids = {}
    nbs_and_weights_by_id = {}
    for img_id in img_ids:
        nbs_and_weights = []

        for img_id_2 in img_ids:
            inters = set(img_to_marker_ids[img_id]) \
                .intersection(img_to_marker_ids[img_id_2])
            marker_inters_by_ids[(img_id, img_id_2)] = inters

            weight = len(inters)
            if weight > 0: nbs_and_weights.append((img_id_2, weight))

        nbs_and_weights_by_id[img_id] = nbs_and_weights

    max_sp_tree = calc_max_sp_tree_without_zero_edges(img_ids, nbs_and_weights_by_id)

    return max_sp_tree, marker_inters_by_ids


def get_accumulated_inverse_transforms_wrt_reference(img_ids, marker_inters_by_ids, img_to_marker_ids, img_to_marker_centers, max_sp_tree):
    # Calculate the pairwise orthogonal procrustes transforms.
    procrustes_inv = {}
    for (last, curr) in max_sp_tree:
        marker_inters = marker_inters_by_ids[(last, curr)]
        from_pts, to_pts = [], []
        for marker_id in marker_inters:
            last_centers, curr_centers = img_to_marker_centers[last], img_to_marker_centers[curr]
            last_ids, curr_ids = img_to_marker_ids[last], img_to_marker_ids[curr]
            from_pts.extend([last_centers[i] for i, id in enumerate(last_ids) if id == marker_id])
            to_pts.extend(curr_centers[i] for i, id in enumerate(curr_ids) if id == marker_id)

        from_pts, to_pts = np.array(from_pts), np.array(to_pts)
        procrustes_inv[(last, curr)] = invert_euclidean_transform_matrix(
            get_euclidean_transform_matrix(from_pts, to_pts)
        )


    # Calculate the accumulated transforms wrt. to the reference.
    ref_img = img_ids[0]
    accumulated_inv = { ref_img: np.eye(3) }
    while len(accumulated_inv) < len(img_ids):
        for (parent, child) in max_sp_tree:
            if parent in accumulated_inv and child not in accumulated_inv:
                accumulated_inv[child] = accumulated_inv[parent] @ procrustes_inv[(parent, child)]

    return accumulated_inv, ref_img


def shift_transforms_into_image_space(accumulated_inv, img_size, img_ids, ref_img):
    # Calculate the new bounding boxes.
    bboxes = {}
    for img_id, inv_transform in accumulated_inv.items():
        origi_pts = np.array([
            [0.0, 0.0, 1.0],
            [0.0, img_size[1], 1.0],
            [img_size[0], 0.0, 1.0],
            [*img_size, 1.0]
        ])
        pts = origi_pts @ inv_transform.T

        if img_id == ref_img:
            bboxes[img_id] = np.array([[0.0, 0.0], list(img_size)])
        else:
            pts = unembed_into_euclidean(pts)
            min = np.min(pts, axis=0)
            max = np.max(pts, axis=0)
            bboxes[img_id] = np.array([min, max])

    # Calculate the union bounding box and shift all other frames st. the tl-corner is (0, 0).
    union_tl_corner = np.min(np.concatenate(list(bboxes.values()), axis=0), axis=0)
    union_size = np.max(np.concatenate(list(bboxes.values()), axis=0), axis=0) - union_tl_corner
    T = np.array([
        [1, 0, -union_tl_corner[0]],
        [0, 1, -union_tl_corner[1]],
        [0, 0, 1.0]
    ])

    # Calculate the final transforms for each image.
    transforms = {img_id: T @ accumulated_inv[img_id] for img_id in img_ids}

    return transforms, union_size


def process_multipart_scan(config: InferenceConfig, multipart_scan_id: int):
    paths = get_multipart_scan_uploads(config, multipart_scan_id)
    if len(paths) == 0:
        return None
    
    img_ids = list(range(len(paths)))
    img_size, img_to_marker_ids, img_to_marker_centers = collect_marker_data(paths)

    max_sp_tree, marker_inters_by_ids = get_max_overlap_tree(img_ids, img_to_marker_ids)
    accumulated_inv, ref_img = get_accumulated_inverse_transforms_wrt_reference(img_ids, marker_inters_by_ids, img_to_marker_ids, img_to_marker_centers, max_sp_tree)
    transforms, union_size = shift_transforms_into_image_space(accumulated_inv, img_size, img_ids, ref_img)


    # Output final composite.
    canvas_w = int(np.ceil(union_size[0]))
    canvas_h = int(np.ceil(union_size[1]))
    sum_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    count_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint32)

    for img_id, path in zip(img_ids, paths):
        # load the processed frame and convert to RGB
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # warp into canvas
        M = transforms[img_id]
        warped_img = cv2.warpPerspective(
            img.astype(np.float32), M,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR
        )

        # build a mask of valid pixels and warp it
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
        warped_mask = cv2.warpPerspective(
            mask, M,
            (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST
        )

        sum_canvas += warped_img
        count_canvas += warped_mask

    # normalize to get the average
    avg_canvas = np.zeros_like(sum_canvas, dtype=np.uint8)
    nonzero = count_canvas > 0
    avg_canvas[nonzero] = (
        sum_canvas[nonzero] / count_canvas[nonzero, None]
    ).astype(np.uint8)

    # display
    multipart_output_dir = config.get_multipart_output_dir()
    Path(multipart_output_dir).mkdir(exist_ok=True)
    path = multipart_output_dir / f"{multipart_scan_id}.jpg"
    cv2.imwrite(path, avg_canvas)
    return path
