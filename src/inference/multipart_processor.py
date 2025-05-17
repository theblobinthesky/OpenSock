# from persistence import get_multipart_scan_uploads
# from image_processor import process_single_image
from shared.utils import unembed_into_euclidean, get_euclidean_transform_matrix, invert_euclidean_transform_matrix
from typing import List, Dict, Set, Tuple
from queue import PriorityQueue
import numpy as np, cv2


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



def process_multipart_scan(multipart_scan_id: int):
    paths = get_multipart_scan_uploads(multipart_scan_id)
    if len(paths) == 0:
        return None

    img_size = (5.0, 5.0)

    img_ids = [0, 1, 2]

    img_to_marker_ids = [
        [0, 1, 2],
        [2],
        [1, 2, 3]
    ]

    img_to_marker_centers = [
        np.array([
            [1.0, 1.0],
            [3.0, 2.0],
            [2.0, 3.0]
        ]),
        np.array([
            [2.0, 3.0]
        ]),
        np.array([
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 4.0]
        ]),
    ]

    # Find the tree of the images with the max. number of overlapping markers.
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
        [0, 1, -union_tl_corner[0]],
        [0, 0, 1.0]
    ])
    procrustes_inv = {key: T @ M for key, M in procrustes_inv.items()}


    # Calculate the final transforms for each image.
    transforms = {curr: M for (_, curr), M in procrustes_inv.items()}
    transforms[ref_img] = np.eye(3)


    # Apply to the images.


    print(procrustes_inv)

   
