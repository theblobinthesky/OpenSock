import numpy as np, scipy, cv2


def embed_into_projective(array_of_vectors: np.ndarray) -> np.ndarray:
    return np.concatenate([array_of_vectors, np.ones((array_of_vectors.shape[0], 1))], axis=1)


def unembed_into_euclidean(array_of_vectors: np.ndarray) -> np.ndarray:
    return array_of_vectors[:, :-1] / array_of_vectors[:, -1, None]


def subtract_projective_into_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a.shape) == 1: a = a[None, :]
    if len(b.shape) == 1: b = b[None, :]
    a_hom = a[:, -1, None]
    b_hom = b[:, -1, None]
    return a[:, :-1] / a_hom - b[:, :-1] / b_hom

def get_similarity_transform_matrix(from_pts: np.ndarray, to_pts: np.ndarray) -> np.ndarray:
    # This implements a Umeyama-like method.

    centroid_from = from_pts.mean(axis=0)
    centroid_to = to_pts.mean(axis=0)
    from_pts -= centroid_from[None, ...]
    to_pts -= centroid_to[None, ...]

    # Use Kabsch Algorithm to solve min_{Q is Orthogonal Matrix}{|| Q@A - B ||_F}.
    cov_matrix = from_pts.T @ to_pts
    U, S, Vh = np.linalg.svd(cov_matrix)
    Q = Vh.T @ U.T


    # Assemble final matrix.
    M1 = np.array([
        [1, 0, -centroid_from[0]],
        [0, 1, -centroid_from[1]],
        [0, 0, 1]
    ])

    R = Q
    s = np.sum(S) / np.linalg.norm(from_pts, ord='fro') ** 2
    M2 = np.array([
        [s * R[0, 0], s * R[0, 1], centroid_to[0]],
        [s * R[1, 0], s * R[1, 1], centroid_to[1]],
        [0, 0, 1]
    ])

    return M2 @ M1
