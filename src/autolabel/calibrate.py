import cv2
import numpy as np
import scipy
from pathlib import Path
from tqdm import tqdm

IMAGE_DIR   = Path("../data/calibration")
ROWS, COLS  = 6, 9            # inner‑corner grid
SQUARE_SIZE = 0.0255          # metres


def detect_corners(images):
    objp = np.zeros((ROWS * COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)

    objpoints, imgpoints, valid_images = [], [], []
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001)

    for path in tqdm(images, desc="Detecting corners"):
        img = cv2.imread(str(path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            (COLS, ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not found:
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp.copy())
        imgpoints.append(corners)
        valid_images.append(img)

    return objpoints, imgpoints, valid_images


def calibrate_opencv(objpoints, imgpoints, image_size):
    scaled = [o * SQUARE_SIZE for o in objpoints]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        scaled, imgpoints, image_size, None, None
    )

    all_errors = []
    for i, op in enumerate(scaled):
        proj, _ = cv2.projectPoints(op, rvecs[i], tvecs[i], K, dist)
        proj = proj.reshape(-1, 2)
        actual = imgpoints[i].reshape(-1, 2)

        # per‑point errors
        errs = np.linalg.norm(actual - proj, axis=1)
        all_errors.extend(errs)

    mean_error = float(np.mean(all_errors))
    rms_error  = float(np.sqrt(np.mean(np.square(all_errors))))

    return {"name": "opencv",  "rms": rms_error, "mean": mean_error}


def calibrate_fisheye(objpoints, imgpoints, image_size):
    scaled = [(o.astype(np.float64).reshape(-1, 1, 3) * SQUARE_SIZE)
              for o in objpoints]
    K = np.zeros((3, 3), np.float64)
    D = np.zeros((4, 1), np.float64)

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        scaled, imgpoints, image_size, K, D, None, None,
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    )

    all_errors = []
    for i, op in enumerate(scaled):
        proj, _ = cv2.fisheye.projectPoints(op, rvecs[i], tvecs[i], K, D)
        proj = proj.reshape(-1, 2)
        actual = imgpoints[i].reshape(-1, 2)

        errs = np.linalg.norm(actual - proj, axis=1)
        all_errors.extend(errs)

    mean_error = float(np.mean(all_errors))
    rms_error  = float(np.sqrt(np.mean(np.square(all_errors))))

    return {"name": "fisheye", "rms": rms_error, "mean": mean_error}


    # # Normalize image points.
    # cx, cy = img_pts.mean(axis=0)
    # pts0 = img_pts - [cx, cy]
    # mean_dist = np.mean(np.linalg.norm(pts0, axis=1), axis=0)
    # s = np.sqrt(2) / mean_dist
    # T = np.array([
    #     [s, 0, -s * cx],
    #     [0, s, -s * cy],
    #     [0, 0, 1]
    # ], np.float64)


    # # Normalize object points.
    # cx, cy, cz = obj_pts.mean(axis=0)
    # pts0 = obj_pts - [cx, cy, cz]
    # mean_dist = np.mean(np.linalg.norm(pts0, axis=1), axis=0)
    # s = np.sqrt(3) / mean_dist
    # U = np.array([
    #     [s, 0, 0, -s * cx],
    #     [0, s, 0, -s * cy],
    #     [0, 0, s, -s * cz],
    #     [0, 0, 0, 1]
    # ], np.float64)


    # # Normalize both point sets.
    # b = obj_pts.shape[0]
    # img_hom_norm = np.hstack([img_pts, np.ones((b, 1))]) @ T.T
    # obj_hom_norm = np.hstack([obj_pts, np.ones((b, 1))]) @ U.T


    # # Initialize using linear solve.
    # rows = []
    # for (x, y, w), X in zip(img_hom_norm, obj_hom_norm):
    #     rows.append(np.hstack([np.zeros(4), -w * X.T, y * X.T]))
    #     rows.append(np.hstack([w * X.T, np.zeros(4), -x * X.T]))
    # M = np.vstack(rows)

    # _, _, Vh = np.linalg.svd(M)
    # P = Vh[-1].reshape((3, 4))

    # # TODO

    # # Apply both normalizations.
    # P = np.linalg.inv(T) @ P @ U

def find_homography(from_pts: np.ndarray, to_pts: np.ndarray) -> np.ndarray:
    src = from_pts.reshape(-1,1,2).astype(np.float64)
    dst =   to_pts.reshape(-1,1,2).astype(np.float64)
    H, mask = cv2.findHomography(src, dst, method=0)  # pure DLT + LS
    return H

def get_null_space_vector(M: np.ndarray) -> np.ndarray:
    _, _, Vh = np.linalg.svd(M)
    return Vh[-1]


def init_intrinsics_and_extrinsics(obj_pts_list, img_pts_list):
    # Estimate a single initial intrinsic matrix
    # and an initial extrinsic matrix for every image.

    Hs = []
    rows = []

    for b in range(len(obj_pts_list)):
        obj_pts = obj_pts_list[b].reshape((-1, 2))
        img_pts = img_pts_list[b].reshape((-1, 2))

        H = find_homography(obj_pts, img_pts).T
        Hs.append(H)

        def calc_vij(i, j):
            return np.array([
                H[i, 0] * H[j, 0],
                H[i, 0] * H[j, 1] + H[i, 1] * H[j, 0],
                H[i, 1] * H[j, 1],
                H[i, 2] * H[j, 0] + H[i, 0] * H[j, 2],
                H[i, 2] * H[j, 1] + H[i, 1] * H[j, 2],
                H[i, 2] * H[j, 2]
            ], np.float64)
    
        v12 = calc_vij(0, 1)
        v11 = calc_vij(0, 0)
        v22 = calc_vij(1, 1)

        rows.append(v12.T)
        rows.append((v11 - v22).T)

    M = np.vstack(rows)
    b = get_null_space_vector(M)

    # Recompute the intrinsic matrix from b first.
    [B11, B12, B22, B13, B23, B33] = b
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 ** 2)
    lam = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha_sqred = lam / B11
    alpha = np.sqrt(alpha_sqred)
    beta = np.sqrt(lam * B11  / (B11 * B22 - B12 ** 2))
    gamma = -B12 * alpha_sqred * beta / lam
    u0 = gamma * v0 / beta - B13 * alpha_sqred / lam

    fx, fy = alpha, beta
    scew = gamma
    px, py = u0, v0

    A = np.array([
        [fx, scew, px],
        [0, fy, py],
        [0, 0, 1],
    ])


    # Recompute [R_i, t_i] for all images.
    A_inv = np.linalg.inv(A)
    Rs = []
    ts = []
    for b in range(len(obj_pts_list)):
        H = Hs[b]
        f = (1.0 / np.linalg.norm(A_inv @ H[0].T) + 1.0 / np.linalg.norm(A_inv @ H[1].T)) * 0.5
        r1 = f * A_inv @ H[0].T
        r2 = f * A_inv @ H[1].T
        r3 = np.linalg.cross(r1, r2)
        t = f * A_inv @ H[2].T
        R = np.hstack([r1, r2, r3]).reshape((3, 3))

        # Snap to the nearest orthonormal matrix.
        U, _, Vh = np.linalg.svd(R)
        R = U @ Vh

        Rs.append(R)
        ts.append(t.reshape((3, 1)))

    return (fx, fy, scew, px, py), Rs, ts


def refine_intrinsics_and_extrinsics(intrinsics, Rs, ts, obj_pts_list, img_pts_list):
    # Refine the initial estimate using Levenberg-Maquard decent.
    fx, fy, scew, px, py = intrinsics
    extrinsics = [np.concatenate([R.ravel(), t.ravel()]) for R, t in zip(Rs, ts)]
    intrinsics = np.array([fx, fy, scew, px, py])

    def pack_params(intrinsics, extrinsics):
        params = np.hstack([intrinsics.ravel(), np.concatenate(extrinsics).ravel()])
        return params
    
    def unpack_params(params: np.ndarray):
        [fx, fy, scew, px, py] = params[:5]
        extrinsics = params[5:].reshape((-1, 9 + 3))
        extrinsics = [(e[:9].reshape((3, 3)), e[9:].reshape((3, 1))) for e in extrinsics]

        return (fx, fy, scew, px, py), extrinsics

    def project_points(pt, intrinsics, rvec, tvec):
        [fx, fy, scew, px, py] = intrinsics
        A = np.array([
            [fx, scew, px],
            [0, fy, py],
            [0, 0, 1]
        ])

        b = pt.shape[0]
        pt = np.hstack([pt, np.zeros((b, 1)), np.ones((b, 1))])
        pt = (A @ np.hstack([rvec, tvec]) @ pt.T).T
        pt = pt[:, :2] / pt[:, 2, None]

        return pt[:, None, :]

    def reprojection_residuals(params, obj_pts_list, img_pts_list):
        intrinsics, extrinsics = unpack_params(params)
        residuals = []
        for (X, x_obs), (rvec, tvec) in zip(zip(obj_pts_list, img_pts_list), extrinsics):
            x_proj = project_points(X, intrinsics, rvec, tvec)
            residuals.append((x_obs - x_proj).ravel())

        return np.hstack(residuals)

    result = scipy.optimize.least_squares(
        fun=reprojection_residuals,
        x0=pack_params(intrinsics, extrinsics),
        args=(obj_pts_list, img_pts_list),
        method='lm',
        max_nfev=5000
    )

    return unpack_params(result.x)


def init_distortion_coefficients(intrinsics, A, extrinsics, obj_pts_list, img_pts_list):
    fx, fy, _, px, py = intrinsics

    # Compute errors.
    all_proj_pts = []

    for b in range(len(obj_pts_list)):
        R, t = extrinsics[b]
        obj_pts = obj_pts_list[b]
        obj_pts = np.concatenate([obj_pts, np.zeros((obj_pts.shape[0], 1)), np.ones((obj_pts.shape[0], 1))], axis=1)

        proj_pts = (A @ np.hstack([R, t]) @ obj_pts.T).T
        proj_pts = proj_pts[:, :2] / proj_pts[:, 2, None]
        all_proj_pts.append(proj_pts)
    
    img_observed = np.array(img_pts_list).reshape((-1, 2))
    pts_ideal = np.concatenate(all_proj_pts)


    # Estimate initial distortion coefficients.
    M_rows, b_rows = [], []
    for (u_hat, v_hat), (u, v) in zip(img_observed, pts_ideal):
        x, y = (u - px) / fx, (v - py) / fy
        s = x ** 2 + y ** 2

        M_rows.append([(u - px) * s, (u - px) * s ** 2])
        M_rows.append([(v - py) * s, (v - py) * s ** 2])

        b_rows.append([u_hat - u])
        b_rows.append([v_hat - v])

    M, b = np.vstack(M_rows), np.vstack(b_rows)
    return np.linalg.lstsq(M, b)[0].ravel()

def project_points(pts, intrinsics, extrinsic, dist_coeff):
    [fx, fy, scew, px, py] = intrinsics
    rvec, tvec = extrinsic
    k1, k2 = dist_coeff

    # Apply intrinsic and extrinsic transforms.
    A = np.array([
        [fx, scew, px],
        [0, fy, py],
        [0, 0, 1]
    ])

    b = pts.shape[0]
    pts = np.hstack([pts, np.zeros((b, 1)), np.ones((b, 1))])
    pts = (A @ np.hstack([rvec, tvec]) @ pts.T).T
    pts = pts[:, :2] / pts[:, 2, None]

    # Apply distortion coefficients.
    u, v = pts[:, 0, None], pts[:, 1, None]
    x, y = (u - px) / fx, (v - py) / fy
    s = x ** 2 + y ** 2
    pts += (pts - [px, py]) * (k1 * s + k2 * s ** 2)

    return pts[:, None, :]


def refine_everything(intrinsics, extrinsics, dist_coeff, obj_pts_list, img_pts_list):
    # Refine the current estimate using Levenberg-Maquard decent.
    extrinsics = [np.concatenate([R.ravel(), t.ravel()]) for R, t in extrinsics]

    def pack_params(dist_coeff, intrinsics, extrinsics):
        dist_coeff = np.array(dist_coeff)
        intrinsics = np.array(intrinsics)
        params = np.hstack([dist_coeff, intrinsics, np.concatenate(extrinsics).ravel()])
        return params
    
    def unpack_params(params: np.ndarray):
        dist_coeff = params[:2]
        intrinsics = params[2:7]
        extrinsics = params[7:].reshape((-1, 9 + 3))
        extrinsics = [(e[:9].reshape((3, 3)), e[9:].reshape((3, 1))) for e in extrinsics]

        return intrinsics, extrinsics, dist_coeff

    def reprojection_residuals(params, obj_pts_list, img_pts_list):
        intrinsics, extrinsics, dist_coeff = unpack_params(params)

        residuals = []
        for (X, x_obs), extrinsic in zip(zip(obj_pts_list, img_pts_list), extrinsics):
            x_proj = project_points(X, intrinsics, extrinsic, dist_coeff)
            residuals.append((x_obs - x_proj).ravel())
        return np.hstack(residuals)

    result = scipy.optimize.least_squares(
        fun=reprojection_residuals,
        x0=pack_params(dist_coeff, intrinsics, extrinsics),
        args=(obj_pts_list, img_pts_list),
        method='lm',
        max_nfev=5000
    )

    return unpack_params(result.x)


def calibrate_custom(obj_pts_list, img_pts_list, image_size):
    obj_pts_list = np.array(obj_pts_list, np.float64)
    img_pts_list = np.array(img_pts_list, np.float64)
    image_size = np.array(image_size, np.float64)

    # TODO: Remove this later. It's stoooopid.
    obj_pts_list = obj_pts_list[:, :, :2]

    assert obj_pts_list.shape[0] == img_pts_list.shape[0]
    assert obj_pts_list.shape[0] >= 3

    intrinsics, Rs, ts = init_intrinsics_and_extrinsics(obj_pts_list, img_pts_list)
    intrinsics, extrinsics = refine_intrinsics_and_extrinsics(intrinsics, Rs, ts, obj_pts_list, img_pts_list)
    (fx, fy, scew, px, py) = intrinsics
    A = np.array([
        [fx, scew, px],
        [0, fy, py],
        [0, 0, 1]
    ])

    dist_coeff = init_distortion_coefficients(intrinsics, A, extrinsics, obj_pts_list, img_pts_list)
    intrinsics, extrinsics, dist_coeff = refine_everything(intrinsics, extrinsics, dist_coeff, obj_pts_list, img_pts_list)

    img_pts = np.array(img_pts_list).reshape((-1, 2))
    proj_pts = [project_points(obj_pts, intrinsics, extrinsic, dist_coeff) for obj_pts, extrinsic in zip(obj_pts_list, extrinsics)]
    proj_pts = np.array(proj_pts).reshape((-1, 2))

    errors = np.linalg.norm(img_pts - proj_pts, ord=2, axis=1)
    rms = np.sqrt(np.mean(errors ** 2))
    mean_error = np.mean(errors)

    return {"name": "custom", "rms": rms, "mean": mean_error}

def main():
    # images = sorted(IMAGE_DIR.glob("*.jpg")) + sorted(IMAGE_DIR.glob("*.png"))
    # if not images:
    #     raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

    # objp, imgp, valid_imgs = detect_corners(images)
    # if not objp:
    #     raise RuntimeError("No corners detected")
    
    # h, w = valid_imgs[0].shape[:2]
    # image_size = (w, h)

    # import pickle
    # with open('../data/calib.pickle', 'wb') as file:
    #     pickle.dump([objp, imgp, image_size], file)

    import pickle
    with open('../data/calib.pickle', 'rb') as file:
        [objp, imgp, image_size] = pickle.load(file)

    results = [
        calibrate_opencv(objp, imgp, image_size),
        calibrate_fisheye(objp, imgp, image_size),
        calibrate_custom(objp, imgp, image_size)
    ]

    for res in results:
        print(f"\n{res['name'].upper()} metrics:")
        print(f"  RMS  : {res['rms']:.4f} px")
        print(f"  Mean : {res['mean']:.4f} px")


if __name__ == "__main__":
    main()
