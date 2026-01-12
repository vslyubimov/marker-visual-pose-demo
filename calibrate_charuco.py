import json
import os
import sys
from glob import glob

import cv2
import numpy as np


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def reprojection_error_charuco(board, all_corners, all_ids, rvecs, tvecs, K, dist):
    total_err2 = 0.0
    total_pts = 0

    chessboard_corners_3d = board.getChessboardCorners()  # (N,3)

    for corners, ids, rvec, tvec in zip(all_corners, all_ids, rvecs, tvecs):
        obj_pts = chessboard_corners_3d[ids.flatten(), :].reshape(-1, 1, 3).astype(np.float32)
        img_pts = corners.reshape(-1, 1, 2).astype(np.float32)

        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        err = cv2.norm(img_pts, proj, cv2.NORM_L2)

        n = len(obj_pts)
        total_err2 += err * err
        total_pts += n

    if total_pts == 0:
        return float("nan")

    return float(np.sqrt(total_err2 / total_pts))


def main():
    config_path = "config.json"
    cfg = load_json(config_path)

    if len(sys.argv) != 2:
        print("Usage: python calibrate_charuco.py <images_dir>")
        print("Example: python calibrate_charuco.py captures/charuco_date_time")
        return

    images_dir = sys.argv[1]

    board_cfg = cfg.get("board", {})

    squares_x = int(board_cfg.get("squares_x"))
    squares_y = int(board_cfg.get("squares_y"))
    square_length_mm = float(board_cfg.get("square_length_mm"))
    marker_length_mm = float(board_cfg.get("marker_length_mm"))
    dict_name = str(board_cfg.get("dictionary"))

    square_len_m = square_length_mm / 1000.0
    marker_len_m = marker_length_mm / 1000.0

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_len_m, marker_len_m, aruco_dict)

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.1

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    img_paths = sorted(
        glob(os.path.join(images_dir, "*.png")) +
        glob(os.path.join(images_dir, "*.jpg")) +
        glob(os.path.join(images_dir, "*.jpeg"))
    )

    if len(img_paths) == 0:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    all_corners = []
    all_ids = []
    used_files = []

    img_size = None
    skipped = 0

    for p in img_paths:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            skipped += 1
            continue

        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (w,h)

        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        if marker_ids is None or len(marker_ids) == 0:
            skipped += 1
            continue

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=marker_corners,
            markerIds=marker_ids,
            image=gray,
            board=board
        )

        if charuco_ids is None or len(charuco_ids) < 6 or int(retval) <= 0:
            skipped += 1
            continue

        all_corners.append(charuco_corners)
        all_ids.append(charuco_ids)
        used_files.append(os.path.basename(p))

    usable = len(all_ids)
    print(f"Total images: {len(img_paths)} | Usable: {usable} | Skipped: {skipped}")

    if usable < 10:
        raise RuntimeError("Too few usable images (need ~15-20+ recommended).")

    # Pinhole + radial and tangential distortion (k1,k2,p1,p2,k3)
    flags = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-9)

    K_init = np.eye(3, dtype=np.float64)
    dist_init = np.zeros((5, 1), dtype=np.float64)

    rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=img_size,
        cameraMatrix=K_init,
        distCoeffs=dist_init,
        flags=flags,
        criteria=criteria
    )

    mean_reproj = reprojection_error_charuco(board, all_corners, all_ids, rvecs, tvecs, K, dist)

    out = {
        "calibration_type": "charuco",
        "image_size": {"width": int(img_size[0]), "height": int(img_size[1])},
        "board": {
            "squares_x": squares_x,
            "squares_y": squares_y,
            "square_length_mm": square_length_mm,
            "marker_length_mm": marker_length_mm,
            "dictionary": dict_name
        },
        "stats": {
            "total_images": int(len(img_paths)),
            "usable_images": int(usable),
            "skipped_images": int(skipped),
            "rms": float(rms),
            "mean_reprojection_error_px": float(mean_reproj)
        },
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.flatten().tolist(),
        "used_images": used_files
    }

    out_path = os.path.join(images_dir, "camera_calibration.json")
    save_json(out_path, out)

    print("=== Calibration saved ===")
    print(f"RMS: {rms:.6f}")
    print(f"Mean reprojection error (px): {mean_reproj:.6f}")
    print(f"Output: {out_path}")
    print("K:\n", K)
    print("dist:\n", dist.flatten())


if __name__ == "__main__":
    main()
