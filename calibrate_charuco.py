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


def reprojection_error(object_points_list, image_points_list, rvecs, tvecs, K, dist):
    total_err2 = 0.0
    total_pts = 0

    for obj_pts, img_pts, rvec, tvec in zip(object_points_list, image_points_list, rvecs, tvecs):
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
    charuco_detector = cv2.aruco.CharucoDetector(board)

    img_paths = sorted(
        glob(os.path.join(images_dir, "*.png")) +
        glob(os.path.join(images_dir, "*.jpg")) +
        glob(os.path.join(images_dir, "*.jpeg"))
    )

    if len(img_paths) == 0:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    object_points_list = []
    image_points_list = []
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

        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
        
        if charuco_ids is None or len(charuco_ids) < 6:
            skipped += 1
            continue

        object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)

        if object_points is None or image_points is None:
            skipped += 1
            continue

        object_points_list.append(object_points.astype(np.float32))
        image_points_list.append(image_points.astype(np.float32))
        used_files.append(os.path.basename(p))

    usable = len(used_files)
    print(f"Total images: {len(img_paths)} | Usable: {usable} | Skipped: {skipped}")

    if usable < 10:
        raise RuntimeError("Too few usable images (need ~15-20+ recommended).")

    # Pinhole + radial and tangential distortion (k1,k2,p1,p2,k3)
    flags = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-9)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=object_points_list,
        imagePoints=image_points_list,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags,
        criteria=criteria
    )

    mean_reproj = reprojection_error(object_points_list, image_points_list, rvecs, tvecs, K, dist)

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
