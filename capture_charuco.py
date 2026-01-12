import json
import os
import time
from datetime import datetime

import cv2
import numpy as np


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    config_path = "config.json"
    cfg = load_json(config_path)

    board_cfg = cfg.get("board", {})
    cam_cfg = cfg.get("camera", {})
    cap_cfg = cfg.get("capture", {})

    squares_x = int(board_cfg.get("squares_x"))
    squares_y = int(board_cfg.get("squares_y"))
    square_length_mm = float(board_cfg.get("square_length_mm"))
    marker_length_mm = float(board_cfg.get("marker_length_mm"))
    dict_name = str(board_cfg.get("dictionary"))

    cam_index = int(cam_cfg.get("index"))
    cam_width = int(cam_cfg.get("width"))
    cam_height = int(cam_cfg.get("height"))
    cam_fps = int(cam_cfg.get("fps"))

    min_charuco_corners = int(cap_cfg.get("min_charuco_corners"))
    min_aruco_markers = int(cap_cfg.get("min_aruco_markers"))
    debounce_sec = float(cap_cfg.get("debounce_sec"))

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

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index={cam_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    cap.set(cv2.CAP_PROP_FPS, cam_fps)

    session_dir = os.path.join("captures", "charuco_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)

    print("=== capture_charuco ===")
    print(f"Saving to: {session_dir}")
    print("SPACE = save (if OK), q/ESC = quit")

    saved = 0
    last_save_t = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        vis = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)
        n_markers = 0 if ids is None else len(ids)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        n_charuco = 0
        charuco_corners = None
        charuco_ids = None

        if ids is not None and len(ids) > 0:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=board
            )
            n_charuco = int(retval) if retval is not None else 0

            if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
                cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids, (0, 255, 0))

        good = (n_markers >= min_aruco_markers) and (n_charuco >= min_charuco_corners)

        h, w = vis.shape[:2]
        status = "Board detected" if good else "No board"
        cv2.putText(vis, f"Saved: {saved}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Aruco markers: {n_markers}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Charuco corners: {n_charuco}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Detection: {status}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if good else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, "SPACE=save  q/ESC=quit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("capture_charuco", vis)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        if key in (27, ord("q")):
            break

        if key == 32:  # SPACE
            if now - last_save_t < debounce_sec:
                continue

            if not good:
                print(f"[SKIP] markers={n_markers} (min {min_aruco_markers}), charuco={n_charuco} (min {min_charuco_corners})")
                last_save_t = now
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_name = f"img_{saved:03d}_{ts}.png"
            img_path = os.path.join(session_dir, img_name)

            cv2.imwrite(img_path, frame)
            print(f"[SAVE] {img_path} markers={n_markers} charuco={n_charuco}")

            saved += 1
            last_save_t = now

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
