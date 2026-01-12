import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rvec_to_euler_zyx_deg(rvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))


def polygon_area(corners_4x2):
    x = corners_4x2[:, 0]
    y = corners_4x2[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def main():
    config_path = "config.json"
    cfg = load_json(config_path)

    if len(sys.argv) != 2:
        print("Usage: python detect_pose_aruco.py <camera_calibration.json>")
        return

    calib_path = sys.argv[1]
    calib = load_json(calib_path)

    board_cfg = cfg.get("board", {})
    cam_cfg = cfg.get("camera_parameters", {})
    pose_cfg = cfg.get("detector_parameters", {})

    dict_name = str(board_cfg.get("dictionary"))
    cam_index = int(cam_cfg.get("index", 0))
    cam_width = int(cam_cfg.get("width", 1280))
    cam_height = int(cam_cfg.get("height", 720))
    cam_fps = int(cam_cfg.get("fps", 30))

    marker_length_mm = float(pose_cfg.get("marker_length_mm", 50.0))
    print_hz = float(pose_cfg.get("print_hz", 2.0))
    target_id = int(pose_cfg.get("target_id", -1))

    marker_length_m = marker_length_mm / 1000.0
    print_period = 1.0 / max(0.1, print_hz)

    K = np.array(calib["camera_matrix"], dtype=np.float64)
    dist = np.array(calib["dist_coeffs"], dtype=np.float64).reshape(-1, 1)

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.1

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    cap.set(cv2.CAP_PROP_FPS, cam_fps)

    # ---- LOG FILE ----
    logs_dir = os.path.join("captures", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_name = f"pose_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(logs_dir, log_name)

    log_file = open(log_path, "w", buffering=1)  # line-buffered
    print(f"Logging to: {log_path}")

    print("=== detect_pose_aruco ===")
    print(f"Dictionary: {dict_name}")
    print(f"Marker length: {marker_length_mm} mm")
    print(f"Print rate: {print_hz} Hz | target_id: {target_id}")
    print("q/ESC = quit")

    next_print_t = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        vis = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        best_idx = None
        best_area = -1.0

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)

            for i, mid in enumerate(ids.flatten().tolist()):
                if target_id >= 0:
                    if mid == target_id:
                        best_idx = i
                        break
                else:
                    c = corners[i].reshape(4, 2)
                    a = polygon_area(c)
                    if a > best_area:
                        best_area = a
                        best_idx = i

        if best_idx is not None:
            chosen_id = int(ids.flatten()[best_idx])

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[best_idx]],
                marker_length_m,
                K,
                dist
            )
            rvec = rvecs[0].reshape(3)
            tvec = tvecs[0].reshape(3)

            cv2.drawFrameAxes(vis, K, dist, rvec.reshape(3, 1), tvec.reshape(3, 1), marker_length_m * 0.5)

            roll, pitch, yaw = rvec_to_euler_zyx_deg(rvec)
            x, y, z = float(tvec[0]), float(tvec[1]), float(tvec[2])

            now = time.time()
            if now >= next_print_t:
                next_print_t = now + print_period

                line = (
                    f"id={chosen_id}  "
                    f"x={x:+.4f}  y={y:+.4f}  z={z:+.4f}  "
                    f"roll={roll:+.2f}  pitch={pitch:+.2f}  yaw={yaw:+.2f}"
                )

                print(line)
                log_file.write(line + "\n")

                        # HUD
            cv2.putText(vis, f"ID: {chosen_id}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(vis, f"x: {x:+.3f} m", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"y: {y:+.3f} m", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"z: {z:+.3f} m", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(vis, f"roll : {roll:+.1f} deg", (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"pitch: {pitch:+.1f} deg", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"yaw  : {yaw:+.1f} deg", (20, 205),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        else:
            cv2.putText(vis, "No marker", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        h, w = vis.shape[:2]
        cv2.putText(vis, "q/ESC=quit", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


        cv2.imshow("detect_pose_aruco", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    log_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
