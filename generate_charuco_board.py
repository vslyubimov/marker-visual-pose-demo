import json
import os

import cv2
import numpy as np


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_aruco_dict(dict_name):
    dict_id = getattr(cv2.aruco, dict_name)
    return cv2.aruco.getPredefinedDictionary(dict_id)



def compute_image_size_px(
    squares_x,
    squares_y,
    square_len_m,
    dpi,
    margin_m):
    inches_per_meter = 39.37007874
    px_per_m = dpi * inches_per_meter

    board_w_m = squares_x * square_len_m
    board_h_m = squares_y * square_len_m
    total_w_m = board_w_m + 2.0 * margin_m
    total_h_m = board_h_m + 2.0 * margin_m

    img_w = int(round(total_w_m * px_per_m))
    img_h = int(round(total_h_m * px_per_m))

    board_only_w = int(round(board_w_m * px_per_m))
    board_only_h = int(round(board_h_m * px_per_m))

    return img_w, img_h, board_only_w, board_only_h


def main():
    config_path = "config.json"

    cfg = load_json(config_path)

    board_cfg = cfg.get("board", {})
    render_cfg = cfg.get("render_parameters", {})

    squares_x = int(board_cfg.get("squares_x"))
    squares_y = int(board_cfg.get("squares_y"))
    square_length_mm = float(board_cfg.get("square_length_mm"))
    marker_length_mm = float(board_cfg.get("marker_length_mm"))
    dict_name = str(board_cfg.get("dictionary"))

    dpi = int(render_cfg.get("dpi"))
    margin_mm = float(render_cfg.get("margin_mm"))
    border_bits = int(render_cfg.get("border_bits"))

    out_path = str(render_cfg.get("path"))

    square_len_m = square_length_mm / 1000.0
    marker_len_m = marker_length_mm / 1000.0
    margin_m = margin_mm / 1000.0
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))

    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_len_m, marker_len_m, aruco_dict)
    img_w, img_h, board_only_w, board_only_h = compute_image_size_px(
        squares_x, squares_y, square_len_m, dpi, margin_m
    )

    canvas = np.full((img_h, img_w), 255, dtype=np.uint8)

    board_only = board.generateImage((board_only_w, board_only_h), marginSize=0, borderBits=border_bits)

    off_x = int(round((img_w - board_only_w) / 2))
    off_y = int(round((img_h - board_only_h) / 2))
    canvas[off_y:off_y + board_only_h, off_x:off_x + board_only_w] = board_only

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ok = cv2.imwrite(out_path, canvas)
    if not ok:
        raise RuntimeError(f"Failed to write image to: {out_path}")

    # Print info
    board_w_mm = squares_x * square_length_mm
    board_h_mm = squares_y * square_length_mm
    total_w_mm = board_w_mm + 2.0 * margin_mm
    total_h_mm = board_h_mm + 2.0 * margin_mm

    print("Saved:", out_path)
    print(f"Board squares: {squares_x} x {squares_y}")
    print(f"Square length: {square_length_mm:.3f} mm (geometry)")
    print(f"Marker length: {marker_length_mm:.3f} mm (geometry)")
    print(f"Dictionary: {dict_name}")
    print(f"Image: {img_w} x {img_h} px @ {dpi} DPI")
    print(f"Print size (with margins): ~{total_w_mm:.1f} x {total_h_mm:.1f} mm")


if __name__ == "__main__":
    main()