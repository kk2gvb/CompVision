from pathlib import Path
import json

import cv2
import numpy as np


def load_calibration(calib_path: Path, scale: float = 0.5, alpha: int = 0):
    with calib_path.open("r", encoding="utf-8") as fp:
        params = json.load(fp)

    Kl = np.array(params["Kl"], dtype=np.float64)
    Dl = np.array(params["Dl"], dtype=np.float64)
    Kr = np.array(params["Kr"], dtype=np.float64)
    Dr = np.array(params["Dr"], dtype=np.float64)
    R = np.array(params["R"], dtype=np.float64)
    T = np.array(params["T"], dtype=np.float64)
    imSize = np.array(params["imSize"], dtype=np.int32)

    imSize = np.round(imSize * scale).astype(np.int32)
    Kl *= scale
    Kr *= scale
    half_width = int(imSize[0])

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        Kl, Dl, Kr, Dr, tuple(imSize), R, T, alpha=alpha
    )

    xmap1, ymap1 = cv2.initUndistortRectifyMap(Kl, Dl, R1, P1, imSize, cv2.CV_32FC1)
    xmap2, ymap2 = cv2.initUndistortRectifyMap(Kr, Dr, R2, P2, imSize, cv2.CV_32FC1)

    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    x = max(x1, x2)
    y = max(y1, y2)
    w = min(x1 + w1, x2 + w2) - x
    h = min(y1 + h1, y2 + h2) - y

    return {
        "imSize": tuple(imSize.tolist()),
        "half_width": half_width,
        "maps": (xmap1, ymap1, xmap2, ymap2),
        "roi": (x, y, w, h),
        "Q": Q,
    }


def rectify_and_crop(frame: np.ndarray, calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    xmap1, ymap1, xmap2, ymap2 = calibration["maps"]
    x, y, w, h = calibration["roi"]
    half_width = calibration["half_width"]

    left = frame[:, :half_width]
    right = frame[:, half_width:]

    left_rect = cv2.remap(left, xmap1, ymap1, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, xmap2, ymap2, cv2.INTER_LINEAR)

    left_rect = left_rect[y : y + h, x : x + w]
    right_rect = right_rect[y : y + h, x : x + w]

    return left_rect, right_rect


def normalize_disparity(disparity: np.ndarray, min_disparity: int = 0, num_disparities: int = 160) -> np.ndarray:
    disparity = disparity.astype(np.float32) / 16.0
    disparity = (disparity - min_disparity) / float(num_disparities)
    return disparity


def colorize_disparity(disparity: np.ndarray, min_disparity: int = 0, num_disparities: int = 160) -> tuple[np.ndarray, np.ndarray]:
    norm = normalize_disparity(disparity, min_disparity, num_disparities)
    valid_mask = norm > 0
    disp_norm = np.zeros_like(norm, dtype=np.uint8)
    disp_norm[valid_mask] = np.clip(norm[valid_mask] * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
    color[~valid_mask] = 0
    return color, valid_mask


def ensure_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
