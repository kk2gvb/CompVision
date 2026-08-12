from pathlib import Path

import cv2
import numpy as np

from stereo_utils import load_calibration, rectify_and_crop, colorize_disparity, ensure_folder


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DEFAULT_CALIB = ROOT_DIR / "data" / "calib" / "90deg_stereocam_calib_param_o1.json"
DEFAULT_VIDEO = ROOT_DIR / "data" / "videos" / "90d_stereo_video_13.mp4"
OUTPUT_DIR = ROOT_DIR / "results" / "output"


def make_stereo_bm(min_disparity=2, num_disparities=160, block_size=15):
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    stereo.setMinDisparity(min_disparity)
    stereo.setTextureThreshold(100)
    stereo.setUniquenessRatio(3)
    stereo.setPreFilterCap(31)
    stereo.setPreFilterSize(23)
    stereo.setPreFilterType(0)
    stereo.setSpeckleRange(10)
    stereo.setSpeckleWindowSize(200)
    stereo.setDisp12MaxDiff(1)
    return stereo


def main():
    ensure_folder(OUTPUT_DIR)
    calib = load_calibration(DEFAULT_CALIB, scale=0.5, alpha=0)
    capture = cv2.VideoCapture(str(DEFAULT_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {DEFAULT_VIDEO}")

    matcher = make_stereo_bm()
    output_path = OUTPUT_DIR / "bm_disparity.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = calib["roi"][2]
    h = calib["roi"][3]
    writer = cv2.VideoWriter(str(output_path), fourcc, 20.0, (w, h))

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (calib["half_width"] * 2, calib["imSize"][1]))
        left_rect, right_rect = rectify_and_crop(frame, calib)
        gray_left = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        disparity = matcher.compute(gray_left, gray_right).astype(np.float32)
        disp_color, _ = colorize_disparity(disparity, min_disparity=2, num_disparities=160)

        writer.write(disp_color)
        cv2.imshow("StereoBM Disparity", disp_color)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
