from pathlib import Path

import cv2
import numpy as np

from stereo_utils import load_calibration, rectify_and_crop, colorize_disparity, ensure_folder


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DEFAULT_CALIB = ROOT_DIR / "data" / "calib" / "90deg_stereocam_calib_param_o1.json"
DEFAULT_VIDEO = ROOT_DIR / "data" / "videos" / "90d_stereo_video_13.mp4"
OUTPUT_DIR = ROOT_DIR / "results" / "output"


def make_wls_matcher(min_disparity=0, num_disparities=208, block_size=5):
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size ** 2,
        P2=32 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)
    return left_matcher, right_matcher, wls_filter


def main():
    ensure_folder(OUTPUT_DIR)
    calib = load_calibration(DEFAULT_CALIB, scale=0.5, alpha=0)
    capture = cv2.VideoCapture(str(DEFAULT_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {DEFAULT_VIDEO}")

    left_matcher, right_matcher, wls_filter = make_wls_matcher()
    output_path = OUTPUT_DIR / "wls_disparity.mp4"
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

        disp_left = left_matcher.compute(gray_left, gray_right).astype(np.int16)
        disp_right = right_matcher.compute(gray_right, gray_left).astype(np.int16)
        filtered = wls_filter.filter(disp_left, gray_left, None, disp_right).astype(np.float32)

        color_disp, _ = colorize_disparity(filtered, min_disparity=0, num_disparities=208)
        writer.write(color_disp)
        cv2.imshow("StereoSGBM+WLS Disparity", color_disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
