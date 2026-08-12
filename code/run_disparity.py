from pathlib import Path

import cv2
import numpy as np

from stereo_utils import ensure_folder, load_calibration, rectify_and_crop, colorize_disparity


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DEFAULT_CALIB = ROOT_DIR / "data" / "calib" / "90deg_stereocam_calib_param_o1.json"
DEFAULT_VIDEO = ROOT_DIR / "data" / "videos" / "90d_stereo_video_13.mp4"
OUTPUT_DIR = ROOT_DIR / "results" / "output"


def make_stereo_matcher(min_disparity=0, num_disparities=208, block_size=5):
    return cv2.StereoSGBM_create(
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


def main():
    ensure_folder(OUTPUT_DIR)

    calib = load_calibration(DEFAULT_CALIB, scale=0.5, alpha=0)
    video = cv2.VideoCapture(str(DEFAULT_VIDEO))
    if not video.isOpened():
        raise RuntimeError(f"Cannot open video: {DEFAULT_VIDEO}")

    matcher = make_stereo_matcher()
    xmap1, ymap1, xmap2, ymap2 = calib["maps"]
    x, y, w, h = calib["roi"]
    Q = calib["Q"]

    output_path = OUTPUT_DIR / "disparity_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, 20.0, (w, h))

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.resize(frame, (calib["half_width"] * 2, calib["imSize"][1]))
        left_rect, right_rect = rectify_and_crop(frame, calib)

        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        disparity = matcher.compute(left_gray, right_gray).astype(np.float32)
        color_disp, valid_mask = colorize_disparity(disparity)

        out.write(color_disp)
        cv2.imshow("Disparity", color_disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
