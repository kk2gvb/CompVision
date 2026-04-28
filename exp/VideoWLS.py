import numpy as np
import cv2
import json
import time
from datetime import datetime

CAMERA_PARAM = "90deg_stereocam_calib_param_o1.json"
VIDEO_PATH = "90d_stereo_video_2.mp4"

# ---- SGBM параметры ----
minDisparity = 0
numDisparities = 192
blockSize = 5

uniquenessRatio = 8
speckleWindowSize = 100
speckleRange = 2
disp12MaxDiff = 1
preFilterCap = 31

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_name = f"./output/T_{timestamp}_ND_{numDisparities}.mp4"
print(video_name)

# ---- Видео ----
camera = cv2.VideoCapture(VIDEO_PATH)

# ---- Калибровка ----
with open(CAMERA_PARAM) as fp:
    cp = json.load(fp)

Kl = np.array(cp["Kl"])
Dl = np.array(cp["Dl"])
Kr = np.array(cp["Kr"])
Dr = np.array(cp["Dr"])
R = np.array(cp["R"])
T = np.array(cp["T"])
imSize = np.array(cp["imSize"])

# ---- Масштаб ----
scale_factor = 0.5
imSize = (imSize * scale_factor).astype(int)
Kl *= scale_factor
Kr *= scale_factor
HALF_WIDTH = imSize[0]

# ---- Rectify + ROI ----
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    Kl, Dl, Kr, Dr, imSize, R, T,
    alpha=0   # убирает чёрные зоны максимально
)

# ---- пересечение ROI (КЛЮЧЕВОЕ) ----
x1, y1, w1, h1 = roi1
x2, y2, w2, h2 = roi2

x = max(x1, x2)
y = max(y1, y2)
w = min(x1 + w1, x2 + w2) - x
h = min(y1 + h1, y2 + h2) - y

print("ROI:", x, y, w, h)

# ---- карты ----
xmap1, ymap1 = cv2.initUndistortRectifyMap(Kl, Dl, R1, P1, imSize, cv2.CV_32FC1)
xmap2, ymap2 = cv2.initUndistortRectifyMap(Kr, Dr, R2, P2, imSize, cv2.CV_32FC1)

# ---- MATCHERS ----
left_matcher = cv2.StereoSGBM_create(
    minDisparity=minDisparity,
    numDisparities=numDisparities,
    blockSize=blockSize,
    P1=8 * blockSize ** 2,
    P2=32 * blockSize ** 2,
    disp12MaxDiff=disp12MaxDiff,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    preFilterCap=preFilterCap,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# ---- WLS ----
wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.5)

# ---- VideoWriter ----
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_name, fourcc, 20.0, (w, h))

# ---- depth ----
current_depth = None

def mouse_callback(event, x_click, y_click, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and current_depth is not None:
        z = current_depth[y_click, x_click]
        print(f"[CLICK] Depth at ({x_click}, {y_click}) = {z:.2f}")

cv2.namedWindow("Disparity")
cv2.setMouseCallback("Disparity", mouse_callback)

prev_time = time.time()

# ---- LOOP ----
while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.resize(frame, (imSize[0]*2, imSize[1]))

    leftImage = frame[:, :HALF_WIDTH]
    rightImage = frame[:, HALF_WIDTH:]

    # rectify
    leftRect = cv2.remap(leftImage, xmap1, ymap1, cv2.INTER_LINEAR)
    rightRect = cv2.remap(rightImage, xmap2, ymap2, cv2.INTER_LINEAR)

    # ---- ROI ----
    leftRect = leftRect[y:y+h, x:x+w]
    rightRect = rightRect[y:y+h, x:x+w]

    leftGray = cv2.cvtColor(leftRect, cv2.COLOR_BGR2GRAY)
    rightGray = cv2.cvtColor(rightRect, cv2.COLOR_BGR2GRAY)

    # ---- disparity ----
    displ = left_matcher.compute(leftGray, rightGray).astype(np.int16)
    dispr = right_matcher.compute(rightGray, leftGray).astype(np.int16)

    filtered = wls_filter.filter(displ, leftGray, None, dispr)
    disparity = filtered.astype(np.float32) / 16.0

    # ---- depth ----
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    depth = points_3D[:, :, 2]
    current_depth = depth

    # ---- mask ----
    valid_mask = disparity > 1

    # ---- визуализация ----
    disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_norm = np.uint8(disp_norm)

    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
    disp_color[~valid_mask] = 0

    out.write(disp_color)

    # FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(disp_color, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Disparity", disp_color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
out.release()
cv2.destroyAllWindows()
