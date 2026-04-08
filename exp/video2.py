import numpy as np
import cv2
import json

CAMERA_PARAM = "90deg_stereocam_calib_param_o1.json"
VIDEO_PATH = "90d_stereo_video_2.mp4"

WIDTH, HEIGHT = 2560, 720
HALF_WIDTH = WIDTH // 2

# ---- SGBM параметры ----
minDisparity = 0
numDisparities = 160  # должно делиться на 16
blockSize = 3         # 3–7 оптимально

uniquenessRatio = 5
speckleWindowSize = 100
speckleRange = 2
disp12MaxDiff = 1
preFilterCap = 31

# ---- Видео ----
camera = cv2.VideoCapture(VIDEO_PATH)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
camera.set(1, 280)

# ---- Загружаем калибровку ----
with open(CAMERA_PARAM) as fp:
    cp = json.load(fp)

Kl = np.array(cp["Kl"])
Dl = np.array(cp["Dl"])
Kr = np.array(cp["Kr"])
Dr = np.array(cp["Dr"])
R = np.array(cp["R"])
T = np.array(cp["T"])
imSize = np.array(cp["imSize"])

# уменьшаем разрешение в 2 раза
imSize = imSize // 2
Kl *= 0.5
Kr *= 0.5
HALF_WIDTH = imSize[0]

# ---- Rectify ----
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(Kl, Dl, Kr, Dr, imSize, R, T)

xmap1, ymap1 = cv2.initUndistortRectifyMap(Kl, Dl, R1, P1, imSize, cv2.CV_32FC1)
xmap2, ymap2 = cv2.initUndistortRectifyMap(Kr, Dr, R2, P2, imSize, cv2.CV_32FC1)

# ---- SGBM ----
stereo = cv2.StereoSGBM_create(
    minDisparity=minDisparity,
    numDisparities=numDisparities,
    blockSize=blockSize,

    P1=8 * 1 * blockSize ** 2,
    P2=32 * 1 * blockSize ** 2,

    disp12MaxDiff=disp12MaxDiff,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,

    preFilterCap=preFilterCap,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# ---- Основной цикл ----
while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break

    # уменьшаем
    frame = cv2.resize(frame, (imSize[0] * 2, imSize[1]))

    # разделяем
    leftImage = frame[:, :HALF_WIDTH]
    rightImage = frame[:, HALF_WIDTH:]

    # rectify
    leftRect = cv2.remap(leftImage, xmap1, ymap1, cv2.INTER_LINEAR)
    rightRect = cv2.remap(rightImage, xmap2, ymap2, cv2.INTER_LINEAR)

    # grayscale
    leftGray = cv2.cvtColor(leftRect, cv2.COLOR_BGR2GRAY)
    rightGray = cv2.cvtColor(rightRect, cv2.COLOR_BGR2GRAY)

    # disparity
    disparity = stereo.compute(leftGray, rightGray).astype(np.float32) / 16.0

    # нормализация
    disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_norm = np.uint8(disp_norm)

    # цвет
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

    # вывод
    cv2.imshow("Disparity", disp_color)
    cv2.imshow("Frame", cv2.resize(frame, (1600, 450)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
