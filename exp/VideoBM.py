
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib

CAMERA_PARAM = "90deg_stereocam_calib_param_o1.json"  # файл с калибровочными параметрами камеры
VIDEO_PATH = "90d_stereo_video_2.mp4"
WIDTH, HEIGHT = 2560, 720  # разрешение камеры
HALF_WIDTH = WIDTH // 2

numDisparities = 160
blockSize = 15
minDisparity = 2
textureThreshold = 100
uniquenessRatio = 3
preFilterCap = 31
preFilterSize = 23
preFilterType = 0
speckleRange = 10
speckleWindowSize = 200
disp12MaxDiff = 1

camera = cv2.VideoCapture(VIDEO_PATH)  # захват камеры
camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
camera.set(1, 280)

stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
stereo.setMinDisparity(minDisparity)
stereo.setTextureThreshold(textureThreshold)
stereo.setUniquenessRatio(uniquenessRatio)
stereo.setPreFilterCap(preFilterCap)
stereo.setPreFilterSize(preFilterSize)
stereo.setPreFilterType(preFilterType)
stereo.setSpeckleRange(speckleRange)
stereo.setSpeckleWindowSize(speckleWindowSize)
stereo.setDisp12MaxDiff(disp12MaxDiff)


with open(CAMERA_PARAM) as fp:
    cp = json.load(fp)
    Kl, Dl, Kr, Dr, R, T, imSize = np.array(cp["Kl"]), np.array(cp["Dl"]), np.array(cp["Kr"]), np.array(cp["Dr"]), \
                                   np.array(cp["R"]), np.array(cp["T"]), np.array(cp["imSize"])

    imSize = imSize // 2            # Делим параметры камер пополам, т.к. в реальном времени будем 
    Kl, Kr = Kl * 0.5, Kr * 0.5     # использовать половину разрешения от изначального видео (т.е. 640x480 вместо 1280x720)
    HALF_WIDTH = imSize[0]          #


R1, R2, sbgmP1, sbgmP2, Q, validRoi1, validRoi2 = cv2.stereoRectify(Kl, Dl, Kr, Dr, imSize, R, T)
xmap1, ymap1 = cv2.initUndistortRectifyMap(Kl, Dl, R1, sbgmP1, imSize, cv2.CV_32FC1)
xmap2, ymap2 = cv2.initUndistortRectifyMap(Kr, Dr, R2, sbgmP2, imSize, cv2.CV_32FC1)


while camera.isOpened():
    ret, frame = camera.read()
    if ret:
        frame = cv2.resize(frame, (imSize[0]*2, imSize[1]))     # уменьшаем разрешение кадра в 2 раза
        
        leftImage = frame[:, :HALF_WIDTH, :]    # разделяем кадр
        rightImage = frame[:, HALF_WIDTH:, :]   #

        leftImageRectified = cv2.remap(leftImage, xmap1, ymap1, cv2.INTER_LINEAR)       # убираем искажения
        rightImageRectified = cv2.remap(rightImage, xmap2, ymap2, cv2.INTER_LINEAR)     #

        leftImageRectifiedGray = cv2.cvtColor(leftImageRectified, cv2.COLOR_BGR2GRAY)   # переводим в градации серого
        rightImageRectifiedGray = cv2.cvtColor(rightImageRectified, cv2.COLOR_BGR2GRAY) #

        disparity = stereo.compute(leftImageRectifiedGray, rightImageRectifiedGray)     # получаем карту несоответствий
        
        disparity = disparity / 16.0                                # нормализуем несоответствие 
        disparity = (disparity - minDisparity) / numDisparities     #

        validPixels = disparity > 0
        normDisparity = np.zeros(shape=disparity.shape).astype("uint8")
        normDisparity[validPixels] = disparity[validPixels] * 255

        colorizedDisparity = np.zeros((normDisparity.shape[0], normDisparity.shape[1], 3), dtype="uint8")
        temp = cv2.applyColorMap(normDisparity.astype("uint8"), cv2.COLORMAP_JET)
        colorizedDisparity[validPixels] = temp[validPixels]

        cv2.imshow("colorized Disparity", colorizedDisparity)
        cv2.imshow("frame", cv2.resize(frame, (1600, 450)))
        #cv2.imshow("frame", cv2.resize(np.concatenate((leftImageRectified, rightImageRectified), axis=1), (1600, 450)))

    key = cv2.waitKey(1)
    if key == ord('q'):  # если нажато Q завершаем работу программы
        break

cv2.destroyAllWindows()
