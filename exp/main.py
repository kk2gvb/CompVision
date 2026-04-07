import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import time

#============================Входные данные============================

f = 1040.0              # фокусное расстояние (в пикселях)
baseline = 85           # расстояние между камерами
numDisparities = 128    # параметры поиска несоответсвий
blockSize = 5           #
scale = 0.8             # соотношение условных единиц глубины с реальными


leftImageRaw = cv2.imread('../RobotDogStereoDemo/opencv_samples/aloeL.jpg')      # тестовые изображения
rightImageRaw = cv2.imread('../RobotDogStereoDemo/opencv_samples/aloeR.jpg')     #


#============================Получение карты несоответствий(BM and SGBM)============================


leftImage = cv2.cvtColor(leftImageRaw, cv2.COLOR_BGR2GRAY)      # переводим в градации серого
rightImage = cv2.cvtColor(rightImageRaw, cv2.COLOR_BGR2GRAY)    #

bm_time_begin_dis = time.process_time()

stereoBM = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
disparityBM = stereoBM.compute(leftImage, rightImage)   # получаем карту несоответствий

bm_time_end_dis = time.process_time()
dis_elapsed1 = (bm_time_begin_dis - bm_time_end_dis)

sgbm_time_begin_dis = time.process_time()

stereoSGBM = cv2.StereoSGBM_create(numDisparities=numDisparities, blockSize=blockSize)
disparitySGBM = stereoSGBM.compute(leftImage, rightImage)   # получаем карту несоответствий

sgbm_time_end_dis = time.process_time()
dis_elapsed2 = (sgbm_time_begin_dis - sgbm_time_end_dis)

fig, (dis_bm, dis_sgbm) = plt.subplots(1, 2)

dis_bm.set_title(f'stereoBM\n{dis_elapsed1:.3f} $nano s$')
dis_bm.imshow(disparityBM)

dis_sgbm.set_title(f'stereoSGBM\n{dis_elapsed2:.3f} $nano s$')
dis_sgbm.imshow(disparitySGBM)

plt.suptitle("Карты несоответствий")
plt.tight_layout()
plt.savefig("../CV/figs/disparity_bm_vs_sgbm.png", dpi=300)
plt.show()

#============================Получение карты глубин============================

validPixelsBM = disparityBM > 0     # избавляемся от деления на 0

bm_time_begin_dep = time.process_time()

depthBM = np.zeros(shape=leftImage.shape).astype("uint8")
depthBM[validPixelsBM] = (f * baseline) / (scale * disparityBM[validPixelsBM])

bm_time_end_dep = time.process_time()
dep_elapsed1 = (bm_time_begin_dep - bm_time_end_dep)

#Наводим красоты
depthBM = cv2.equalizeHist(depthBM)
colorDepthBM = np.zeros((leftImage.shape[0], leftImage.shape[1], 3), dtype="uint8")
tmp = cv2.applyColorMap(depthBM, cv2.COLORMAP_JET)
colorDepthBM[validPixelsBM] = tmp[validPixelsBM]


validPixelsSGBM = disparitySGBM > 0     # избавляемся от деления на 0

sgbm_time_begin_dep = time.process_time()

depthSGBM = np.zeros(shape=leftImage.shape).astype("uint8")
depthSGBM[validPixelsSGBM] = (f * baseline) / (scale * disparitySGBM[validPixelsSGBM])

#Наводим красоты
depthSGBM = cv2.equalizeHist(depthSGBM)
colorDepthSGBM = np.zeros((leftImage.shape[0], leftImage.shape[1], 3), dtype="uint8")
tmp = cv2.applyColorMap(depthSGBM, cv2.COLORMAP_JET)
colorDepthSGBM[validPixelsSGBM] = tmp[validPixelsSGBM]

sgbm_time_end_dep = time.process_time()
dep_elapsed2 = (sgbm_time_begin_dep - sgbm_time_end_dep)

fig, ((dep_bm, dep_sgbm), (color_dep_bm, color_dep_sgbm)) = plt.subplots(2, 2)

# dep_bm.subplot(2,2,1)
dep_bm.set_title(f'stereoBM\n{dep_elapsed1:.3f} $nano s$')
dep_bm.imshow(depthBM)

# dep_bm.subplot(2,2,2)
color_dep_bm.set_title(f'stereoBM(color)')
color_dep_bm.imshow(colorDepthBM)


# dep_bm.subplot(2,2,3)
dep_sgbm.set_title(f'stereoSGBM\n{dep_elapsed2:.3f} $nano s$')
dep_sgbm.imshow(depthSGBM)

# dep_bm.subplot(2,2,4)
color_dep_sgbm.set_title(f'stereoSGBM(color)')
color_dep_sgbm.imshow(colorDepthSGBM)


plt.suptitle("Карты глубин")
plt.tight_layout()
plt.savefig("../CV/figs/depth_bm_vs_sgbm.png", dpi=300)
plt.show()
