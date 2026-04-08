import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import time

def gradGrayImage(leftImageRaw, rightImageRaw):
    leftImage = cv2.cvtColor(leftImageRaw, cv2.COLOR_BGR2GRAY)      # переводим в градации серого
    rightImage = cv2.cvtColor(rightImageRaw, cv2.COLOR_BGR2GRAY)    #

    return leftImage, rightImage

def disparity_BMandSGBM(leftImageRaw, rightImageRaw, f, Baseline, numDisparities, blockSize, scale, figfile):
    
    leftImage, rightImage = gradGrayImage(leftImageRaw, rightImageRaw)

    stereoBM = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparityBM = stereoBM.compute(leftImage, rightImage)   # получаем карту несоответствий

    stereoSGBM = cv2.StereoSGBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparitySGBM = stereoSGBM.compute(leftImage, rightImage)   # получаем карту несоответствий

    fig, (dis_bm, dis_sgbm) = plt.subplots(1, 2)

    dis_bm.set_title('stereoBM')
    dis_bm.imshow(disparityBM)

    dis_sgbm.set_title('stereoSGBM')
    dis_sgbm.imshow(disparitySGBM)

    plt.suptitle("Карты несоответствий")
    plt.tight_layout()
    plt.savefig(figfile, dpi=300)
    plt.show()
    
    return disparityBM, disparitySGBM


def depth_BMandSGBM(leftImageRaw, rightImageRaw, f, Baseline, numDisparities, blockSize, scale, figfile, disparityBM, disparitySGBM):

    leftImage, rightImage = gradGrayImage(leftImageRaw, rightImageRaw)
    
    validPixelsBM = disparityBM > 0     # избавляемся от деления на 0

    depthBM = np.zeros(shape=leftImage.shape).astype("uint8")
    depthBM[validPixelsBM] = (f * baseline) / (scale * disparityBM[validPixelsBM])

    #Наводим красоты
    depthBM = cv2.equalizeHist(depthBM)
    colorDepthBM = np.zeros((leftImage.shape[0], leftImage.shape[1], 3), dtype="uint8")
    tmp = cv2.applyColorMap(depthBM, cv2.COLORMAP_JET)
    colorDepthBM[validPixelsBM] = tmp[validPixelsBM]

    validPixelsSGBM = disparitySGBM > 0     # избавляемся от деления на 0

    depthSGBM = np.zeros(shape=leftImage.shape).astype("uint8")
    depthSGBM[validPixelsSGBM] = (f * baseline) / (scale * disparitySGBM[validPixelsSGBM])

    #Наводим красоты
    depthSGBM = cv2.equalizeHist(depthSGBM)
    colorDepthSGBM = np.zeros((leftImage.shape[0], leftImage.shape[1], 3), dtype="uint8")
    tmp = cv2.applyColorMap(depthSGBM, cv2.COLORMAP_JET)
    colorDepthSGBM[validPixelsSGBM] = tmp[validPixelsSGBM]

    fig, ((dep_bm, dep_sgbm), (color_dep_bm, color_dep_sgbm)) = plt.subplots(2, 2)

    dep_bm.set_title(f'stereoBM')
    dep_bm.imshow(depthBM)

    color_dep_bm.set_title(f'stereoBM(color)')
    color_dep_bm.imshow(colorDepthBM)

    dep_sgbm.set_title(f'stereoSGBM')
    dep_sgbm.imshow(depthSGBM)

    color_dep_sgbm.set_title(f'stereoSGBM(color)')
    color_dep_sgbm.imshow(colorDepthSGBM)

    plt.suptitle("Карты глубин")
    plt.tight_layout()
    plt.savefig(figfile, dpi=300)
    plt.show()
    
    return depthBM, depthSGBM, colorDepthBM, colorDepthSGBM


#============================Входные данные============================

dis_filename = "../CV/figs/disparity_bm_vs_sgbm.png"
dep_filename = "../CV/figs/depth_bm_vs_sgbm.png"

f = 1040.0              # фокусное расстояние (в пикселях)
baseline = 85           # расстояние между камерами
numDisparities = 128    # параметры поиска несоответсвий
blockSize = 25           #
scale = 0.8             # соотношение условных единиц глубины с реальными


leftImageRaw = cv2.imread('../RobotDogStereoDemo/opencv_samples/aloeL.jpg')      # тестовые изображения
rightImageRaw = cv2.imread('../RobotDogStereoDemo/opencv_samples/aloeR.jpg')     #

disparityBM, disparitySGBM = disparity_BMandSGBM(leftImageRaw, rightImageRaw, f, baseline, numDisparities, blockSize, scale, dis_filename)
depthBM, depthSGBM, colorDepthBM, colorDepthSGBM = depth_BMandSGBM(leftImageRaw, rightImageRaw, f, baseline, numDisparities, blockSize, scale,dep_filename, disparityBM, disparitySGBM)

