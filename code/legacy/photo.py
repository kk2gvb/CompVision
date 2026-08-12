# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib
# import time
#
# def gradGrayImage(leftImageRaw, rightImageRaw):
#     leftImage = cv2.cvtColor(leftImageRaw, cv2.COLOR_BGR2GRAY)      # переводим в градации серого
#     rightImage = cv2.cvtColor(rightImageRaw, cv2.COLOR_BGR2GRAY)    #
#
#     return leftImage, rightImage
#
# def disparity_BMandSGBM(leftImageRaw, rightImageRaw, f, Baseline, numDisparities, blockSize, scale, figfile):
#
#     leftImage, rightImage = gradGrayImage(leftImageRaw, rightImageRaw)
#
#     stereoBM = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
#     disparityBM = stereoBM.compute(leftImage, rightImage)   # получаем карту несоответствий
#
#     stereoSGBM = cv2.StereoSGBM_create(numDisparities=numDisparities, blockSize=blockSize)
#     disparitySGBM = stereoSGBM.compute(leftImage, rightImage)   # получаем карту несоответствий
#
#     fig, (dis_bm, dis_sgbm) = plt.subplots(1, 2)
#
#     dis_bm.set_title('stereoBM')
#     dis_bm.imshow(disparityBM)
#
#     dis_sgbm.set_title('stereoSGBM')
#     dis_sgbm.imshow(disparitySGBM)
#
#     plt.suptitle("Карты несоответствий")
#     plt.tight_layout()
#     plt.savefig(figfile, dpi=300)
#     plt.show()
#
#     return disparityBM, disparitySGBM
#
#
# def depth_BMandSGBM(leftImageRaw, rightImageRaw, f, Baseline, numDisparities, blockSize, scale, figfile, disparityBM, disparitySGBM):
#
#     leftImage, rightImage = gradGrayImage(leftImageRaw, rightImageRaw)
#
#     validPixelsBM = disparityBM > 0     # избавляемся от деления на 0
#
#     depthBM = np.zeros(shape=leftImage.shape).astype("uint8")
#     depthBM[validPixelsBM] = (f * baseline) / (scale * disparityBM[validPixelsBM])
#
#     #Наводим красоты
#     depthBM = cv2.equalizeHist(depthBM)
#     colorDepthBM = np.zeros((leftImage.shape[0], leftImage.shape[1], 3), dtype="uint8")
#     tmp = cv2.applyColorMap(depthBM, cv2.COLORMAP_JET)
#     colorDepthBM[validPixelsBM] = tmp[validPixelsBM]
#
#     validPixelsSGBM = disparitySGBM > 0     # избавляемся от деления на 0
#
#     depthSGBM = np.zeros(shape=leftImage.shape).astype("uint8")
#     depthSGBM[validPixelsSGBM] = (f * baseline) / (scale * disparitySGBM[validPixelsSGBM])
#
#     #Наводим красоты
#     depthSGBM = cv2.equalizeHist(depthSGBM)
#     colorDepthSGBM = np.zeros((leftImage.shape[0], leftImage.shape[1], 3), dtype="uint8")
#     tmp = cv2.applyColorMap(depthSGBM, cv2.COLORMAP_JET)
#     colorDepthSGBM[validPixelsSGBM] = tmp[validPixelsSGBM]
#
#     fig, ((dep_bm, dep_sgbm), (color_dep_bm, color_dep_sgbm)) = plt.subplots(2, 2)
#
#     dep_bm.set_title(f'stereoBM')
#     dep_bm.imshow(depthBM)
#
#     color_dep_bm.set_title(f'stereoBM(color)')
#     color_dep_bm.imshow(colorDepthBM)
#
#     dep_sgbm.set_title(f'stereoSGBM')
#     dep_sgbm.imshow(depthSGBM)
#
#     color_dep_sgbm.set_title(f'stereoSGBM(color)')
#     color_dep_sgbm.imshow(colorDepthSGBM)
#
#     plt.suptitle("Карты глубин")
#     plt.tight_layout()
#     plt.savefig(figfile, dpi=300)
#     plt.show()
#
#     return depthBM, depthSGBM, colorDepthBM, colorDepthSGBM
#
#
# #============================Входные данные============================
#
# dis_filename = "../CV/figs/disparity_bm_vs_sgbm.png"
# dep_filename = "../CV/figs/depth_bm_vs_sgbm.png"
#
# f = 1040.0              # фокусное расстояние (в пикселях)
# baseline = 85           # расстояние между камерами
# numDisparities = 128    # параметры поиска несоответсвий
# blockSize = 25          # Размер блока который смотрит соответствие
# scale = 0.8             # соотношение условных единиц глубины с реальными
#
#
# leftImageRaw = cv2.imread('../RobotDogStereoDemo/opencv_samples/aloeL.jpg')      # тестовые изображения
# rightImageRaw = cv2.imread('../RobotDogStereoDemo/opencv_samples/aloeR.jpg')     #
#
# disparityBM, disparitySGBM = disparity_BMandSGBM(leftImageRaw, rightImageRaw, f, baseline, numDisparities, blockSize, scale, dis_filename)
# depthBM, depthSGBM, colorDepthBM, colorDepthSGBM = depth_BMandSGBM(leftImageRaw, rightImageRaw, f, baseline, numDisparities, blockSize, scale,dep_filename, disparityBM, disparitySGBM)
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

    # 1. StereoBM
    stereoBM = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparityBM = stereoBM.compute(leftImage, rightImage)   # получаем карту несоответствий

    # 2. StereoSGBM
    stereoSGBM = cv2.StereoSGBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparitySGBM = stereoSGBM.compute(leftImage, rightImage)   # получаем карту несоответствий

    # 3. StereoSGBM + WLS (Добавлено)
    # Создаем matcher для правого изображения (нужен для проверки согласованности WLS)
    right_matcher = cv2.ximgproc.createRightMatcher(stereoSGBM)
    disparitySGBM_right = right_matcher.compute(rightImage, leftImage)

    # Создаем и настраиваем WLS фильтр
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereoSGBM)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)

    # Применяем фильтр
    disparityWLS = wls_filter.filter(disparitySGBM, leftImage, None, disparitySGBM_right)

    # Визуализация (Теперь 1x3)
    fig, (dis_bm, dis_sgbm, dis_wls) = plt.subplots(1, 3, figsize=(15, 5))

    dis_bm.set_title('StereoBM')
    dis_bm.imshow(disparityBM, cmap='gray')

    dis_sgbm.set_title('StereoSGBM')
    dis_sgbm.imshow(disparitySGBM, cmap='gray')

    dis_wls.set_title('StereoSGBM + WLS')
    # disparityWLS имеет тип float32, нормализуем для корректного отображения
    disparityWLS_viz = cv2.normalize(disparityWLS, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    dis_wls.imshow(disparityWLS_viz, cmap='gray') 

    plt.suptitle("Сравнение методов (BM, SGBM, SGBM+WLS)")
    plt.tight_layout()
    plt.savefig(figfile, dpi=300)
    plt.show()

    return disparityBM, disparitySGBM, disparityWLS

def depth_BMandSGBM(leftImageRaw, rightImageRaw, f, Baseline, numDisparities, blockSize, scale, figfile, disparityBM, disparitySGBM, disparityWLS):
    leftImage, rightImage = gradGrayImage(leftImageRaw, rightImageRaw)

    # --- StereoBM Depth ---
    validPixelsBM = disparityBM > 0     # избавляемся от деления на 0
    depthBM = np.zeros(shape=leftImage.shape).astype("uint8")
    depthBM[validPixelsBM] = (f * Baseline) / (scale * disparityBM[validPixelsBM].astype(np.float32))

    # Наводим красоту
    depthBM = cv2.equalizeHist(depthBM)
    colorDepthBM = np.zeros((leftImage.shape[0], leftImage.shape[1], 3), dtype="uint8")
    tmp = cv2.applyColorMap(depthBM, cv2.COLORMAP_JET)
    colorDepthBM[validPixelsBM] = tmp[validPixelsBM]

    # --- StereoSGBM Depth ---
    validPixelsSGBM = disparitySGBM > 0     # избавляемся от деления на 0
    depthSGBM = np.zeros(shape=leftImage.shape).astype("uint8")
    depthSGBM[validPixelsSGBM] = (f * Baseline) / (scale * disparitySGBM[validPixelsSGBM].astype(np.float32))

    # Наводим красоту
    depthSGBM = cv2.equalizeHist(depthSGBM)
    colorDepthSGBM = np.zeros((leftImage.shape[0], leftImage.shape[1], 3), dtype="uint8")
    tmp = cv2.applyColorMap(depthSGBM, cv2.COLORMAP_JET)
    colorDepthSGBM[validPixelsSGBM] = tmp[validPixelsSGBM]


    #TODO: Надо сделать чтобы и на фотке работал нормально WLS

    # --- StereoSGBM+WLS Depth (Добавлено) ---
    # disparityWLS - это float32, делить на 16 не нужно (зависит от версии, но обычно фильтр выдает реальные значения)
    # Если глубина получается странной, возможно, стоит поделить disparityWLS на 16.
    validPixelsWLS = disparityWLS > 0
    depthWLS = np.zeros(shape=leftImage.shape).astype("float32")
    depthWLS[validPixelsWLS] = (f * Baseline) / (scale * disparityWLS[validPixelsWLS])

    # Нормализация для визуализации
    depthWLS_viz = cv2.normalize(depthWLS, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    colorDepthWLS = np.zeros((leftImage.shape[0], leftImage.shape[1], 3), dtype="uint8")
    tmp = cv2.applyColorMap(depthWLS_viz, cv2.COLORMAP_JET)
    colorDepthWLS[validPixelsWLS] = tmp[validPixelsWLS]

    # Визуализация (Теперь 2x3)
    fig, ((dep_bm, dep_sgbm, dep_wls), (color_dep_bm, color_dep_sgbm, color_dep_wls)) = plt.subplots(2, 3, figsize=(18, 10))

    dep_bm.set_title('Depth BM')
    dep_bm.imshow(depthBM, cmap='gray')

    color_dep_bm.set_title('Color Depth BM')
    color_dep_bm.imshow(colorDepthBM)

    dep_sgbm.set_title('Depth SGBM')
    dep_sgbm.imshow(depthSGBM, cmap='gray')

    color_dep_sgbm.set_title('Color Depth SGBM')
    color_dep_sgbm.imshow(colorDepthSGBM)

    dep_wls.set_title('Depth SGBM+WLS')
    dep_wls.imshow(depthWLS_viz, cmap='gray')

    color_dep_wls.set_title('Color Depth SGBM+WLS')
    color_dep_wls.imshow(colorDepthWLS)

    plt.suptitle("Карты глубин (BM, SGBM, SGBM+WLS)")
    plt.tight_layout()
    plt.savefig(figfile, dpi=300)
    plt.show()

    return depthBM, depthSGBM, depthWLS, colorDepthBM, colorDepthSGBM, colorDepthWLS

#============================Входные данные============================
dis_filename = "../CV/figs/disparity_bm_vs_sgbm.png"
dep_filename = "../CV/figs/depth_bm_vs_sgbm.png"

f = 1040.0              # фокусное расстояние (в пикселях)
baseline = 85           # расстояние между камерами
numDisparities = 128    # параметры поиска несоответсвий
blockSize = 25          # Размер блока который смотрит соответствие
scale = 0.8             # соотношение условных единиц глубины с реальными

leftImageRaw = cv2.imread('../RobotDogStereoDemo/opencv_samples/aloeL.jpg')      # тестовые изображения
rightImageRaw = cv2.imread('../RobotDogStereoDemo/opencv_samples/aloeR.jpg')     #

# Вызов функций (теперь возвращают 3 значения)
disparityBM, disparitySGBM, disparityWLS = disparity_BMandSGBM(leftImageRaw, rightImageRaw, f, baseline, numDisparities, blockSize, scale, dis_filename)

depthBM, depthSGBM, depthWLS, colorDepthBM, colorDepthSGBM, colorDepthWLS = depth_BMandSGBM(
    leftImageRaw, rightImageRaw, f, baseline, numDisparities, blockSize, scale, dep_filename, 
    disparityBM, disparitySGBM, disparityWLS
)
