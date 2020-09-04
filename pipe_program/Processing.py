# 가능한한 ROI를 최적으로 따고 나머지를 다 0인 이미지로 만들어서 grap컷시켜버림
#  DEFINE: 원본은 busanFinal(ROIgrab).py임
# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
import urllib
import os
from matplotlib import pyplot as plt
import imutils
import argparse
import glob
import os
import Processing_line
kernel = np.ones((3, 3), np.uint8)
kernel2 = np.ones((5, 9), np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)


def for_display(img):
    img = cv.resize(img, dsize=(250, 250), interpolation=cv.INTER_AREA)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def resizeing(image):
    global x
    global y
    img = cv.resize(image, dsize=(0, 0),  fx=0.3,
                    fy=0.3, interpolation=cv.INTER_AREA)
    x = img.shape[1]
    y = img.shape[0]
    return img


def read_image(PATH):
    image = cv.imread(PATH)
    return image


def region_on_interest(img2):
    return img2[110:365, 0:img2.shape[1]]


def save_region_on_interest(PATH, filename, image):
    roi_dir = PATH+"/roi"
    if os.path.isdir(roi_dir) == True:
        cv.imwrite(
            roi_dir+"/"+filename, image)
    else:
        os.mkdir(roi_dir)
        cv.imwrite(
            roi_dir+"/"+filename, image)


def save_mask_image(PATH, filename, image):
    mask_dir = PATH+"/mask"
    if os.path.isdir(mask_dir) == True:
        cv.imwrite(
            mask_dir+"/"+filename, image)
    else:
        os.mkdir(mask_dir)
        cv.imwrite(
            mask_dir+"/"+filename, image)


def save_line_image(PATH, filename, image):
    line_dir = PATH+"/line"
    if os.path.isdir(line_dir) == True:
        cv.imwrite(
            line_dir+"/"+filename, image)
    else:
        os.mkdir(line_dir)
        cv.imwrite(
            line_dir+"/"+filename, image)


def processing_init(PATH, filename):
    image = read_image(PATH+"/"+filename)
    img = resizeing(image)
    original_display = for_display(img)
    original = img.copy()
    img2 = original.copy()

    roi_image = region_on_interest(original)
    roi_display = for_display(roi_image)
    rect = (0, 110, x, 365)  # mask가 초기화 안되있을시에 사용됨
    save_region_on_interest(PATH, filename, roi_image)
    mask = np.zeros(original.shape[:2], np.uint8)
    try:
        cv.grabCut(original, mask, rect, bgdModel,
                   fgdModel, 5, cv.GC_INIT_WITH_RECT)
    except cv.error as e:
        print("GrabCut Error")
        return None, None
# DEFINE: 파이프를 기준으로 하단부 배경
    # 원본
    for i in range(365, y):  # y
        for j in range(0, x):  # x
            cv.circle(mask, (j, i), 3, cv.GC_BGD, -1)
# DEFINE: 파이프를 기준으로 상단부 배경
    # 원본
    for i in range(0, 110):
        for j in range(0, x):
            cv.circle(mask, (j, i), 3, cv.GC_BGD, -1)
    cv.grabCut(original, mask, rect, bgdModel,
               fgdModel, 1, cv.GC_INIT_WITH_MASK)
# DEFINE: 파이프(내부를 범위로 잡고)를 전경으로 설정

    for i in range(185, 295):
        for j in range(0, x):
            cv.circle(mask, (j, i), 3, cv.GC_FGD, -1)
    cv.grabCut(original, mask, rect, bgdModel,
               fgdModel, 1, cv.GC_INIT_WITH_MASK)
    img2[(mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD)] = 0
    mask_display = for_display(img2)
    line_image = Processing_line.line_processing_display(img2)
    save_mask_image(PATH, filename, img2)
    save_line_image(PATH, filename, line_image)
    img = cv.cvtColor(line_image, cv.COLOR_BGR2RGB)
    # cv.imshow("result", line_image)
    cv.waitKey()
    cv.destroyAllWindows()

    return img, original_display, roi_display, mask_display
