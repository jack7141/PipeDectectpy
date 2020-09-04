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
kernel = np.ones((3, 3), np.uint8)
kernel2 = np.ones((5, 9), np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)


def remove_noise(image):
    y = image.shape[0]
    x = image.shape[1]
    # image = cv.imread(PATH)
    # img = cv.resize(image, dsize=(0, 0),  fx=0.3,
    #                 fy=0.3, interpolation=cv.INTER_AREA)
    img = image
    original = img.copy()
    # 상단부 ROI
    roi = img[0:100, 0:y]
    roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    ori_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    roihist = cv.calcHist([roi_hsv], [0, 1], None, [
                          180, 256], [0, 180, 0, 256])

    dst = cv.calcBackProject([ori_hsv], [0, 1], roihist, [0, 180, 0, 256], 1)

    # Now convolute with circular disc
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    cv.filter2D(dst, -1, disc, dst)
    thresh = cv.threshold(dst, 50, 255, 0, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    bitwise_not = cv.bitwise_not(thresh)
    erode = cv.erode(bitwise_not, kernel, iterations=5)
    dilate = cv.dilate(erode, kernel2, iterations=11)
    thresh = cv.merge((dilate, dilate, dilate))
    res = cv.bitwise_and(img, thresh)
    # img = img - res
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    res = cv.threshold(gray, 50, 255, 0)[1]
    contours, _ = cv.findContours(
        dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img2 = original.copy()
    # DEFINE: ROI잡으려고 설정해둔 코드임 boundingRect까지
    # c = max(contours, key=cv.contourArea)
    # x, y, w, h = cv.boundingRect(c)

    # 원본
    roi_image = img2[160:580, 0:img.shape[1]]
    # mask가 초기화 안되있을시에 사용됨 rect값으로 roi의 지정된 값을 인식시켜줘야하는듯?
    rect = (0, 160, img.shape[1], 580)
    cv.imshow("roi", roi_image)

    mask = np.zeros(original.shape[:2], np.uint8)
    try:
        cv.grabCut(original, mask, rect, bgdModel,
                   fgdModel, 5, cv.GC_INIT_WITH_RECT)
    except cv.error as e:
        print("GrabCut Error")
        return None, None
# DEFINE: 파이프를 기준으로 하단부 배경
    # 원본
    for i in range(580, img.shape[0]):  # y
        for j in range(0, img.shape[1]):  # x
            cv.circle(mask, (j, i), 3, cv.GC_BGD, -1)
# DEFINE: 파이프를 기준으로 상단부 배경
    # 원본
    for i in range(0, 160):
        for j in range(0, img.shape[1]):
            cv.circle(mask, (j, i), 3, cv.GC_BGD, -1)
    cv.grabCut(original, mask, rect, bgdModel,
               fgdModel, 1, cv.GC_INIT_WITH_MASK)
# DEFINE: 파이프(내부를 범위로 잡고)를 전경으로 설정

    for i in range(220, 530):
        for j in range(0, img.shape[1]):
            cv.circle(mask, (j, i), 3, cv.GC_FGD, -1)
    cv.grabCut(original, mask, rect, bgdModel,
               fgdModel, 1, cv.GC_INIT_WITH_MASK)
    img2[(mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD)] = 0
    # cv.imshow("result", img2)
    # cv.imwrite(
    #     "/home/hgh/hgh/busan_image/20200821_result/gnsea03 _20200821131832.png", img2)
    # cv.imshow("b23_ori.jpg", original)
    # plt.imshow(original)
    # plt.show()
    # cv.waitKey()
    # cv.destroyAllWindows()
    return img2


# PATH = "/home/hgh/hgh/busan_image/Image_Test_Folder/gnsea03_2000702011635.png"
# remove_noise(PATH)

# gnsea03_20200702011635
if __name__ == '__main__':

    # load the image and setup the mouse callback function
    global img
    files = glob.glob(
        '/home/hgh/hgh/busan_image/sample/gnsea05_*.png')

    # files = glob.glob('/home/hgh/hgh/project/busan_project/b*.jpg')
    files.sort()
    img = cv.imread(files[0])
    # img = cv.resize(img, (400, 400))
    img = cv.resize(img, dsize=(0, 0),  fx=0.3,
                    fy=0.3, interpolation=cv.INTER_AREA)
    filesName = os.path.basename(files[0])
    img = remove_noise(img)
    print(filesName)
    cv.imwrite(
        "/home/hgh/hgh/busan_image/sample_result/"+filesName, img)
    cv.imshow('PRESS P for Previous, N for Next Image', img)
    # Create an empty window
    cv.namedWindow('PRESS P for Previous, N for Next Image')
    # Create a callback function for any event on the mouse
    # cv.setMouseCallback(
    #     'PRESS P for Previous, N for Next Image', showPixelValue)

    i = 0

    while(1):
        k = cv.waitKey(1) & 0xFF
        # check next image in the folder
        if k == ord('n'):
            i += 1
            img = cv.imread(files[i % len(files)])
            # img = cv.resize(img, (400, 400))
            img = cv.resize(img, dsize=(0, 0),  fx=0.3,
                            fy=0.3, interpolation=cv.INTER_AREA)
            img = remove_noise(img)
            filesName = os.path.basename(files[i % len(files)])
            cv.imwrite(
                "/home/hgh/hgh/busan_image/sample_result/"+filesName, img)
            cv.imshow('PRESS P for Previous, N for Next Image', img)

        # check previous image in folder
        elif k == ord('p'):
            i -= 1
            img = cv.imread(files[i % len(files)])
            # img = cv.resize(img, (400, 400))
            img = cv.resize(img, dsize=(0, 0),  fx=0.3,
                            fy=0.3, interpolation=cv.INTER_AREA)
            img = remove_noise(img)
            cv.imshow('PRESS P for Previous, N for Next Image', img)

        elif k == 27:
            cv.destroyAllWindows()
            break
