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


def region_of_interest(image):
    height = image.shape[0]
    roi = image[0:height, 100:400]
    fill = np.zeros_like(roi)
    return fill


def display_line(image, lines):
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return image


def canny_filter(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    imgThresh = cv.adaptiveThreshold(blur, 255.0, cv.ADAPTIVE_THRESH_MEAN_C,
                                     cv.THRESH_BINARY_INV, 21, 5)
    # canny = cv.Canny(blur, 0, 255)
    return imgThresh


def contour_func(image):
    draw_image = image.copy()
    contours, hierarchy = cv.findContours(
        image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours_image = cv.drawContours(draw_image, contours, -1, (0, 255, 0), 3)
    cv.imshow('contours_image', contours_image)
    return contours


def draw_rectangle(image, contours):
    contours_dict = []
    temp_result = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(temp_result, pt1=(x, y), pt2=(
            x+w, y+h), color=(255, 255, 0), thickness=2)
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x+(w/2),
            'cy': y+(h/2)
        })
    cv.imshow("contour", temp_result)
    return contours_dict


def select_pipe(contours_dict, image):
    temp_result = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    MIN_AREA = 0
    MIN_WIDTH, MIN_HEIGHT = 0, 0
    MIN_RATIO, MAX_RATIO = 1.5, 6

    possible_contour = []
    cnt = 0
    for d in contours_dict:
        area = d['w']*d['h']
        ratio = d['h']/d['w']
        print(ratio)
        if area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contour.append(d)
    for d in possible_contour:
        cv.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(
            d['x']+d['w'], d['y']+d['h']), color=(255, 255, 0), thickness=2)
    cv.imshow("contour2", temp_result)
    return None


def remove_noise(image):
    original = image.copy()
    image = canny_filter(image)
    contours = contour_func(image)
    contours_dict = draw_rectangle(image, contours)
    print(contours_dict)
    select_pipe(contours_dict, image)
    # lines = cv.HoughLinesP(image, 2, np.pi/180, 100,
    #                        maxLineGap=30)
    # line_image = display_line(original, lines)
    # image2 = region_of_interest(image)
    # plt.imshow(image)
    # plt.show()
    return image

# PATH = "/home/hgh/hgh/busan_image/Image_Test_Folder/gnsea03_2000702011635.png"
# remove_noise(PATH)


if __name__ == '__main__':

    global img
    files = glob.glob(
        '/home/hgh/hgh/busan_image/Image_original/gnsea03_*.png')
    files.sort()
    img = cv.imread(files[0])
    img = cv.resize(img, dsize=(0, 0),  fx=0.3,
                    fy=0.3, interpolation=cv.INTER_AREA)
    filesName = os.path.basename(files[0])
    img = remove_noise(img)

    print(filesName)
    # cv.imwrite(
    #     "/home/hgh/hgh/busan_image/20200821_result/"+filesName, img)
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
            # cv.imwrite(
            #     "/home/hgh/hgh/busan_image/20200821_result/"+filesName, img)
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
