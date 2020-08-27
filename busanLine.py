# 가능한한 ROI를 최적으로 따고 나머지를 다 0인 이미지로 만들어서 grap컷시켜버림

# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import urllib
import imutils
import math
from itertools import permutations
import glob
import os
kernel = np.ones((3, 3), np.uint8)
kernel2 = np.ones((5, 9), np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)


def make_coordinates(image, parameter):
    기울기, 절편 = parameter
    x1 = 0
    x2 = image.shape[1]
    y1 = int(기울기*x1+절편)
    y2 = int(기울기*x2+절편)
    return np.array([x1, y1, x2, y2])


def calculate_angle(angles, lines):
    각도 = 0
    top_degree_height = math.atan2(
        (lines[0][0] - angles[0]), (lines[0][1] - angles[1]))
    top_degree_bottom = math.atan2(
        (angles[0]-lines[2][0]), (angles[1]-lines[2][1]))
    degree = (top_degree_height-top_degree_bottom)*180/math.pi
    if abs(degree) > 180:
        각도 = abs(int(degree))-180
    else:
        각도 = 180-abs(degree)
    return 각도


def find_crossing(image, lines):
    t_X = 0
    t_Y = 0
    newlines = list(permutations(lines, len(lines)))
    for j in range(len(newlines)):
        for i in range(len(lines[1:])):
            a1 = newlines[j][i][3] - newlines[j][i][1]
            b1 = newlines[j][i][0] - newlines[j][i][2]
            c1 = a1*newlines[j][i][0] + b1*newlines[j][i][1]

            a2 = newlines[j][i+1][3] - newlines[j][i+1][1]
            b2 = newlines[j][i+1][0] - newlines[j][i+1][2]
            c2 = a2*newlines[j][i+1][0] + b2*newlines[j][i+1][1]

            deteminate = a1*b2 - a2*b1
            if deteminate != 0:
                try:
                    t_X = (b2*c1 - b1*c2)/deteminate
                    t_Y = (a1*c2 - a2*c1)/deteminate
                except ZeroDivisionError:
                    t_X = (b2*c1 - b1*c2)/1
                    t_Y = (a1*c2 - a2*c1)/1
        if t_X > 1 and t_X < image.shape[1] and t_Y > 1 and t_Y < image.shape[0]:
            return np.array([t_X, t_Y])
    else:
        return


def find_slop(x1, y1, x2, y2):
    return (y2-y1)/(x2-x1)


def display_crossing_point(image, points):
    x, y = points.reshape(2)
    cv.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)
    return image


def average_lines(image, lines):  # 직선선분만 찾기
    top_fit = []
    bottom_fit = []
    curve_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            다항식피팅 = np.polyfit((x1, x2), (y1, y2), 1)  # 최소제곱 다항식 곡선
            기울기 = 다항식피팅[0]
            절편 = 다항식피팅[1]
            slop = find_slop(x1, y1, x2, y2)
            # print(slop)
            if 절편 < 250 and abs(slop) < 0.1 and x1 < 350:
                top_fit.append((기울기, 절편))
            elif 절편 > 250 and abs(slop) < 0.1 and x1 < 350:
                bottom_fit.append((기울기, 절편))  # null일 경우에 문제 발생
            elif abs(slop) > 0.06 and x1 > 350 and y1 < 250:  # FIXME:곡관이 시작하는 기준점도 정해야할듯
                curve_fit.append((기울기, 절편))
        상단_라인_평균 = np.average(top_fit, axis=0)
        하단_라인_평균 = np.average(bottom_fit, axis=0)
        하단_라인 = make_coordinates(image, 하단_라인_평균)
        상단_라인 = make_coordinates(image, 상단_라인_평균)
        if curve_fit is not None:
            try:
                커브_라인_평균 = np.average(curve_fit, axis=0)
                커브_라인 = make_coordinates(image, 커브_라인_평균)
                return np.array([상단_라인, 하단_라인, 커브_라인])
            except:
                print("커브라인을 찾을 수 없습니다.")
        return np.array([상단_라인, 하단_라인])


def Canny_filter(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, rthresh1 = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    result = cv.dilate(rthresh1, kernel, iterations=9)
    result = cv.erode(result, kernel, iterations=10)
    edges = cv.Canny(result, 50, 150)
    return edges


def display_line(image, lines):
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return image


def remove_noise(PATH):
    image = cv.imread(PATH)
    # img = cv.resize(image, dsize=(0, 0),  fx=0.3,
    #                 fy=0.3, interpolation=cv.INTER_AREA)
    img = image
    original = img.copy()
    edges = Canny_filter(original)
    lines = cv.HoughLinesP(edges, 2, np.pi/180, 100,
                           maxLineGap=30)
    average_line = average_lines(original, lines)
    line_image = display_line(original, average_line)
    # test = display_line(original, lines)
    this_is_curve = find_crossing(original, average_line)
    if this_is_curve is None:
        print("직관")
    else:
        display_crossing_point(original, this_is_curve)
        degree = calculate_angle(this_is_curve, average_line)
        print("곡관 :", degree)
    # cv.imwrite(
    #     "/home/hgh/hgh/busan_image/busan_3_line.jpeg", original)

    # img = cv.resize(img, dsize=(0, 0),  fx=0.3,
    cv.imshow("og", line_image)
    # cv.imshow("test", test)
    # return line_image
    #                 fy=0.3, interpolation=cv.INTER_AREA)
    # plt.show()
    cv.waitKey()
    cv.destroyAllWindows()


PATH = "/home/hgh/hgh/busan_image/Image_result_Folder/gnsea03_20200702111635.png"
remove_noise(PATH)


# if __name__ == '__main__':

#     # load the image and setup the mouse callback function
#     global img
#     files = glob.glob(
#         '/home/hgh/hgh/busan_image/Image_result_Folder/gnsea03_*.png')

#     # files = glob.glob('/home/hgh/hgh/project/busan_project/b*.jpg')
#     files.sort()
#     img = cv.imread(files[0])
#     # img = cv.resize(img, (400, 400))
#     # img = cv.resize(img, dsize=(0, 0),  fx=0.3,
#     #                 fy=0.3, interpolation=cv.INTER_AREA)
#     img = remove_noise(img)
#     filesName = os.path.basename(files[0])
#     # print(filesName)
#     cv.imwrite(
#         "/home/hgh/hgh/busan_image/Image_Line_result/"+filesName, img)
#     cv.imshow(filesName, img)

#     # Create an empty window
#     # cv.namedWindow('PRESS P for Previous, N for Next Image')
#     # Create a callback function for any event on the mouse
#     # cv.setMouseCallback(
#     #     'PRESS P for Previous, N for Next Image', showPixelValue)

#     i = 0

#     while(1):
#         k = cv.waitKey(1) & 0xFF
#         # check next image in the folder
#         if k == ord('n'):
#             i += 1
#             img = cv.imread(files[i % len(files)])
#             # img = cv.resize(img, (400, 400))
#             # img = cv.resize(img, dsize=(0, 0),  fx=0.3,
#             #                 # fy=0.3, interpolation=cv.INTER_AREA)
#             img = remove_noise(img)
#             filesName = os.path.basename(files[i % len(files)])
#             print(filesName)
#             cv.imwrite(
#                 "/home/hgh/hgh/busan_image/Image_Line_result/"+filesName, img)
#             cv.imshow(filesName, img)

#         # check previous image in folder
#         elif k == ord('p'):
#             i -= 1
#             img = cv.imread(files[i % len(files)])
#             # img = cv.resize(img, (400, 400))
#             # img = cv.resize(img, dsize=(0, 0),  fx=0.3,
#             #                 fy=0.3, interpolation=cv.INTER_AREA)
#             img = remove_noise(img)
#             cv.imshow('PRESS P for Previous, N for Next Image', img)

#         elif k == 27:
#             cv.destroyAllWindows()
#             break
