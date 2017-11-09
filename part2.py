import os, sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.optimize import minimize

# input_image = cv2.imread('radial_distortion/rosten_2008_camera-figure-4-b.tiff')
# input_image = cv2.resize(input_image, (853,640))

input_image = cv2.imread('radial_distortion/rosten_2008_camera.tiff')
cv2.imshow('input_image', input_image)

# ======= Canny Edges ======= 

# canny_edges_threshold1 = 100
# canny_edges_threshold2 = 200
# def update_canny_edges():
#     canny_edges = cv2.Canny(input_image, canny_edges_threshold1, canny_edges_threshold2)
#     cv2.imshow('canny_deges', canny_edges)
#     print('update_canny_edges', canny_edges_threshold1, canny_edges_threshold2)

# update_canny_edges()
# while 1:
#     key = cv2.waitKey(33)
#     if key == ord('a'):
#         canny_edges_threshold1 += 1
#         update_canny_edges()
#     elif key == ord('z'):
#         canny_edges_threshold1 -= 1
#         update_canny_edges()
#     elif key == ord('s'):
#         canny_edges_threshold2 += 1
#         update_canny_edges()
#     elif key == ord('x'):
#         canny_edges_threshold2 -= 1
#         update_canny_edges()
#     elif key == ord('\x1B'):
#         sys.exit()

cv2.namedWindow('canny_deges')
canny_edges = cv2.Canny(input_image, 245, 400)

# ======= Line Selection ======= 
lines = []

def add_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print('add_point({}, {})'.format(x, y), file = sys.stderr)
        if len(lines) == 0:
            lines.append(list())
        line = lines[-1]
        if len(line) != 0 and line[-1][0] >= x:
            line = list()
            lines.append(line)
        line.append((x, y))
        update_lines()
    elif event == cv2.EVENT_RBUTTONUP:
        if len(lines):
            del lines[-1]
        update_lines()
cv2.setMouseCallback('canny_deges', add_point)

_cmap_lines = plt.get_cmap('jet')
def cmap_lines(li):
    color = np.array(_cmap_lines(li/len(lines)))    # Apply color map
    color = color[:3]*255                           # Change color range
    return color.astype(int).tolist()               # Cast to Python types

def fit_int_line(sample_points):
    x, y = sample_points.T
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y)[0]
    return lambda x: int(a*x+b)

def update_lines():
    canny_edges_lines = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB)
    for li, line in enumerate(lines):
        if len(line):                               # Fit & render line
            func_line = fit_int_line(np.array(line))
            point1 = (line[0][0], func_line(line[0][0]))
            point2 = (line[-1][0], func_line(line[-1][0]))
            canny_edges_lines = cv2.line(canny_edges_lines, point1, point2, (255, 0, 255))
        for point in line:                          # Render points
            canny_edges_lines = cv2.circle(canny_edges_lines, point, 6, cmap_lines(li), -1)
    cv2.imshow('canny_deges', canny_edges_lines)

# ======= Undistortion =======
def cost(k):
    print('cost', k)
    error = 0.0
    for line in lines:
        if len(line):
            points = np.array(line)
            radial2 = np.linalg.norm(points, axis=1) ** 2
            undistortion = 1. + k[0] * radial2 + k[1] * (radial2 ** 2) + k[2] * (radial2 ** 3)
            undistorted = points * undistortion[..., np.newaxis]
            func_line = fit_int_line(np.array(undistorted))
            point1 = (undistorted[0][0], func_line(undistorted[0][0]))
            point2 = (undistorted[-1][0], func_line(undistorted[-1][0]))
            norm = np.array([point1[1]-point2[1], point2[0]-point1[0]]).astype(float)
            norm /= np.linalg.norm(norm).tolist()
            distances = np.abs(np.dot(np.array(undistorted) - np.array(undistorted[0])[np.newaxis, ...], norm))
            error += np.sum(distances**2)
    return error

update_lines()
while(1):
    key = cv2.waitKey(33) & 0xFF
    if key == ord('\x20'):
        k = minimize(cost, np.array([0.1] * 3), method='BFGS')
        try:
            print(cost(k), '@', k)
        except KeyError as e:

            print(e)
    elif key == ord('\x1B'):
        sys.exit()

cv2.waitKey(0)
sys.exit()
