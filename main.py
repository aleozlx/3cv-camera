import numpy as np
import cv2, glob
from matplotlib import pyplot as plt

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

pattern = (12, 13)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((np.product(pattern),3), np.float32)
objp[:,:2] = np.mgrid[:pattern[0],:pattern[1]].T.reshape(-1,2)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.tif')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, pattern, corners2, ret)
        cv2.imshow(fname, img)
        cv2.waitKey(500)

# cv2.destroyAllWindows()

# NA(3,3)A(1,5)L4(3,1)L4(3,1)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

bold = '\x1B[1m{}\x1B[0m'.format
print(bold('Camera matrix'))
print(mtx)
print(bold('Distortion'))
print(dist)
print(bold('Rotation vectors'))
for v in rvecs:
    print(v)
print(bold('Translation vectors'))
for v in tvecs:
    print(v)

h,w=img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imshow('result', dst)

cv2.waitKey(1000)

# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html

"""
x=PX homography
H=K[r1 r2 t] 

"""
