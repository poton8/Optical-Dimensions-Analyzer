import numpy as np
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import argparse
import sys
import os


focalLength = 525.0
centerX = 319.5
centerY = 239.5
scalingFactor = 5000.0

#imgL = cv2.imread('calibratedleft1.png')
#imgR = cv2.imread('calibratedright1.png')
capL = cv2.VideoCapture(1)
capR = cv2.VideoCapture(2)
ret, imgL = capL.read()
ret2, imgR = capR.read()
cv2.imwrite('rgbImage.png',imgL)
mtx = np.load('mtx.npy')
dist = np.load('dist.npy')
newcameramtx = np.load('newcameramtx.npy')
roi = np.load('roi.npy')
x,y,w,h = roi
dstL = cv2.undistort(imgL, mtx, dist, None, newcameramtx)
dstR = cv2.undistort(imgR, mtx, dist, None, newcameramtx)
dstL = dstL[y:y + h, x:x + w]
dstR = dstR[y:y + h, x:x + w]
dstL = cv2.cvtColor(dstL,cv2.COLOR_BGR2GRAY)
dstR = cv2.cvtColor(dstR,cv2.COLOR_BGR2GRAY)


# SGBM Parameters -----------------
window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=256,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=5,
    P1=8 * 3 * window_size ** 2,
    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)


# FILTER Parameters
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)


print('computing disparity...')
displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
filteredImg = np.uint8(filteredImg)
cv2.imshow('Disparity Map', filteredImg)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite('depth-small.png',filteredImg)


rgb = Image.open('rgbImage.png')
depth = Image.open('depth-small.png')

if rgb.size != depth.size:
    raise Exception("Color and depth image do not have the same resolution.")
if rgb.mode != "RGB":
    raise Exception("Color image is not in RGB format")


points = []
for v in range(rgb.size[1]):
    for u in range(rgb.size[0]):
        color = rgb.getpixel((u, v))
        Z = depth.getpixel((u, v)) / scalingFactor
        if Z == 0: continue
        X = (u - centerX) * Z / focalLength
        Y = (v - centerY) * Z / focalLength
        points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
file = open('ply_file.ply', "w")
file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
file.close()