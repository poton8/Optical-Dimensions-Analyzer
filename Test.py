import numpy as np
import cv2
from matplotlib import pyplot as plt


CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
capL = cv2.VideoCapture(1)
capR = cv2.VideoCapture(2)
capL.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
stereoMatcher = cv2.StereoBM_create()



ret, imgL = capL.read()
ret2, imgR = capR.read()
grayLeft = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayRight = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
depth = stereoMatcher.compute(grayLeft, grayRight)
plt.imshow(depth, cmap='gray')
'''
plt.title('Left'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(imgR, cmap='gray')
plt.title('Right'), plt.xticks([]), plt.yticks([])'''

plt.show()