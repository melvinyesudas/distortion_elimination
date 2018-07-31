import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from sklearn.externals import joblib


def cal_undistort(img, objpoints, imgpoints):
    img = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst


#img = cv2.imread('test.jpg')
#resize = cv2.resize(img, (640, 480))
#undistorted = cal_undistort(resize, objpoints, imgpoints)
#
## Display the original image and undistorted image
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()
#
#ax1.imshow(resize, cmap = 'gray')
#ax1.set_title('Original Image', fontsize=40)
#
#ax2.imshow(undistorted, cmap='gray')
#ax2.set_title('Undistorted Image', fontsize=40)
#
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#plt.show()
#
#undistorted = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
#cv2.imwrite('./output_images/Undistorted_image.jpg', undistorted)
#
#print('Undistorted image saved to "output_images" folder')

imgpoints = joblib.load("pickle_out1.pkl")

objpoints = joblib.load("pickle_out.pkl")

img = cv2.imread('test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 480))

img = cal_undistort(img, objpoints, imgpoints)
cv2.waitKey(500)
    
# Save the result for "output_images" folder
#save_count += 1
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('./output_images/'+ str(78) + '_undistorted'+'.jpg', img)

print('Undistorted images saved to "output_images" folder')

