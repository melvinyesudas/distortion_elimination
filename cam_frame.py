import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from sklearn.externals import joblib
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

count = 0
save_count = 0
os.chdir("C:/Users/melvin/Desktop/images")
images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow("mekr",img)
    img = cv2.resize(img, (640, 480))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    balanced = cv2.equalizeHist(gray)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(balanced, (9, 7), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 7), corners2, ret)
        #cv2.imshow("img",img)
        #cv2.waitKey()
        
        
        # Save the result for "output_images" folder
        count += 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('./output_images/'+ str(count) + '_corners'+'.jpg', img)


#pickle_out = open("objpoints.pickle", "wb")
#pickle_out1 = open("imgpoints.pickle", "wb")

joblib.dump(objpoints, "pickle_out.pkl")
joblib.dump(imgpoints, "pickle_out1.pkl")

print('Image and object points are saved in pickle file for later use.')