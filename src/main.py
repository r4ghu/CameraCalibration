import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os
import pickle

def calibrateCamera():
    # Check if calibration is already done
    calib_file_loc = './data/calib.pkl'
    if not os.path.exists(calib_file_loc):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('data/GO*.jpg')

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (8, 6), corners, ret)
                # write_name = 'corners_found'+str(idx)+'.jpg'
                # cv2.imwrite(write_name, img)
                cv2.imshow('img', img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

        img_size = (img.shape[1], img.shape[0])

        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        # Save Calibration file
        with open(calib_file_loc,'wb') as f:
            data = {'K': mtx, 'D': dist}
            pickle.dump(data, f)
    else:
        with open(calib_file_loc,'rb') as f:
            data = pickle.load(f)

    # Test undistortion on an image
    img = cv2.imread('data/test_image.jpg')
    dst = cv2.undistort(img, data['K'], data['D'], None, data['K'])
    plt.imshow(np.hstack((img,dst)))
    plt.show()

if __name__ == '__main__':
    calibrateCamera()