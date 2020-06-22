import cv2 as cv
import numpy as np

def setCamResolution(cap, width, height):
    cap.set(3,width)
    cap.set(4,height)

def cameraCalibrate(camId, width, height, output):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    ret1, mtx, dist, rvecs, tvecs = False, None, None, None, None
    
    cap = cv.VideoCapture(camId)
    setCamResolution(cap, width, height)
    
    while(True):
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,6), cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        print(corners)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(frame, (7,6), corners2, ret)
            # print([ret, mtx, dist, rvecs, tvecs])
        cv.imshow('img', frame)
        # print(ret1, mtx, dist, rvecs, tvecs)
        if cv.waitKey(1) & 0xFF == ord('q'):    
            break
    
    cap.release()
    cv.destroyAllWindows()
    ret1, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            
    if (ret1):
        np.savez(output, mtx = mtx, dist = dist, rvecs = rvecs, tvecs = tvecs)
        
    print("done")
    
cameraCalibrate(0, 960, 540, "cam")