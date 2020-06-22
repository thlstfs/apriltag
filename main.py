import pupil_apriltags as at
import cv2 as cv
import numpy as np
at_detector = at.Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)


cap = cv.VideoCapture(0)
with np.load('cam.npz') as X:
    mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

def drawPoly(img, pointsArray):
    length = len(pointsArray)
    for i in range(-1, length - 1):
        x1, y1 = pointsArray[i][0], pointsArray[i][1]
        x2, y2 = pointsArray[i+1][0], pointsArray[i+1][1]
        cv.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255 * ((i + 1) % 2),255 * ((i + 2) % 2)),2)


while(True):
    # print(rvecs)
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=[mtx[0][0],mtx[1][1], mtx[0][2], mtx[1][2]], tag_size=0.2)
    # print(len(frame[0]))
    if (tags):
        for tag in tags:
            drawPoly(frame, tag.corners)
            rotZ = tag.pose_R[0][0] / tag.pose_R[0][1]
            rotZ = np.arctan(rotZ)
            rotZ = np.rad2deg(rotZ)
            rotY = np.arcsin(tag.pose_R[0][2])
            rotY = np.rad2deg(rotY)
            rotX = tag.pose_R[1][2] / tag.pose_R[2][2]
            rotX= np.arctan(rotX)
            rotX = np.rad2deg(rotX)
            print("Rotation X axis ", rotX, " degrees")
            print("Rotation Y axis ", rotY, " degrees")
            print("Rotation Z axis ", rotZ, " degrees")
            print(tag.pose_t)
            print("------------------------------------------------------------")
                
    cv.imshow('frame',frame)
    # print(tags)
    if cv.waitKey(1) & 0xFF == ord('q'):    
        break

cap.release()
cv.destroyAllWindows()
