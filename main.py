import pupil_apriltags as at
import cv2 as cv
import numpy as np
import threading
import serial
import time
ser = serial.Serial('COM7',baudrate=9600, timeout=1)
ser.isOpen()

detectedTag = None


def rotateCamera(rotation):
    global ser
    rotY = np.rad2deg(rotation[1][0])
    angle = -int(rotY)
    angle += 91
    angle = str(angle)
    ser.write(angle.encode('ascii'))
    #time.sleep(0.1)
    print(angle)

def zoomIn(frame, center, width, height):
    global detectedTag
    i = (height / 10) / (detectedTag.pose_t[2] + 1)
    print(detectedTag.pose_t[2])
    xMax = int(center[0][0][0] + i)
    xMin = int(center[0][0][0] - i)
    yMax = int(center[0][0][1] + i)
    yMin = int(center[0][0][1] - i)
    if (xMax > width):
        xMax = width
    if (xMin < 0):
        xMin = 0
    if (yMax > height):
        yMax = height
    if (yMin < 0):
        yMin = 0
    return frame[yMin:yMax, xMin:xMax]

def calculateRotation(tagPose, rotation, translation, mtx, dist, width, height, precision):
    imgpts, jac = cv.projectPoints(tagPose, rotation, translation, mtx, dist)
    center = imgpts
    prevRotation = rotation[1][0]
    hMin = 0
    hMax = height
    wMin = 2 * (width / 5)
    wMax = width - wMin
    rotY = np.rad2deg(rotation[1][0])
    rad = np.deg2rad(precision)
    while True:
        imgpts, jac = cv.projectPoints(tagPose, rotation, translation, mtx, dist)
        center = imgpts
        #print(imgpts)
        if (center[0][0][0] > wMax):
            rotation[1][0] -= rad
        elif (center[0][0][0] < wMin):
            rotation[1][0] += rad
        else:
            break    
    if (prevRotation != rotation[1][0]):
        ret = True
    else:
        ret = False
    return rotation, imgpts, jac, ret

def drawBorders(img, corners):
    length = len(corners)
    for i in range(-1, length - 1):
        x1, y1 = int(corners[i][0]), int(corners[i][1])
        x2, y2 = int(corners[i+1][0]), int(corners[i+1][1])
        cv.line(img,(x1,y1),(x2,y2),(0,255 * ((i + 1) % 2),255 * ((i + 2) % 2)),2)

    
def cam2():
    global detectedTag
    cap = cv.VideoCapture(3)
    setCamResolution(cap, 1920, 1080)
    with np.load('cam1.npz') as X:
        mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    while(True):
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        data, bbox, _ = detectQRcode(gray)
        focusTheTag(data)        
        # if (not data):
        #     print("zoom in")
        cv.imshow('frame1',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):    
            break
    
    cap.release()
    cv.destroyAllWindows()
    

def setCamResolution(cap, width, height):
    cap.set(3,width)
    cap.set(4,height)

def focusTheTag(data):
    if (not data):
        print("focusing")
        return False
    else:
        return True

def getRotationInfo(tagPoseR):
    rotZ = tagPoseR[0][0] / tagPoseR[0][1]
    rotZ = np.arctan(rotZ)
    rotZ = np.rad2deg(rotZ)
    rotY = np.arcsin(-tagPoseR[0][2])
    rotY = np.rad2deg(rotY)
    rotX = tagPoseR[1][2] / tagPoseR[2][2]
    rotX= np.arctan(rotX)
    rotX = np.rad2deg(rotX)
    return -rotX, -rotY, -rotZ
    
def detectQRcode(grayImg):
    detector = cv.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(grayImg)
    if data:
        print("QR Code detected-->", data)
    return data, bbox, _ 

def getCamIDs():
    index = 0
    arr = []
    while True:
        cap = cv.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    print(arr)
    return arr

def cam1():
    global detectedTag
    at_detector = at.Detector(families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)
    
    
    cap = cv.VideoCapture(0)
    setCamResolution(cap, 960, 540)
    with np.load('webcam540p.npz') as X:
        mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    
    while(True):
        # print(rvecs)
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=[mtx[0][0],mtx[1][1], mtx[0][2], mtx[1][2]], tag_size=0.182)
        #print(len(frame))
        if (tags):
            for tag in tags:
                detectedTag = tag
                # print(tag)
                drawBorders(frame, tag.corners)
                rotX, rotY, rotZ = getRotationInfo(tag.pose_R)
                # print("Rotation: ", rotX, ",", rotY, ",", rotZ)
                # print(tag.pose_t)
                print("------------------------------------------------------------")
                # t = np.float32([[0], [0], [0]])
                
                # R = np.float32([[0], [-1], [0]])
                # imgpts, jac = cv.projectPoints(detectedTag.pose_t, R, t, mtx, dist)
                # print(imgpts)
                # x1, y1 = int(imgpts[0][0][0]), int(imgpts[0][0][1])
                # x2, y2 = int(imgpts[0][0][0]), int(imgpts[0][0][1])
                # cv.line(frame,(x1,y1),(x2,y2),(0,255 * ((0 + 1) % 2),255 * ((0 + 2) % 2)),2)
                # rotation_mat = np.zeros(shape=(3, 3))
                # R = cv.Rodrigues(rvecs[0], rotation_mat)[0]
                # P = np.column_stack((np.matmul(mtx,R), tvecs[0]))  
                # M = mtx.dot(P)
                # uv1 = M.dot(cam1Vec)
                # uv = P.dot(cam1Vec)
                # x = mtx[0][0] * (uv[0] / uv[2]) + mtx[0][2]
                # print(uv)
                # print(x)
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):    
            break
    
    cap.release()
    cv.destroyAllWindows()
def cam3():
    global detectedTag
    at_detector = at.Detector(families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)
    
    
    cap = cv.VideoCapture(2)
    setCamResolution(cap, 1920, 1080)
    with np.load('cam1.npz') as X:
        mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
   
    R = np.float32([[0], [0], [0]])
    while(True):
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=[mtx[0][0],mtx[1][1], mtx[0][2], mtx[1][2]], tag_size=0.182)
        #print(len(frame))
        if (tags):
            for tag in tags:
                
                drawBorders(frame, tag.corners)
                rotX, rotY, rotZ = getRotationInfo(tag.pose_R)
                
                # print("Rotation2: ", rotX, ",", rotY, ",", rotZ)
                # print(tag.pose_t)
                # print("------------------------------------------------------------")
                cam2Vec = tag.pose_t
                cam2Rot = [rotX, rotY, rotZ]
        if (detectedTag):
            t = np.float32([[0.34], [-0.03], [0]])
            R, imgpts, jac, r = calculateRotation(detectedTag.pose_t, R, t, mtx, dist, 1920, 1080, 0.01)
            #print(imgpts)
            if (r):
                rotateCamera(R)
            x1, y1 = int(imgpts[0][0][0]), int(imgpts[0][0][1])
            x2, y2 = int(imgpts[0][0][0]), int(imgpts[0][0][1])
            cv.line(frame,(x1,y1),(x2,y2),(0,255 * ((0 + 1) % 2),255 * ((0 + 2) % 2)),2)
            frame2 = zoomIn(frame, imgpts, 1920, 1080)
            cv.imshow('zoom',frame2)
        cv.imshow('frame2',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):    
            break
    
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    getCamIDs()

    t1 = threading.Thread(target=cam1)
    t2 = threading.Thread(target=cam3)
    # starting thread 1 
    t1.start() 
    # starting thread 2 
    t2.start() 
  
    # wait until thread 1 is completely executed 
    t1.join() 
    # wait until thread 2 is completely executed 
    t2.join() 
    # both threads completely executed 
    ser.close()
    print("Done!")
