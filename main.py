import pupil_apriltags as at
import cv2 as cv
import numpy as np
import threading

detectedTag = None

def drawBorders(img, corners):
    length = len(corners)
    for i in range(-1, length - 1):
        x1, y1 = int(corners[i][0]), int(corners[i][1])
        x2, y2 = int(corners[i+1][0]), int(corners[i+1][1])
        cv.line(img,(x1,y1),(x2,y2),(0,255 * ((i + 1) % 2),255 * ((i + 2) % 2)),2)

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
    # setCamResolution(cap, 960, 540)
    with np.load('cam0.npz') as X:
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
                print(tag)
                drawBorders(frame, tag.corners)
                rotX, rotY, rotZ = getRotationInfo(tag.pose_R)
                print("Rotation: ", rotX, ",", rotY, ",", rotZ)
                print(tag.pose_t)
                print("------------------------------------------------------------")
        
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):    
            break
    
    cap.release()
    cv.destroyAllWindows()
    
    
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
    
def cam3():
    global detectedTag
    at_detector = at.Detector(families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)
    
    
    cap = cv.VideoCapture(3)
    setCamResolution(cap, 1920, 1080)
    with np.load('cam1.npz') as X:
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
                print(tag)
                drawBorders(frame, tag.corners)
                rotX, rotY, rotZ = getRotationInfo(tag.pose_R)
                print("Rotation: ", rotX, ",", rotY, ",", rotZ)
                print(tag.pose_t)
                print("------------------------------------------------------------")
        
        cv.imshow('frame3',frame)
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
    return rotX, rotY, rotZ
    
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


if __name__ == "__main__":
    getCamIDs()

    t1 = threading.Thread(target=cam1)
    t2 = threading.Thread(target=cam2)
    # starting thread 1 
    t1.start() 
    # starting thread 2 
    t2.start() 
  
    # wait until thread 1 is completely executed 
    t1.join() 
    # wait until thread 2 is completely executed 
    t2.join() 
    # both threads completely executed 
    print("Done!")