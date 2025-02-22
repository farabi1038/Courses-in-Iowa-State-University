import numpy as np
import cv2
import time



def draw_flow(img, flow, step=3):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


def draw_hsv(flow):

    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 0
    hsv[...,2] = v*100#np.minimum(v*100, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr




cap = cv2.VideoCapture("p5_video3.m4v")

suc, prev = cap.read()
prev = cv2.resize(prev,(524,224))
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
flow = None

size = (1040, 544)
result = cv2.VideoWriter('p5a_video3_Output.m4v',
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         10, size)
#Taken from github
while True:

    suc, img = cap.read()
    if img is not None:
        img = cv2.resize(img,(524,224))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

        # start time to calculate FPS
        start = time.time()


        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0)
    
        prevgray = gray


        # End time
        end = time.time()
        # calculate the FPS for current frame detection
        fps = 1 / (end-start)

        flow=draw_flow(gray, flow)
        cv2.imshow('flow',flow )
        result.write(flow)
    else:
        break


result.release()
cap.release()
cv2.destroyAllWindows()

