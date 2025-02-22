from collections import deque
import numpy as np
import cv2 as cv2
import time
import imutils



# pts = deque(maxlen = 30)

pts = []

cap = cv2.VideoCapture("p5_video3.m4v")
size = (1040, 544)
result = cv2.VideoWriter('p5b_video3_Output.m4v',
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         10, size)

while True:

    suc, img = cap.read()
    if img is not None:
        frame = cv2.resize(img,(1040,524))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 130, 130])
        upper_red = np.array([5, 255, 255])
        mask0 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([176, 130, 130])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)


        lower_white = np.array([0, 0, 20])
        upper_white = np.array([232, 54, 255])
        mask2 = cv2.inRange(hsv, lower_white, upper_white)

        mask = mask0 + mask1 +mask2
        new_img = cv2.bitwise_and(frame,frame,mask = mask)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None


        for cnt in cnts:
            # c = max(cnt, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            M = cv2.moments(cnt)

            if  M["m00"] > 0 and radius> 11:

                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                pts.append(center)




        # loop over the set of tracked points
        for i in pts[-2300:]:

            cv2.circle(frame,i,5, (255, 0, 0), -1)



        cv2.imshow('flow', frame)
        cv2.imshow('new_image', new_img)
        cv2.imshow('mask', mask)
        result.write(frame)
    else:
        break

result.release()

cap.release()
cv2.destroyAllWindows()
