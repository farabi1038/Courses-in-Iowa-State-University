import numpy as np
import cv2 as cv
img = cv.imread('/Users/ibnefarabishihab/Desktop/Course materials /COMS 575/HW3/part1/p1_image3.png',cv.IMREAD_COLOR)
cimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cimg = cv.medianBlur(cimg,5)
#cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(cimg,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=16,maxRadius=41)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
        cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(img,(i[0],i[1]),3,(0,0,255),3)
#cv.imshow('detected circles',cimg)

#cv.waitKey(0)
cv.imwrite('part1bOutput3.png',np.hstack([img]),[cv.IMWRITE_JPEG_QUALITY, 70])
