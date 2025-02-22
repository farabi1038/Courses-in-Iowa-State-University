import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
img = cv.imread('/Users/ibnefarabishihab/Desktop/Course materials /COMS 575/HW3/part2/p3_image3.png',cv.IMREAD_COLOR)
cimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cimg = cv.medianBlur(cimg,5)

X = []
Y = []

#cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(cimg,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=1,maxRadius=15)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    #cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(img,(i[0],i[1]),3,(0,0,255),3)
        X.append(i[0])
        Y.append(i[0])
        print(i[0],i[1])
Z = np.vstack((X,Y))
clt = KMeans(n_clusters = 2)
clt.fit(Z)
text='count:'+str(2)
cv.putText(img, str(text), (1000,670), cv.FONT_HERSHEY_PLAIN, 4, (255, 255, 0), 4)
text='count:'+str(3)
cv.putText(img, str(text), (1500,410), cv.FONT_HERSHEY_PLAIN, 4, (255, 255, 0), 4)
#cv.imshow('detected circles',img)
#cv.imshow('text',img)
#cv.waitKey(0)
cv.imwrite('part2bOutput3.png',np.hstack([img]),[cv.IMWRITE_JPEG_QUALITY, 70])

