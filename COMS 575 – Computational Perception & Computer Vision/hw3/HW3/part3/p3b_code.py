import cv2
from pytesseract import pytesseract
from pytesseract import Output
import numpy as np




def roll(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,(99,35,135),(105,73,230))
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 15)
    res = cv2.bitwise_and(img, img, mask=mask)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(res, 180, 255, cv2.THRESH_BINARY)
    res = ~blackAndWhiteImage



    image_data = pytesseract.image_to_data(res, output_type=Output.DICT)

    point = 'Roll Dice !'
    point2 = 'Roll Dice'
    for i, word in enumerate(image_data['text']):
        if word == 'Roll':       
            x,y,w,h = image_data['left'][i],image_data['top'][i],image_data['width'][i],image_data['height'][i]
            cv2.rectangle(img, pt1=(x-10,y-10), pt2=(x+130,y+30), color=(230,50,50), thickness=-1)
            cv2.putText(img,point,(x,y-70),cv2.FONT_HERSHEY_COMPLEX,3,(255,255,5),5)
            cv2.putText(img,point2,(x-5,y+13),cv2.FONT_HERSHEY_COMPLEX,.8,(5,5,255),5)


    return img



cap = cv2.VideoCapture("p3_video2.avi")
size = (1040, 544)
result = cv2.VideoWriter('p3b_video2_Output.m4v',
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         10, size)
while True:
    ret, frame = cap.read()
    if frame is not None:
        frame = cv2.resize(frame,(1040,544))

        frame = roll(frame)
        result.write(frame)
        cv2.imshow('frame',frame)
        cv2.waitKey(1)

result.release()
cap.release()
cv2.destroyAllWindows()

