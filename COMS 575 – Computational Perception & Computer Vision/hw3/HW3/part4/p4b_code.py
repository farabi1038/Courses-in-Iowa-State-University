from signal import pause
import cv2
import numpy as np
import time

start_time = time.time()


cap = cv2.VideoCapture("p4_video3.avi")

size = (1040, 544)
result = cv2.VideoWriter('p4b_video3_Output.m4v',
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         10, size)


def variables():
    CountBallPerTriangle = np.zeros(24, dtype=int)
    triangleColor = np.empty(24, dtype=str)

    TriangleBottomLeft = []
    TriangleTopLeft = []
    return CountBallPerTriangle,triangleColor,TriangleBottomLeft,TriangleTopLeft

def defineT():
    trinagleStartX = 27
    trinagleEndX = 99
    difBetweenTriangle = 5
    triangleSizeX = trinagleEndX - trinagleStartX
    TraingleY = 30
    return trinagleStartX,trinagleEndX,difBetweenTriangle,triangleSizeX,TraingleY


def populateTriangleUpper(image):
    trinagleStartX, trinagleEndX, difBetweenTriangle, triangleSizeX, TraingleY = defineT()
    for n in range(0, 12):
        if (n >= 6):
            trinagleStartX = 83
            difBetweenTriangle = 4
        xCor = trinagleStartX + (triangleSizeX) * n + difBetweenTriangle

        x = xCor + triangleSizeX

        cv2.putText(image, str("."), (x, TraingleY), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 4)
        cv2.putText(image, str("."), (xCor,210), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 4)
        TriangleBottomLeft.append((x, TraingleY))

        TriangleTopLeft.append((xCor,210))

def populateTriangleLower(image):
    trinagleStartX, trinagleEndX, difBetweenTriangle, triangleSizeX, TraingleY = defineT()
    TraingleY = 520
    trinagleStartX = 26
    for n in range(0, 12):
        if (n >= 6):
            trinagleStartX = 83
            difBetweenTriangle = 4
        xCor = trinagleStartX + (triangleSizeX) * n + difBetweenTriangle

        x = xCor + triangleSizeX

        cv2.putText(image, str("."), (x, TraingleY), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 4)
        TriangleTopLeft.append((x, TraingleY))

        TriangleBottomLeft.append((xCor,341))

def checkTriangle(bl, tr, p):
    if (int(p[0]) > int(bl[0]) and int(p[0]) < int(tr[0]) and int(p[1]) > int(bl[1]) and int(p[1] < tr[1])):
        print('in')
        return True
    else:
        return False


def checkTriangle2(bl, tr, p):
    if (int(p[0]) > int(tr[0]) and int(p[0]) < int(bl[0]) and int(p[1]) > int(bl[1]) and int(p[1] < tr[1])):
        print('in')
        return True
    else:
        return False



def checkBallTriangle(pts,image):

    #print('size',len(TriangleBottomLeft))
    #cv2.putText(image, str(0), (pts[0], pts[1]), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 4)
    for i in range(len(TriangleBottomLeft)):
        #print('checking for triangle')
        bool1=checkTriangle(TriangleBottomLeft[i],TriangleTopLeft[i],pts)
        bool2 = checkTriangle2(TriangleBottomLeft[i], TriangleTopLeft[i], pts)
        #print(TriangleBottomLeft[i],TriangleTopLeft[i],pts)

        if bool1 or bool2:
            print('triangle',i)
            #cv2.putText(image, str(i), (pts[0], pts[1]), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 4)
            #CountBallPerTriangle[i]=CountBallPerTriangle[i]+1
            #print('asasa',CountBallPerTriangle[i],i,"done")
            return i




def putValues(index,color,image):
    wCount = 0
    rCount = 0
    base = 54
    y_pos = 230
    count=0
    for num in range(1, 13):
        if num > 6:
            base = 111
        x_position = base + (num - 1) * 72
        text = str(index[count])+color[count]
        count=count+1
        cv2.putText(image, str(text), (x_position, y_pos), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 4)
    base = 65
    y_pos = 285
    for num in range(1, 13):
        if num > 6:
            base = 120
        x_position = base + (num - 1) * 72
        text = str(index[count]) + color[count]
        count=count+1
        cv2.putText(image, str(text), (x_position, y_pos), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 4)







while True:
    count=0

    ret, frame = cap.read()
    if frame is not None:
        frame = cv2.resize(frame, (1040, 544))


        # Convert to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur using 3 * 3 kernel.
        gray_blurred = cv2.blur(gray, (3, 3))
        CountBallPerTriangle,triangleColor,TriangleBottomLeft,TriangleTopLeft=variables()
        populateTriangleUpper(frame)
        populateTriangleLower(frame)
        # Apply Hough transform on the blurred image.
        detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 25, param1=150,
                                        param2=30, minRadius=15, maxRadius=22)

        # Draw circles that are detected.
        if detected_circles is not None:
            print("circle count",len(detected_circles[0, :]))
            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))
            color='w' #White else will be for black
            for pt in detected_circles[0, :]:

                bool=True
                a, b, r = pt[0], pt[1], pt[2]
                x,y=int(b - r / 2), int(a - r / 2)

                if (gray_blurred[x,y] > 150):
                    # Draw the circumference of the circle.
                    cv2.circle(frame, (a, b), r, (255, 0, 0), -1)

                else:
                    cv2.circle(frame, (a, b), r, (0, 255, 0), -1)
                    bool=False
                print("chekcing ball")
                ball=checkBallTriangle((a, b),frame)
                print('ball id ',ball)
                CountBallPerTriangle[ball]=CountBallPerTriangle[ball]+1
                if bool:
                    triangleColor[ball]='B'
                else:
                    triangleColor[ball] = 'G'
                    print('in red')
                print('count', triangleColor)
            putValues(CountBallPerTriangle,triangleColor,frame)


            # Draw a small circle (of radius 1) to show the center.
        cv2.imshow("Detected Circle", frame)
        result.write(frame)
        print('one frame done')
    else:
        break
        # cv2.imwrite(str(time.time()-start_time)+'.jpeg', frame)
result.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")

cap.release()
cv2.destroyAllWindows()