from signal import pause
import cv2
import numpy as np
import time

start_time = time.time()


cap = cv2.VideoCapture("p4_video3.avi")
size = (1040, 544)
result = cv2.VideoWriter('p4a_video3_Output.m4v',
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         10, size)

while True:
    ret, frame = cap.read()
    if frame is not None:
        frame = cv2.resize(frame, (1040, 544))

        # Convert to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur using 3 * 3 kernel.
        gray_blurred = cv2.blur(gray, (3, 3))

        # Apply Hough transform on the blurred image.
        detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 25, param1=150,
                                        param2=30, minRadius=15, maxRadius=22)

        # Draw circles that are detected.
        if detected_circles is not None:

            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))

            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
                if (gray_blurred[int(b - r / 2), int(a - r / 2)] < 150):
                    # Draw the circumference of the circle.
                    cv2.circle(frame, (a, b), r, (255, 0, 0), -1)
                else:
                    cv2.circle(frame, (a, b), r, (0, 255, 0), -1)


            # Draw a small circle (of radius 1) to show the center.
            #cv2.imshow("Detected Circle", frame)
            result.write(frame)
    else:
        break

        # cv2.imwrite(str(time.time()-start_time)+'.jpeg', frame)


result.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")

cap.release()
cv2.destroyAllWindows()