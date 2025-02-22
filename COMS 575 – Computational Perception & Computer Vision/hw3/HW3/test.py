import matplotlib as plt
import numpy as np
import cv2
img = cv2.imread('/Users/ibnefarabishihab/Desktop/Course materials /COMS 575/HW3/part1/p1_image3.png',cv2.IMREAD_COLOR)
#img = cv2.resize(img,(256,256))




rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

#light_blue = (90, 70, 50)
#dark_blue = (128, 255, 255)

# lower mask (0-10)
lower_red = np.array([0,130,130])
upper_red = np.array([5,255,255])
mask0 = cv2.inRange(hsv_img, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([176,130,130])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(hsv_img, lower_red, upper_red)
mask=mask0+mask1

# join my masks


# You can use the following values for green
# light_green = (40, 40, 40)
# dark_greek = (70, 255, 255)
#mask = cv2.inRange(hsv_img, mask0, mask1)
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("draw", result)
cv2.waitKey(0)