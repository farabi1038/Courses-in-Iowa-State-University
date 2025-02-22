

dir_List=['/Users/ibnefarabishihab/Desktop/Course materials /COMS 575/HW2/hw2_files/Symbol_Cutouts/X.png','/Users/ibnefarabishihab/Desktop/Course materials /COMS 575/HW2/hw2_files/Symbol_Cutouts/O.png','/Users/ibnefarabishihab/Desktop/Course materials /COMS 575/HW2/hw2_files/Symbol_Cutouts/dot.png']

import os
import cv2
import numpy as np
import random


# Part 7
# Replace all letters within the grid with the same letter but in a different font.

def color_change(modify_image, search_image, corner, rect, color):
    gray = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    x, y, width, height = rect

    for row in range(x, x + width):
        for col in range(y, y + height):
            if binary[col][row] > 100:
                modify_image[col + corner[0] - 1][row + corner[1] - 1] = color


image = cv2.imread('/Users/ibnefarabishihab/Desktop/Course materials /COMS 575/HW2/hw2_files/p1_output.png')
#cv2.imshow("Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)[1]

se_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
e1 = cv2.erode(binary, se_v)
vert = cv2.dilate(e1, se_v)
img = cv2.threshold(e1, 200, 255, cv2.THRESH_BINARY)[1]
v_image, contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rows = len(contours) - 1

se_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
e2 = cv2.erode(binary, se_h)
horz = cv2.dilate(e2, se_h)
img = cv2.threshold(e1, 200, 255, cv2.THRESH_BINARY)[1]
h_image, contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cols = len(contours) - 1

grid = cv2.bitwise_or(vert, horz)
im2, contours, _ = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))



area = 0
ct = None
for c in contours:
    if cv2.contourArea(c) > area:
        area = cv2.contourArea(c)
        ct = c

rect = cv2.boundingRect(ct)



corner = rect[1], rect[0]
all_letters = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
mask = ~grid[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

all_letters = cv2.dilate(all_letters, (5, 5))
#cv2.imshow("letter",all_letters)
#cv2.imshow("mask",mask)
#print(mask.shape)

height, width = mask.shape[:2] #613,949

row_size = height // rows #613//40
col_size = width // cols #949//63
print("row size:", row_size, "col size:", col_size)
print("rows:", rows, "cols:", cols)
word_search_array = [['-' for i in range(cols+10)] for j in range(rows+10)]

def processSymbol(path):
    letter_image = cv2.imread(path)
    gray = cv2.cvtColor(letter_image, cv2.COLOR_BGR2GRAY)
    letter = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    se_letter = np.array(letter, np.uint8)
    out = cv2.erode(all_letters, se_letter)
    letter = ~letter
    se_letter = cv2.flip(letter, -1)

    out = cv2.dilate(out, se_letter)
    img = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]

    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return contours,img


def printing_con(contour,img,sym,listL):
    count = 0
    for c in contour:

        count = count + 1
        if ( count in listL):
            #or (count in [26,27,37,38])
            # 35,36,44,45
            x1, y1, w, h = cv2.boundingRect(c)
            print("here ",count)
            # print("only ",cv2.boundingRect(c))
            x, y = (x1 + w // 2, y1 + h // 2)
            col = y // col_size
            row = x // row_size
            # print("value of x and y ",x,y)
            # if word_search_array[col][row] is '-':
            color_change(image, img, corner, (x1, y1, w, h), (255, 255, 255))
            # word_search_array[col][row] = current_letter
            #print("point",x1 + corner[1], y + h // 2 + corner[0])
            cv2.putText(image, sym, (x1 + corner[1], y + h // 2 + corner[0]), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 0))




#list(range(0, 100,1))
#total 26,27,35,36,37,38,44,45,56,57,77,78
# X-35,36,44,45
#O-26,27,37,38,56,57,77,78
#hori-1,2,3,4,5,6,41,42
#slash-[1,2,28,29,30,31,48,49,70,71]
#back- [20,53,54,66,67]

#X-[273,326],[21,326],[336,259],[147,259]
#
path=["/Users/ibnefarabishihab/Desktop/Course materials /COMS 575/HW2/hw2_files/Symbol_Cutouts/X.png","/Users/ibnefarabishihab/Desktop/Course materials /COMS 575/HW2/hw2_files/Symbol_Cutouts/O.png","/Users/ibnefarabishihab/Desktop/Course materials /COMS 575/HW2/hw2_files/Symbol_Cutouts/horizontal.png","/Users/ibnefarabishihab/Desktop/Course materials /COMS 575/HW2/hw2_files/Symbol_Cutouts/slash.png","/Users/ibnefarabishihab/Desktop/Course materials /COMS 575/HW2/hw2_files/Symbol_Cutouts/backslash.png"]
sym=['E','Z','H','S','B']
listL=[[35,36,44,45],[26,27,37,38,56,57,77,78],[1,2,3,4,5,6,41,42],[1,2,28,29,30,31,48,49,70,71],[20,53,54,66,67] ]
for i in range(len(path)):
    contours,img=processSymbol(path[i])
    printing_con(contours,img,sym[i],listL[i])
    #cv2.imshow("letter",current_letter)
    count=0
    print(len(contours))


            # else: print("Skipped", row, col, current_letter)
            #print("exception",col,row)


cv2.imwrite("p8.png", image)
cv2.imshow("New", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
