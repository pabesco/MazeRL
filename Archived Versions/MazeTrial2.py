import numpy as np
import cv2
import copy
import time

img1 = cv2.imread('imageStartEnd.jpg', cv2.IMREAD_COLOR)
WIDTH, HEIGHT = img1.shape[0], img1.shape[1]
print(WIDTH, HEIGHT)

UNIT_SIZE = 27.5

x = 1
y = 1
RED = (0,0,255)

done = False
while not done:
    coordinate = (x*UNIT_SIZE, y*UNIT_SIZE)
    #corn2 = (x+30, y+30)
    img = copy.deepcopy(img1)
    cv2.circle(img,coordinate, 15, RED, -1)
    #cv2.rectangle(img,corn1,corn2, RED, -1)
    cv2.imshow('image',img)
    cv2.waitKey(20)
    #cv2.destroyAllWindows()

    #Check1 = img <= (10,10,10)
    #print('BBBBBBB', Check1 == False)
    #print('AAAAAAA \n', Check1)

    for i in range(y, y+30):
        for ii in range(x, x+30):
            #print('Location = ',img1[i][ii])
            compare_border = img1[i][ii] > (220, 220, 220)
            compare_start = img1[i][ii] == (0,0,255)
            compare_end = img[i][ii] > (0,255,0)
            if compare_border.all():
                done = True
                break
            if compare_border.all():
                done = True
                print('Start Point')
                break
            if compare_border.all():
                done = True
                print('End Point')
                break

    choice = np.random.randint(0,4)
    if choice == 0:
        x+=10
    elif choice == 1:
        x-=10
    elif choice == 2:
        y+=10
    elif choice == 3:
        y-=10


'''
diff_img = abs(img1 - img)
#diff_img = 255 - img1

cv2.imshow('image',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(diff_img)
cv2.imshow('image',diff_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


class Maze:
    def __init__(self):
        self.x = 17
        self.y = 20

    def __str__(self):
        return f"{self.x}, {self.y}"

    def action(self, choice):
        if choice == 0:
            self.move(x=5, y=0)
        elif choice == 1:
            self.move(x=-5, y=0)
        elif choice == 2:
            self.move(x=0, y=5)
        elif choice == 3:
            self.move(x=0, y=-5)

    def move(self, x, y):
        self.x += x
        self.y += y

    def end(self):
        if self.x < 0 or self.x > WIDTH or self.y < 0 or self.y > HEIGHT:
            return True
