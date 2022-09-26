import numpy as np
import os
import cv2
from PIL import Image
import pygame

img = Image.open("MazeStartEnd.jpg")
#print(img)
width, height = img.size
#print(width, height)
image_array = np.asarray(img)
image_array_BGR = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

#print(type(image_array))

print(image_array.shape)
#print(image_array)
cv2.imshow("image", np.array(image_array_BGR))
#image_array[40][40] = (255, 175, 0)
cv2.waitKey(0)

pygame.init()
game_display = pygame.display.set_mode((width, height))

window = True
while window:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            window = False


    Img = pygame.image.load('MazeStartEnd.jpg')
    game_display.blit(Img, (0,0))
    #pygame.draw.rect(game_display, (255,0,0), [20, 15, 40, 40], 0)
    #pygame.draw.circle(game_display, (255,0,0),[40,40], 10)
    #pygame.draw.rect(game_display, (255,0,0), [70, 15, 40, 40], 0)
    #pygame.draw.circle(game_display, (0,255,0), [525, 256], 10)
    pygame.display.set_caption('Image')
    pygame.display.update()
    img_data = pygame.surfarray.array3d(game_display)
    img_data = img_data.swapaxes(0,1)

pygame.quit()
print("============================================================================================")
print(img_data.shape)
img_data_BGR = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

cv2.imshow("Processed", np.array(img_data_BGR))
cv2.waitKey(0)
Diff_img = img_data_BGR - image_array_BGR
cv2.imshow("Difference", np.array(Diff_img))
cv2.waitKey(0)
