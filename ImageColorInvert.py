from PIL import Image, ImageChops
import cv2
img = cv2.imread("NewMaze2.jpg")
print(img.shape)
img = Image.open('NewMaze2.jpg')
inv_img = ImageChops.invert(img)
#inv_image = 255 - img
inv_img.show()
inv_img.save("NewMaze2BW.jpg")
