from PIL import Image, ImageChops
import cv2

def ImageColorInvert(file):
    img = cv2.imread(file)
    print(img.shape)
    img = Image.open(file)
    inv_img = ImageChops.invert(img)
    #inv_image = 255 - img
    inv_img.show()
    inv_img.save(f"{file}_Invert.jpg")
    return f"{file}_Invert.jpg"
