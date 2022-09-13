from PIL import Image, ImageChops
#import cv2

def ImageColorInvert(file = 'NewMaze2.jpg'):
    #img = cv2.imread(file)
    #print(img.shape)
    img = Image.open(file)
    inv_img = ImageChops.invert(img)
    inv_img = inv_img.resize((round(inv_img.size[0]/100)*100, round(inv_img.size[1]/100)*100))
    print(inv_img.size)
    #inv_image = 255 - img
    #inv_img.show()
    inv_img.save(f"{file}_Invert.jpg")
    return f"{file}_Invert.jpg"
