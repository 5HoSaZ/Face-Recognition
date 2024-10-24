import cv2
from PIL import Image


def show_img(img, source="cv2"):
    if source == "cv2":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img.show()


def crop_center(img, width, height):
    shape = img.shape
    x = shape[0] / 2 - width / 2
    y = shape[1] / 2 - height / 2
    return img[int(y) : int(y + height), int(x) : int(x + width)]
