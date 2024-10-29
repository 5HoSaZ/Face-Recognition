import cv2
import numpy as np
import scipy.cluster
import sklearn.cluster
from PIL import Image


def dominant_colors_fill(image):  # PIL image input
    ar = np.asarray(image)
    shape = ar.shape
    ar = ar.reshape(np.prod(shape[:2]), shape[2]).astype(float)

    kmeans = sklearn.cluster.MiniBatchKMeans(
        n_clusters=10, init="k-means++", max_iter=20, random_state=1000
    ).fit(ar)
    codes = kmeans.cluster_centers_

    vecs, _dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
    counts, _bins = np.histogram(vecs, len(codes))  # count occurrences

    colors = []
    for index in np.argsort(counts)[::-1]:
        color = tuple([int(code) for code in codes[index]])
        if color != (0, 0, 0):
            colors.append(color)

    image[np.where((image == [0, 0, 0]).all(axis=2))] = colors[0]
    print(colors[index])
    return image


def pad_img(img, padding):
    img = cv2.copyMakeBorder(
        img, padding, padding, padding, padding, cv2.BORDER_CONSTANT
    )
    return img


def show_img(img, source="cv2"):
    if source == "cv2":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img.show()


def crop_center(img, width, height):
    shape = img.shape
    x = shape[1] / 2 - width / 2
    y = shape[0] / 2 - height / 2
    return img[int(y) : int(y + height), int(x) : int(x + width)]
