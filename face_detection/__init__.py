from .face_detector import FaceDetector
from .face_parsing import FaceSegmentation
from .facenet_pytorch import MTCNN

# Other dependencies
from PIL import Image
import cv2
import numpy as np


class FaceDetectionPipeline:
    __EXCLUDE_LABELS = [0, 14, 17, 18]

    def __init__(self, device):
        self.mtcnn = MTCNN(keep_all=True, device=device)
        self.segm = FaceSegmentation(device)

    def __masked_image(self, image: Image):
        img_array = np.asarray(image)
        # Get segmentation labels
        labels = self.segm.get_mask(image)
        # Create mask
        mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
        mask.fill(255)
        for v in self.__EXCLUDE_LABELS:
            mask[labels == v] = 0
        masked_img = cv2.bitwise_and(img_array, img_array, mask=mask)
        return Image.fromarray(masked_img)

    def __get_bounding_box(self, image: Image):
        def fit_box_to_image(box, image):
            x1, y1, x2, y2 = box
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, image.size[0]), min(y2, image.size[1])
            return x1, y1, x2, y2

        def get_box_size(box):
            return int((box[2] - box[0]) * (box[3] - box[1]))

        boxes, accs = self.mtcnn.detect(image)
        if boxes is None or len(boxes) == 0:
            x1, y1, x2, y2 = (0, 0) + image.size
        else:
            boxes = [fit_box_to_image(box, image) for box in boxes]
            detected = list(
                sorted(
                    zip(boxes, accs),
                    key=lambda x: (get_box_size(x[0]), float(x[1])),
                    reverse=True,
                )
            )
            x1, y1, x2, y2 = (int(v) for v in detected[0][0])
        return ((x1, y1), (x2, y2))

    def __crop_to_box(image, box, resize=(224, 224)):
        img_array = np.asarray(image)
        (x1, y1), (x2, y2) = box
        cropped = img_array[y1:y2, x1:x2]
        cropped_image = Image.fromarray(cropped)
        cropped_image = cropped_image.resize(resize)
        return cropped_image

    def __call__(self, image, resize=(224, 224)) -> Image:
        masked_image = self.__masked_image(image)
        box = self.__get_bounding_box(image)
        cropped = FaceDetectionPipeline.__crop_to_box(masked_image, box, resize=resize)
        return cropped
