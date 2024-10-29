import dlib
import cv2
import numpy as np


class FaceDetector:
    """
    Perform face detection and landmarks extraction using dlib face detector.
    """

    def __init__(self) -> None:
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "./common/shape_predictor_68_face_landmarks.dat"
        )

    def crop_face(self, img, padding: int = 0):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = self.detector(gray_img, 1)[0]
        img = img[face.top() : face.bottom(), face.left() : face.right()]
        if padding > 0:
            img = cv2.copyMakeBorder(
                img, padding, padding, padding, padding, cv2.BORDER_CONSTANT
            )
        return img

    def get_landmarks(self, img) -> list[np.ndarray]:
        """
        Return a list of landmarks cordinates (in float32) for each detected faces.
        """
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray_img, 1)
        # Get landmarks coordinates for each face
        landmarks = []
        for face in faces:
            lmarks = self.predictor(gray_img, face)
            lmark_coords = []
            for n in range(68):
                x = lmarks.part(n).x
                y = lmarks.part(n).y
                lmark_coords.append([x, y])
            # Convert to numpy array
            lmark_coords = np.array(lmark_coords, dtype=np.float32)
            landmarks.append(lmark_coords)
        return landmarks

    def get_face_only(self, img, landmarks: np.ndarray):
        """
        Transform input image to only show face defined by landmark.
        """
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Create a convex hull from landmarks
        convex_hull = cv2.convexHull(landmarks.astype(np.int32))
        # Generate a mask from convex hull
        mask = np.zeros_like(gray_img)
        cv2.fillConvexPoly(mask, convex_hull, 255)
        # Return image with only face visible
        img = cv2.bitwise_and(img, img, mask=mask)
        return img
