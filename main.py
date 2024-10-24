import cv2


from face_detection import FaceDetector
from frontalizer import M3D_Model, Cam3DEstimate, Frontalizer


if __name__ == "__main__":
    detector = FaceDetector()
    img = cv2.imread("./images/Medvedev-Profile.png")
    # img = cv2.imread("./images/Channing-Tatum.jpg")
    img = detector.crop_face(img, 20)
    landmarks = detector.get_landmarks(img)
    face_only = detector.get_face_only(img, landmarks[0])

    model3D = M3D_Model()
    estimate_camera = Cam3DEstimate(model3D)
    frontalizer = Frontalizer()

    proj_matrix, camera_matrix, rmat, tvec = estimate_camera(landmarks[0])
    frontal_raw, frontal_sym = frontalizer(face_only, proj_matrix, model3D.ref_U)
    cv2.imshow("mev", face_only)
    cv2.waitKey(0)
    cv2.imshow("mev", frontal_raw)
    cv2.waitKey(0)
    cv2.imshow("mev", frontal_sym)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
