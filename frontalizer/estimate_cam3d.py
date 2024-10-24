from .m3d_model import M3D_Model

import cv2
import numpy as np


class Cam3DEstimate:
    """
    Estimate a 3D projection matrix for frontalization.
    """

    def __init__(self, model: M3D_Model):
        self.model3D = model

    def __call__(self, landmarks: np.ndarray):
        rmat, tvec = self.__calib_camera(landmarks)
        RT = np.hstack((rmat, tvec))
        projection_matrix = self.model3D.out_A * RT
        return projection_matrix, self.model3D.out_A, rmat, tvec

    def __calib_camera(self, fidu_XY):
        # compute pose using refrence 3D points + query 2D points
        ret, rvecs, tvec = cv2.solvePnP(
            self.model3D.model_TD, fidu_XY, self.model3D.out_A, None, None, None, False
        )
        rmat, jacobian = cv2.Rodrigues(rvecs, None)

        inside = self.__calc_inside(
            self.model3D.out_A,
            rmat,
            tvec,
            self.model3D.size_U[0],
            self.model3D.size_U[1],
            self.model3D.model_TD,
        )
        if inside == 0:
            tvec = -tvec
            t = np.pi
            RRz180 = np.asmatrix(
                [np.cos(t), -np.sin(t), 0, np.sin(t), np.cos(t), 0, 0, 0, 1]
            ).reshape((3, 3))
            rmat = RRz180 * rmat
        return rmat, tvec

    def __get_opengl_matrices(self, camera_matrix, rmat, tvec, width, height):
        projection_matrix = np.asmatrix(np.zeros((4, 4)))
        near_plane = 0.0001
        far_plane = 10000

        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        px = camera_matrix[0, 2]
        py = camera_matrix[1, 2]

        projection_matrix[0, 0] = 2.0 * fx / width
        projection_matrix[1, 1] = 2.0 * fy / height
        projection_matrix[0, 2] = 2.0 * (px / width) - 1.0
        projection_matrix[1, 2] = 2.0 * (py / height) - 1.0
        projection_matrix[2, 2] = -(far_plane + near_plane) / (far_plane - near_plane)
        projection_matrix[3, 2] = -1
        projection_matrix[2, 3] = (
            -2.0 * far_plane * near_plane / (far_plane - near_plane)
        )

        deg = 180
        t = deg * np.pi / 180.0
        RRz = np.asmatrix(
            [np.cos(t), -np.sin(t), 0, np.sin(t), np.cos(t), 0, 0, 0, 1]
        ).reshape((3, 3))
        RRy = np.asmatrix(
            [np.cos(t), 0, np.sin(t), 0, 1, 0, -np.sin(t), 0, np.cos(t)]
        ).reshape((3, 3))
        rmat = RRz * RRy * rmat

        mv = np.asmatrix(np.zeros((4, 4)))
        mv[0:3, 0:3] = rmat
        mv[0, 3] = tvec[0].item()
        mv[1, 3] = -tvec[1].item()
        mv[2, 3] = -tvec[2].item()
        mv[3, 3] = 1.0
        return mv, projection_matrix

    def __extract_frustum(self, camera_matrix, rmat, tvec, width, height):
        mv, proj = self.__get_opengl_matrices(camera_matrix, rmat, tvec, width, height)
        clip = proj * mv
        frustum = np.asmatrix(np.zeros((6, 4)))
        # /* Extract the numbers for the RIGHT plane */
        frustum[0, :] = clip[3, :] - clip[0, :]
        # /* Normalize the result */
        v = frustum[0, :3]
        t = np.sqrt(np.sum(np.multiply(v, v)))
        frustum[0, :] = frustum[0, :] / t

        # /* Extract the numbers for the LEFT plane */
        frustum[1, :] = clip[3, :] + clip[0, :]
        # /* Normalize the result */
        v = frustum[1, :3]
        t = np.sqrt(np.sum(np.multiply(v, v)))
        frustum[1, :] = frustum[1, :] / t

        # /* Extract the BOTTOM plane */
        frustum[2, :] = clip[3, :] + clip[1, :]
        # /* Normalize the result */
        v = frustum[2, :3]
        t = np.sqrt(np.sum(np.multiply(v, v)))
        frustum[2, :] = frustum[2, :] / t

        # /* Extract the TOP plane */
        frustum[3, :] = clip[3, :] - clip[1, :]
        # /* Normalize the result */
        v = frustum[3, :3]
        t = np.sqrt(np.sum(np.multiply(v, v)))
        frustum[3, :] = frustum[3, :] / t

        # /* Extract the FAR plane */
        frustum[4, :] = clip[3, :] - clip[2, :]
        # /* Normalize the result */
        v = frustum[4, :3]
        t = np.sqrt(np.sum(np.multiply(v, v)))
        frustum[4, :] = frustum[4, :] / t

        # /* Extract the NEAR plane */
        frustum[5, :] = clip[3, :] + clip[2, :]
        # /* Normalize the result */
        v = frustum[5, :3]
        t = np.sqrt(np.sum(np.multiply(v, v)))
        frustum[5, :] = frustum[5, :] / t
        return frustum

    def __calc_inside(self, camera_matrix, rmat, tvec, width, height, obj_points):
        frustum = self.__extract_frustum(camera_matrix, rmat, tvec, width, height)
        inside = 0
        for point in obj_points:
            if self.__point_in_frustum(point[0], point[1], point[2], frustum) > 0:
                inside += 1
        return inside

    def __point_in_frustum(self, x, y, z, frustum):
        for p in range(0, 3):
            if (
                frustum[p, 0] * x
                + frustum[p, 1] * y
                + frustum[p, 2]
                + z
                + frustum[p, 3]
                <= 0
            ):
                return False
        return True
