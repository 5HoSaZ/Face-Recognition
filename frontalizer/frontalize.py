import cv2
import scipy.io
import numpy as np


class Frontalizer:
    eyemask_path = "./common/eyemask.mat"

    def __init__(self):
        self.eyemask = np.asarray(scipy.io.loadmat(self.eyemask_path)["eyemask"])

    def __call__(self, img, proj_matrix, ref_U):
        ACC_CONST = 800
        img = img.astype("float32")
        # print("query image shape:", img.shape)

        bgind = np.sum(np.abs(ref_U), 2) == 0
        # Count the number of times each pixel in the query is accessed
        threedee = np.reshape(ref_U, (-1, 3), order="F").transpose()
        temp_proj = proj_matrix * np.vstack((threedee, np.ones((1, threedee.shape[1]))))
        temp_proj2 = np.divide(temp_proj[0:2, :], np.tile(temp_proj[2, :], (2, 1)))

        bad = np.logical_or(temp_proj2.min(axis=0) < 1, temp_proj2[1, :] > img.shape[0])
        bad = np.logical_or(bad, temp_proj2[0, :] > img.shape[1])
        bad = np.logical_or(bad, bgind.reshape((-1), order="F"))
        bad = np.asarray(bad).reshape((-1), order="F")

        temp_proj2 -= 1

        badind = np.nonzero(bad > 0)[0]
        temp_proj2[:, badind] = 0

        ind = np.ravel_multi_index(
            (
                np.asarray(temp_proj2[1, :].round(), dtype="int64"),
                np.asarray(temp_proj2[0, :].round(), dtype="int64"),
            ),
            dims=img.shape[:-1],
            order="F",
        )

        synth_frontal_acc = np.zeros(ref_U.shape[:-1])
        ind_frontal = np.arange(0, ref_U.shape[0] * ref_U.shape[1])

        c, ic = np.unique(ind, return_inverse=True)
        bin_edges = np.r_[-np.Inf, 0.5 * (c[:-1] + c[1:]), np.Inf]
        count, bin_edges = np.histogram(ind, bin_edges)
        synth_frontal_acc = synth_frontal_acc.reshape(-1, order="F")
        synth_frontal_acc[ind_frontal] = count[ic]
        synth_frontal_acc = synth_frontal_acc.reshape((320, 320), order="F")
        synth_frontal_acc[bgind] = 0
        synth_frontal_acc = cv2.GaussianBlur(
            synth_frontal_acc, (15, 15), 30.0, borderType=cv2.BORDER_REPLICATE
        )

        # remap
        mapX = temp_proj2[0, :].astype(np.float32)
        mapY = temp_proj2[1, :].astype(np.float32)

        mapX = np.reshape(mapX, (-1, 320), order="F")
        mapY = np.reshape(mapY, (-1, 320), order="F")

        frontal_raw = cv2.remap(img, mapX, mapY, cv2.INTER_CUBIC)

        frontal_raw = frontal_raw.reshape((-1, 3), order="F")
        frontal_raw[badind, :] = 0
        frontal_raw = frontal_raw.reshape((320, 320, 3), order="F")

        # which side has more occlusions?
        midcolumn = round(ref_U.shape[1] / 2)
        # print(midcolumn, type(midcolumn))
        sumaccs = synth_frontal_acc.sum(axis=0)
        sum_left = sumaccs[0:midcolumn].sum()
        sum_right = sumaccs[midcolumn + 1 :].sum()
        sum_diff = sum_left - sum_right

        if np.abs(sum_diff) > ACC_CONST:  # one side is ocluded
            ones = np.ones((ref_U.shape[0], midcolumn))
            zeros = np.zeros((ref_U.shape[0], midcolumn))
            if sum_diff > ACC_CONST:  # left side of face has more occlusions
                weights = np.hstack((zeros, ones))
            else:  # right side of face has more occlusions
                weights = np.hstack((ones, zeros))
            weights = cv2.GaussianBlur(
                weights, (33, 33), 60.5, borderType=cv2.BORDER_REPLICATE
            )

            # apply soft symmetry to use whatever parts are visible in ocluded side
            synth_frontal_acc /= synth_frontal_acc.max()
            weight_take_from_org = 1.0 / np.exp(0.5 + synth_frontal_acc)
            weight_take_from_sym = 1 - weight_take_from_org

            weight_take_from_org = np.multiply(weight_take_from_org, np.fliplr(weights))
            weight_take_from_sym = np.multiply(weight_take_from_sym, np.fliplr(weights))

            weight_take_from_org = np.tile(
                weight_take_from_org.reshape(320, 320, 1), (1, 1, 3)
            )
            weight_take_from_sym = np.tile(
                weight_take_from_sym.reshape(320, 320, 1), (1, 1, 3)
            )
            weights = np.tile(weights.reshape(320, 320, 1), (1, 1, 3))

            denominator = weights + weight_take_from_org + weight_take_from_sym
            frontal_sym = (
                np.multiply(frontal_raw, weights)
                + np.multiply(frontal_raw, weight_take_from_org)
                + np.multiply(np.fliplr(frontal_raw), weight_take_from_sym)
            )
            frontal_sym = np.divide(frontal_sym, denominator)
            # Exclude eyes from symmetry
            frontal_sym = np.multiply(frontal_sym, 1 - self.eyemask) + np.multiply(
                frontal_raw, self.eyemask
            )
            frontal_raw[frontal_raw > 255] = 255
            frontal_raw[frontal_raw < 0] = 0
            frontal_raw = frontal_raw.astype(np.uint8)
            frontal_sym[frontal_sym > 255] = 255
            frontal_sym[frontal_sym < 0] = 0
            frontal_sym = frontal_sym.astype(np.uint8)
        # If both sides are occluded pretty much to the same extent -- dont use symmetry
        else:
            frontal_sym = frontal_raw
        return frontal_raw, frontal_sym
