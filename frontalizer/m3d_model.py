import scipy.io
import numpy as np


class M3D_Model:
    model_path = "./common/model3Ddlib.mat"
    model_name = "model_dlib"

    def __init__(self):
        model = scipy.io.loadmat(self.model_path)[self.model_name]
        self.out_A = np.asmatrix(model["outA"][0, 0], dtype=np.float32)
        self.size_U = model["sizeU"][0, 0][0]
        self.model_TD = np.asarray(model["threedee"][0, 0], dtype=np.float32)
        self.indbad = model["indbad"][0, 0]
        self.ref_U = np.asarray(model["refU"][0, 0])
