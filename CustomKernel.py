import numpy as np
import sklearn.gaussian_process as gp
from scipy.spatial.distance import cdist


class CustomKernel(gp.kernels.Kernel):
    def __init__(self, v=1.0, v_bounds=(1e-5, 1e5), length_scale=1.0, length_scale_bounds=(1e-5, 1e5),
                 p=1.0, p_bounds=(1e-5, 1e5), s=1.0, s_bounds=(1e-5, 1e5)):
        self.v = v
        self.v_bounds = v_bounds
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.p = p
        self.p_bounds = p_bounds
        self.s = s
        self.s_bounds = s_bounds

    @property
    def hyperparameter_length_scale(self):
        # return Hyperparameter(name, type, bounds, num.dimensions).
        # The Hyperparameter class also has a property .fixed.
        return gp.kernels.Hyperparameter("length_scale", 'numeric', self.length_scale_bounds, 1)

    @property
    def hyperparameter_v(self):
        return gp.kernels.Hyperparameter("v", "numeric", self.v_bounds)

    @property
    def hyperparameter_p(self):
        return gp.kernels.Hyperparameter("p", "numeric", self.p_bounds)

    @property
    def hyperparameter_s(self):
        return gp.kernels.Hyperparameter("s", "numeric", self.s_bounds)

    def is_stationary(self):
        return True

    def diag(self, X):
        # The code, as originally posted, returned self(X,X), which is incorrect.
        return np.copy(np.diagonal(self(X, X)))

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y) if Y is not None else X
        if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != 2 or Y.shape[1] != 2:
            raise ValueError("Features must be scalars")

        v_param = np.squeeze(self.v).astype(float)
        if v_param.ndim != 0:
            raise ValueError("v must be a number")

        p_param = np.squeeze(self.p).astype(float)
        if p_param.ndim != 0:
            raise ValueError("p must be a number")

        s_param = np.squeeze(self.s).astype(float)
        if s_param.ndim != 0:
            raise ValueError("s must be a number")

        length_scale = np.squeeze(self.length_scale).astype(float)
        if length_scale.ndim != 0:
            raise ValueError("Length scale must be a number")

        A = list(zip(*X))
        TX = np.array(A[0])[:, np.newaxis]
        KX = np.array(A[1])[:, np.newaxis]

        B = list(zip(*Y))
        TY = np.array(B[0])[:, np.newaxis]
        KY = np.array(B[1])[:, np.newaxis]

        deltafunc = np.vectorize(lambda x: 1 if x == 0 else 0)
        Dkk = deltafunc(cdist(KX, KY, metric='sqeuclidean'))
        Dtt = deltafunc(cdist(TX, TY, metric='sqeuclidean'))

        K = v_param**2 * np.exp(-0.5 * cdist(TX, TY, metric='sqeuclidean') / length_scale ** 2) \
            + p_param**2 * Dkk + s_param**2 * Dkk * Dtt

        if not eval_gradient:
            return K

        # [ length_scale, p, s, v]
        if self.hyperparameter_length_scale.fixed:
            l_gradient = np.empty((len(X), len(Y), 0))
        else:
            l_gradient = v_param**2 *np.exp(-0.5 * cdist(TX, TY, metric='sqeuclidean') / length_scale ** 2) * (cdist(TX, TY, metric='sqeuclidean') / length_scale ** 3)

        if self.hyperparameter_p.fixed:
            p_gradient = np.empty((len(X), len(Y), 0))
        else:
            p_gradient = 2 * p_param * Dkk
        if self.hyperparameter_s.fixed:
            s_gradient = np.empty((len(X), len(Y), 0))
        else:
            s_gradient = 2 * s_param * Dkk * Dtt
        if self.hyperparameter_v.fixed:
            v_gradient = np.empty((len(X), len(Y), 0))
        else:
            v_gradient = 2 * v_param * np.exp(-0.5 * cdist(TX, TY, metric='sqeuclidean') / length_scale ** 2)
        return K, np.dstack((l_gradient,p_gradient, s_gradient,v_gradient))

    def __repr__(self):
        return f"{self.v:.3g}**2 * RBF(length_scale={self.length_scale:.3g}) + {self.p:.3g}**2 * dkk + {self.s:.3g}**2 * dtt * dkk"