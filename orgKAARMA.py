from kernels import heaviside, gaussian, Phi
import numpy as np


class KAARMA:
    def __init__(self, ns, ny, a_s, a_u, t, lr):
        self.ns = ns
        self.ny = ny
        self.a_s = a_s
        self.a_u = a_u
        self.t = t
        self.lr = lr
        self.A = np.random.random((1, ns))
        # self.S = np.random.random((1, ns))

        self.II = np.zeros((ny, ns))
        self.II[:, ns - ny:] = np.eye(ny)

        self.PHI = [Phi([])]
        self.m = 1

    def compute_kernel(self, u1, u2, s1, s2):
        return gaussian(s1, s2, self.a_s) * heaviside(u1, u2, self.t, self.a_u)

    def compute_kernel_batches(self, u, s1, s2):
        mm = np.zeros(self.m)
        for i in range(self.m):
            mm[i] = self.compute_kernel(self.PHI[i], u, s1, s2)
        return mm.T

    def train(self, u, y, epochs):
        for i in range(epochs):
            for (du, dy) in zip(u, y):
                s = np.random.random((2, self.ns))
                for frame in du:
                    s = np.vstack([s, self.A.T @ self.compute_kernel_batches(Phi(frame), s[-1], s[-2])])

    def test(self, u, y):
        for (du, dy) in zip(u, y):
            s = np.random.random((2, self.ns))
            for frame in du:
                s = np.vstack([s, self.A.T @ self.compute_kernel_batches(Phi(frame), s[-1], s[-2])])
            pred = self.II @ s[-1]
            print(pred)






