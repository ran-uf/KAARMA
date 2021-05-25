from kernels import heaviside_multi_channel, gaussian, Phi
import numpy as np
import logging
from multiprocessing.dummy import Pool as ThreadPool


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)


class KAARMA:
    def __init__(self, ns, ny, a_s, a_u, t):
        self.ns = ns
        self.ny = ny
        self.a_s = a_s
        self.a_u = a_u
        self.t = t
        self.A = np.random.random((1, ns))
        self.S = np.random.random((1, ns))

        self.II = np.zeros((ny, ns))
        self.II[:, ns - ny:] = np.eye(ny)

        self.PHI = [Phi([[]])]
        self.m = 1

    def compute_kernel(self, u1, u2, s1, s2):
        return gaussian(s1, s2, self.a_s) * heaviside_multi_channel(u1, u2, self.t, self.a_u)

    def compute_kernel_multi(self, a):
        (u1, u2, s1, s2) = a
        return gaussian(s1, s2, self.a_s) * heaviside_multi_channel(u1, u2, self.t, self.a_u)

    def compute_kernel_batches(self, u, s):
        mm = np.zeros(self.m)
        for i in range(self.m):
            mm[i] = self.compute_kernel(self.PHI[i], u, s, self.S[i])
        return mm.T

    def update_weights(self, pred, y, v, phi_p, s_p, lr):
        e = y - pred
        a_p = []
        for i in range(self.ns):
            print(v.shape)
            a_p.append((v[:, i, :] @ self.II.T @ e))
        a_p = np.array(a_p).T

        self.A = np.concatenate([self.A, lr * a_p], axis=0)
        self.PHI = self.PHI + phi_p
        self.S = np.concatenate([self.S, s_p], axis=0)
        self.m = self.m + len(phi_p)

    def update_quan_weights(self, pred, y, v, phi_p, s_p, lr, q):
        e = y - pred
        a_p = []
        if v.ndim != 3:
            print('v_dim: ', v.ndim)
            return
        for i in range(self.ns):
            a_p.append((v[:, i, :] @ self.II.T @ e))
        a_p = np.array(a_p).T

        m_p = len(phi_p)
        for j in range(m_p):
            # dis = np.zeros(self.m)
            # for i in range(self.m):
            #     dis[i] = 1 - self.compute_kernel(self.PHI[i], phi_p[j], self.S[i], s_p[j])

            pool = ThreadPool()
            dis = pool.map(self.compute_kernel_multi, zip(self.PHI, [phi_p[j]] * self.m, self.S, [s_p[j]] * self.m))
            pool.close()
            pool.join()

            index = np.argmin(dis)
            if np.min(dis) < q:
                self.A[index] = self.A[index] + a_p[j]
                # print('one memory saved')
            else:
                self.PHI = self.PHI + [phi_p[j]]
                self.S = np.concatenate([self.S, s_p[j][np.newaxis, :]], axis=0)
                self.A = np.concatenate([self.A, lr * a_p[j][np.newaxis, :]], axis=0)
                self.m = self.m + 1

    def train(self, u, y, epochs, lr):
        sz = y.shape[0]
        for i in range(epochs):
            print('epoch: ', i)
            j = 0
            for (du, dy) in zip(u, y):
                j = j + 1
                s_p = np.zeros((1, self.ns))
                phi_p = []
                v = []

                for frame in du:
                    k = self.compute_kernel_batches(Phi(frame), s_p[-1])
                    s_p = np.vstack([s_p, self.A.T @ k])
                    d_i = self.S - s_p[-2]
                    k_i = np.diag(k)
                    gamma_i = 2 * self.a_s * self.A.T @ k_i @ d_i
                    phi_p = phi_p + [Phi(frame)]

                    new_v = []
                    for vv in v:
                        new_v.append(gamma_i @ vv)
                    new_v.append(np.eye(self.ns))
                    v = new_v

                pred = self.II @ s_p[-1]
                # self.update_weights(pred, y, np.array(v), phi_p, s_p[:-1], lr)
                self.update_quan_weights(pred, dy, np.array(v), phi_p, s_p[:-1], lr, 0.2)
                logging.info("m: "+str(self.m) + ' loss_train: ' + str(np.mean((pred - dy)) ** 2))
                print('\rm: ', self.m, 'loss_train:', np.mean((pred - dy)) ** 2, 'progress: ', j / sz, end=' ')

    def test(self, u, y, index):
        loss_lt = []
        sz = np.size(index)
        corr = 0
        for (du, dy, label) in zip(u, y, index):
            s = np.zeros((1, self.ns))
            for frame in du:
                s = np.vstack([s, self.A.T @ self.compute_kernel_batches(Phi(frame), s[-1])])
            pred = self.II @ s[-1]
            loss_lt.append(np.sum((pred - dy) ** 2))
            if np.argmax(pred) == label:
                corr = corr + 1
        logging.info("avgloss_test: " + str(np.mean(loss_lt)) + ' accuracy: ' + str(corr / sz))
        print("avgloss_test: " + str(np.mean(loss_lt)) + ' accuracy: ' + str(corr / sz))
        return np.mean(loss_lt)






