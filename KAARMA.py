from kernels import heaviside_multi_channel, gaussian, Phi, norm_spikes_multi_channel
import numpy as np
import logging
from multiprocessing.dummy import Pool as ThreadPool
import time


SHOW_TIME = False


class NICE:
    def __init__(self, dc, dq, ns, lr):
        self.dc = dc
        self.dq = dq
        self.lr = lr
        self.ns = ns
        self.phis = []
        self.S = []
        self.A = []
        self.nClusters = 0

    def initialize(self, phi, s, y1):
        self.A = self.A.append(y1.reshape((1, self.ns)) * self.lr)
        self.phis.append([phi])
        self.S = self.S.append(s.reshape((self.ns, 1)))
        self.nClusters = 1

    def compute_distance(self, i, phi, s):
        # for (phi0, s0) in zip(self.phis, self.S):
        return

    def update(self, phi, s, e):
        dis = np.zeros(self.nClusters)
        for i in range(self.nClusters):
            dis[i] = self.compute_distance(i, phi, s)


class KAARMA:
    def __init__(self, ns, ny, a_s, a_u, u_type, kfunc):
        self.ns = ns
        self.ny = ny
        self.a_s = a_s
        self.a_u = a_u
        self.S = 2 * np.random.random((1, ns)) - 1
        self.phi = u_type[np.newaxis, :]
        self.A = 2 * np.random.random((1, ns)) - 1
        self.II = np.zeros((ny, ns))
        self.II[:, ns - ny:] = np.eye(ny)

        self.kfunc = kfunc

    def compute_kernel_u(self, u):
        return self.kfunc(self.phi, u, self.a_u)
        # return np.exp(-self.a_u * (self.phi - u) ** 2)

    def update_weights(self, pred, y, v, phi_p, s_p, lr):
        e = y - pred
        a_p = []
        for i in range(self.ns):
            # print(v.shape)
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
            dis = np.zeros(self.m)
            for i in range(self.m):
                # dis[i] = 1 - self.compute_kernel(self.PHI[i], phi_p[j], self.S[i], s_p[j])
                dis[i] = self.a_s * np.sum((self.S[i] - s_p[j]) ** 2) + self.a_u * norm_spikes_multi_channel(self.PHI[i], phi_p[j])

            # pool = ThreadPool()
            # dis = pool.map(self.compute_kernel_multi, zip(self.PHI, [phi_p[j]] * self.m, self.S, [s_p[j]] * self.m))
            # pool.close()
            # pool.join()

            index = np.argmin(dis)
            # print(np.min(dis))
            if np.min(dis) < q:
                self.A[index] = self.A[index] + a_p[j]
                # print('one memory saved')
            else:
                self.PHI = self.PHI + [phi_p[j]]
                self.S = np.concatenate([self.S, s_p[j][np.newaxis, :]], axis=0)
                self.A = np.concatenate([self.A, lr * a_p[j][np.newaxis, :]], axis=0)
                self.m = self.m + 1

    def forward(self, x):
        s = np.zeros((1, self.ns))
        for f in x:
            di = self.S - s
            k_s = np.exp(-self.a_s * np.sum(di ** 2, axis=1))
            k_u = np.exp(-self.a_u * (self.phi - f) ** 2)
            ki = k_s * k_u.reshape(-1)
            s = self.A.T @ ki
        pred = self.II @ s

        return pred

    def test_one_sampe(self, x, y):
        pred = self.forward(x)
        return np.sum((y - pred) ** 2), np.argmax(pred) == np.argmax(y)

    def train(self, x, y, lr, dq):
        for (u, d) in zip(x, y):
            # generate s-1
            d = np.float64(d)
            s_p = []
            phi = []
            v = []
            ss = np.random.random((1, self.ns))
            for j in range(u.shape[0]):
                s_p.append(ss)
                phi.append(u[j])
                di = self.S - ss
                if di.dtype == 'object':
                    print(di.dtype)
                k_s = np.exp(-self.a_s * np.sum(di ** 2, axis=1))[:, np.newaxis]
                k_u = self.compute_kernel_u(u[j])
                ki = k_s * k_u
                ss = self.A.T @ ki
                ss = ss.T
                # print(ki.tolist())
                ki = np.diag(ki.reshape(-1))

                gamma_i = 2 * self.a_s * self.A.T @ ki
                if gamma_i.ndim == 1:
                    gamma_i = gamma_i[:, np.newaxis]
                gamma_i = gamma_i @ di
                if j == 0:
                    v.append(np.eye(self.ns))
                else:
                    for index in range(len(v)):
                        v[index] = gamma_i @ v[index]
                    v.append(np.eye(self.ns))
            pred = self.II @ ss.T

            e = np.atleast_2d(d).T - pred
            # if i == u.shape[0] - 1:
            #     print('\rerror:', e, ' m:', self.A.shape[0])

            # update weights
            start = max(0, len(s_p) - 5)
            num_steps = 0
            for (s, uu, vv) in zip(s_p, phi, v):
                num_steps += 1
                if num_steps > start:
                    m = self.A.shape[0]
                    # if m > 2000:
                    #    print('bug')
                    dis_s = np.sum((self.S - s) ** 2, axis=1)
                    dis_u = (self.phi - uu) ** 2
                    dis_u = dis_u.reshape(-1)
                    dis = self.a_s * dis_s + self.a_u * dis_u
                    dis = dis[:m]
                    index = np.argmin(dis)
                    a = self.II @ vv
                    a = a.T @ e

                    if dis[index] < dq:
                        a = a.reshape(-1)
                        self.A[index] = self.A[index] + lr * a
                    else:
                        # print(self.S.shape, s.shap
                        self.A = np.concatenate((self.A, lr * a.T), axis=0)
                        self.phi = np.concatenate((self.phi, np.array([uu])[np.newaxis, :]), axis=0)
                        self.S = np.concatenate((self.S, s), axis=0)

        loss_train = []
        num_train = 0
        for (train_x, train_y) in zip(x, y):
            ls, count = self.test_one_sampe(train_x, train_y)
            loss_train.append(ls)
            num_train = num_train + count

        print('\rloss_train: ', np.mean(loss_train), ' acc_train:', num_train / len(loss_train), ' m:', self.A.shape[0])

        return np.mean(loss_train), num_train / len(loss_train)

    def test(self, u, y, index):
        loss_lt = []
        sz = np.size(index)
        corr = 0
        for (du, dy, label) in zip(u, y, index):
            s = np.zeros((1, self.ns))
            for frame in du:
                b = self.compute_kernel_batches(frame, s[-1])
                s = np.vstack([s, self.A.T @ self.compute_kernel_batches(frame, s[-1])])
            pred = self.II @ s[-1]
            loss_lt.append(np.mean((pred - dy) ** 2))

            if np.argmax(pred) == label:
                corr = corr + 1
        # logging.info("avgloss_test: " + str(np.mean(loss_lt)) + ' accuracy: ' + str(corr / sz))
        # print("\ravgloss_test: " + str(np.mean(loss_lt)) + ' accuracy: ' + str(corr / sz))
        acc = corr / sz
        return np.mean(loss_lt), acc


if __name__ == '__main__':
    from tomita import generate_tomita4
    from kernels import gaussian

    x_train = []
    y_train = []
    for i in [6, 7, 8, 9, 10]:
        x, y = generate_tomita4(40, i)
        for xx, yy in zip(x, y):
            x_train.append(xx)
            # yyy = yy[-1]
            yyy = np.zeros(2)
            yyy[int(yy[-1])] = 1
            y_train.append(yyy)
    x_train = np.array(x_train, dtype='object')
    y_train = np.array(y_train, dtype='object')

    model = KAARMA(6, 2, 2, 2, np.array([x[0, 0]]), gaussian)
    ls_loss = [1]
    ls_acc = []
    lr = 0.1
    for i in range(100):
        # model.train_1(x_train, y_train, .05, 0.01)
        loss, acc = model.train(x_train, y_train, lr, 0.5)
        if loss < 0.95 * ls_loss[-1]:
            lr = 0.95 * lr
        ls_loss.append(loss)
        ls_acc.append(acc)
