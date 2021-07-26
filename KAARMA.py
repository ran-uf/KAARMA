from kernels import heaviside_multi_channel, gaussian, Phi, norm_spikes_multi_channel
import numpy as np
import logging
from multiprocessing.dummy import Pool as ThreadPool
import time


class KAARMA:
    def __init__(self, ns, ny, a_s, a_u, u_type, kfunc):
        self.ns = ns
        self.ny = ny
        self.a_s = a_s
        self.a_u = a_u
        np.random.seed(0)
        self.S = np.random.random((1, ns))
        self.phi = u_type[np.newaxis, :]
        np.random.seed(1)
        self.A = np.random.random((1, ns))
        self.II = np.zeros((ny, ns))
        self.II[:, ns - ny:] = np.eye(ny)

        self.alpha = 1
        self.alpha_r = 1
        m_k = self.alpha * self.A @ self.A.T + self.alpha_r
        m_k = np.atleast_2d(m_k)
        m_k = np.linalg.inv(m_k)
        self.b = self.alpha ** 2 * self.A.T @ m_k @ self.A

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
            k_u = self.compute_kernel_u(f)
            ki = k_s * k_u.reshape(-1)
            s = self.A.T @ ki
        pred = self.II @ s

        return pred

    def test_one_sampe(self, x, y):
        pred = self.forward(x)
        # return np.sum((y - pred) ** 2), np.argmax(pred) == np.argmax(y)
        return np.sum((y - pred) ** 2), round(pred[0]) == y

    def train(self, x, y, lr, dq, test_x=None, test_y=None):
        min_loss = 2
        num_batches = 8
        for (u, d) in zip(x, y):

            s_p = np.zeros((u.shape[0], self.ns))
            phi = np.zeros(u.shape)
            v = np.zeros((u.shape[0], self.ns, self.ns))
            ss = np.zeros((1, self.ns))
            for j in range(u.shape[0]):
                s_p[j] = ss.reshape(-1)
                phi[j] = u[j]
                di = self.S - ss
                # if di.dtype == 'object':
                #     print(di.dtype)
                k_s = np.exp(-self.a_s * np.sum(di ** 2, axis=1))[:, np.newaxis]
                k_u = self.compute_kernel_u(u[j])
                ki = k_s * k_u
                # print(ki.tolist())
                ss = self.A.T @ ki
                ss = ss.T
                # print(ki.tolist())
                ki = np.diag(ki.reshape(-1))

                gamma_i = 2 * self.a_s * self.A.T @ ki
                if gamma_i.ndim == 1:
                    gamma_i = gamma_i[:, np.newaxis]
                gamma_i = gamma_i @ di
                if j == 0:
                    v[j] = np.eye(self.ns)
                else:
                    for index in range(len(v)):
                        v[index] = gamma_i @ v[index]
                    v[j] = np.eye(self.ns)
            pred = self.II @ ss.T

            e = np.atleast_2d(d).T - pred
            # if i == u.shape[0] - 1:
            #     print('\rerror:', e, ' m:', self.A.shape[0])

            # update weights
            start = max(0, len(s_p) - 10) + 1
            num_steps = 0

            s_p = s_p[start:]
            phi = phi[start:]
            v = v[start:]

            for (s, uu, vv) in zip(s_p, phi, v):
                num_steps += 1
                # if num_steps > start:
                m = self.A.shape[0]
                # if m > 2000:
                #    print('bug')
                dis_s = np.sum((self.S - s) ** 2, axis=1)
                dis_u = (self.phi - uu) ** 2
                dis_u = np.array([np.sum(dis_u[i]) for i in range(dis_u.shape[0])])
                # dis_u = np.sum(np.sum(dis_u, axis=1), axis=1)
                dis_u = dis_u.reshape(-1)
                # dis = self.a_s * dis_s + self.a_u * dis_u
                dis = 1 - np.exp(-self.a_s * dis_s) * np.exp(-self.a_u * dis_u)
                dis = dis[:m]
                index = np.argmin(dis)
                a = self.II @ vv
                a = a.T @ e

                if dis[index] < dq:
                    a = a.reshape(-1)
                    self.A[index] = self.A[index] + lr * a
                else:
                    self.A = np.concatenate((self.A, lr * a.T), axis=0)
                    self.phi = np.concatenate((self.phi, uu[np.newaxis, :]), axis=0)
                    # self.phi = np.concatenate((self.phi, np.array([[uu]])), axis=0)
                    self.S = np.concatenate((self.S, s[np.newaxis, :]), axis=0)

        loss_train = []
        num_train = 0
        for (train_x, train_y) in zip(x, y):
            ls, count = self.test_one_sampe(train_x, train_y)
            loss_train.append(ls)
            num_train = num_train + count

        print('\rloss_train: ', np.mean(loss_train), ' acc_train:', num_train / len(loss_train), ' m:', self.A.shape[0])

        return np.mean(loss_train), num_train / len(loss_train), self.A.shape[0]

    def train_kalman(self, x, y, dq):

        for (u, d) in zip(x, y):
            s_p = np.zeros((u.shape[0], self.ns))
            phi = np.zeros(u.shape)
            v = np.zeros((u.shape[0], self.ns, self.ns))
            ss = np.zeros((1, self.ns))
            for j in range(u.shape[0]):

                s_p[j] = ss.reshape(-1)
                phi[j] = u[j]
                di = self.S - ss
                k_s = np.exp(-self.a_s * np.sum(di ** 2, axis=1))[:, np.newaxis]
                k_u = self.compute_kernel_u(u[j])
                ki = k_s * k_u
                ss = self.A.T @ ki
                ss = ss.T
                ki = np.diag(ki.reshape(-1))

                gamma_i = 2 * self.a_s * self.A.T @ ki
                if gamma_i.ndim == 1:
                    gamma_i = gamma_i[:, np.newaxis]
                gamma_i = gamma_i @ di
                if j == 0:
                    v[j] = np.eye(self.ns)
                else:
                    for index in range(len(v)):
                        v[index] = gamma_i @ v[index]
                    v[j] = np.eye(self.ns)
            pred = self.II @ ss.T

            e = np.atleast_2d(d).T - pred

            # update weights
            start = max(0, len(s_p) - 10)
            s_p = s_p[start:]
            phi = phi[start:]
            v = v[start:]
            num_steps = 0
            a = np.array([[]])
            p_k = []
            p_kk = []
            dis = []
            for (s, uu, vv) in zip(s_p, phi, v):
                num_steps += 1
                m = self.A.shape[0]

                dis_s = np.sum((self.S - s) ** 2, axis=1)
                dis_u = (self.phi - uu) ** 2
                dis_u = np.array([np.sum(dis_u[i]) for i in range(dis_u.shape[0])])
                # dis_u = np.sum(np.sum(dis_u, axis=1), axis=1)
                dis_u = dis_u.reshape(-1)
                corr = np.exp(-self.a_s * dis_s) * np.exp(-self.a_u * dis_u)
                # todo
                # dis.append(self.a_s * dis_s + self.a_u * dis_u)
                dis.append(1 - np.exp(-self.a_s * dis_s) * np.exp(-self.a_u * dis_u))
                p = np.zeros((m * self.ns, self.ns))
                for mm in range(m):
                    p[mm * self.ns:mm * self.ns + self.ns, :] = np.eye(self.ns) * corr[mm]
                p_k.append(p)
                # dis = dis[:m]
                # index = np.argmin(dis)

                m = s_p.shape[0]
                dis_s_p = np.sum((s_p - s) ** 2, axis=1)
                dis_u_p = (phi - uu) ** 2
                dis_u_p = np.array([np.sum(dis_u_p[i]) for i in range(dis_u_p.shape[0])])
                # dis_u = np.sum(np.sum(dis_u, axis=1), axis=1)
                dis_u_p = dis_u_p.reshape(-1)
                corr_p = np.exp(-self.a_s * dis_s_p) * np.exp(-self.a_u * dis_u_p)

                p_p = np.zeros((m * self.ns, self.ns))
                for mm in range(m):
                    p_p[mm * self.ns:mm * self.ns + self.ns, :] = np.eye(self.ns) * corr_p[mm]
                p_kk.append(p_p)

                a = np.hstack((a, self.II @ vv))

            p_k = np.hstack(p_k)
            p_kk = np.hstack(p_kk)
            a = a.T

            mk = self.alpha * a.T @ p_kk @ a - a.T @ p_k.T @ self.b @ p_k @ a + self.alpha_r
            mk = np.linalg.inv(np.atleast_2d(mk))

            da = -self.b @ p_k @ a @ mk @ e
            da_p = self.alpha * a @ mk @ e

            db = -self.alpha * self.b @ p_k @ a @ mk @ a.T
            b = np.hstack((self.b, db))
            self.b = np.vstack((b, np.hstack((db.T, self.alpha ** 2 * a @ mk @ a.T))))

            da = da.reshape(-1, self.ns)
            da_p = da_p.reshape(-1, self.ns)
            self.A = self.A + da

            dis = np.array(dis).T
            min_dis = np.min(dis, axis=0)
            index_A = np.argmin(dis, axis=0)
            index_merge = np.nonzero(min_dis <= dq)[0]
            index_keep = np.nonzero(min_dis > dq)[0]
            index_A = [index_A[pos] for pos in index_merge]

            if len(index_merge) == 0:
                self.A = np.concatenate((self.A, da_p), axis=0)
                # print(u == phi)
                self.phi = np.concatenate((self.phi, phi), axis=0)
                self.S = np.concatenate((self.S, s_p), axis=0)

            else:
                # merge
                self.A[index_A, :] = self.A[index_A, :] + da_p[index_merge, :]
                # rewrite b
                n_a = len(self.A)
                ind_k = [na for na in range(n_a * self.ns)]
                for i_k in index_keep:
                    st = (i_k + n_a) * self.ns
                    ind_k = ind_k + [na for na in range(st, st + self.ns)]
                ind_m = []
                for i_m in index_merge:
                    st = (i_m + n_a) * self.ns
                    ind_m = ind_m + [na for na in range(st, st + self.ns)]
                ind_a = []
                for i_a in index_A:
                    st = i_a * self.ns
                    ind_a = ind_a + [na for na in range(st, st + self.ns)]
                b = self.b[ind_k, :]
                b[ind_a, :] = b[ind_a, :] + self.b[ind_m, :]
                self.b = b
                b = self.b[:, ind_k]
                b[:, ind_a] = b[:, ind_a] + self.b[:, ind_m]
                self.b = b

                self.A = np.concatenate((self.A, da_p[index_keep, :]), axis=0)
                self.phi = np.concatenate((self.phi, phi[index_keep, :]), axis=0)
                self.S = np.concatenate((self.S, s_p[index_keep, :]), axis=0)

        loss_train = []
        num_train = 0
        for (train_x, train_y) in zip(x, y):
            ls, count = self.test_one_sampe(train_x, train_y)
            loss_train.append(ls)
            num_train = num_train + count
        print('\rloss_train: ', np.mean(loss_train), ' acc_train:', num_train / len(loss_train), ' m:', self.A.shape[0])

        return np.mean(loss_train), num_train / len(loss_train), self.A.shape[0]

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

    # x_train = []
    # y_train = []
    # for i in [6]:
    #     x, y = generate_tomita4(40, i)
    #     for xx, yy in zip(x, y):
    #         x_train.append(xx.T[:, np.newaxis])
    #         yyy = yy[-1]
    #         # yyy = np.zeros(2)
    #         # yyy[int(yy[-1])] = 1
    #         y_train.append(yyy)
    # x_train = np.array(x_train, dtype='object')
    # y_train = np.array(y_train, dtype='object')

    # np.save('x_train_tomita.npy', x_train)
    # np.save('y_train_tomita.npy', y_train)

    x_train = np.load('x_train_tomita.npy', allow_pickle=True)
    y_train = np.load('y_train_tomita.npy', allow_pickle=True)
    x, y = generate_tomita4(100, 12)

    # model = NiceKAARMA(6, 1, 2, 2, np.array([x[0, 0]]), 100000, 0.01)
    ls_loss_gd = []
    ls_acc_gd = []
    m_size_gd = []
    lr = 0.1
    # model = KAARMA(5, 1, 2, 2, x_train[0][0], gaussian)
    # print('SGD')
    # for i in range(20):
    #     loss, acc, m = model.train(x_train, y_train, lr, 0.1, x, y)
    #     ls_loss_gd.append(loss)
    #     ls_acc_gd.append(acc)
    #     m_size_gd.append(m)

    ls_loss_kf = []
    ls_acc_kf = []
    m_size_kf = []
    model = KAARMA(5, 1, 2, 2, x_train[0][0], gaussian)
    print('KF')
    for i in range(20):
        loss, acc, m = model.train_kalman(x_train, y_train, 0.1)
        ls_loss_kf.append(loss)
        ls_acc_kf.append(acc)
        m_size_kf.append(m)

    import matplotlib.pyplot as plt
    plt.title('learning curve')
    plt.ylabel('loss')
    plt.xlabel('epoches')
    plt.plot(ls_loss_gd, label='SGD')
    plt.plot(ls_loss_kf, label='KF')
    plt.legend()
    plt.show()

    plt.title('memory size')
    plt.xlabel('epoches')
    plt.plot(m_size_gd, label='SGD')
    plt.plot(m_size_kf, label='KF')
    plt.legend()
    plt.show()
