import os

import numpy as np
import time
import pickle


class Cluster:
    def __init__(self, a_s, a_u, u, s, a, alpha_r, e=None, m_k=None):

        self.a_s = a_s
        self.a_u = a_u
        self.ns = np.size(s)

        self.alpha = 1
        self.alpha_r = alpha_r

        self.S = s[np.newaxis, :]
        self.phi = u[np.newaxis, :]

        # if m_k is None:
        #     m_k = self.alpha_r
        #     # m_k = self.alpha * self.A @ self.A.T + self.alpha_r
        #     m_k = np.atleast_2d(m_k)
        #     m_k = np.linalg.inv(m_k)
        if np.ndim(a) == 1:
            a = a[:, np.newaxis]
        if e is not None:
            self.A = (a @ m_k @ e).T
        else:
            self.A = a.T
            m_k = self.alpha * self.A @ self.A.T + self.alpha_r
            m_k = np.atleast_2d(m_k)
            m_k = np.linalg.inv(m_k)

        self.center = (u, self.S[0])
        self.num = 1

        # self.b = self.alpha ** 2 * a @ m_k @ a.T
        self.b = self.alpha ** 2 * self.A.T @ m_k @ self.A

        self.a_k = None
        self.p_k = None
        self.dis = None

    def distance(self, u, s):
        return self.a_u * np.sum((self.center[0] - u) ** 2) + self.a_s * np.sum((self.center[1] - s) ** 2)

    def udistance(self, u):
        dis_u = (self.phi - u) ** 2
        dis_u = np.array([np.sum(dis_u[i]) for i in range(dis_u.shape[0])])
        return dis_u.reshape(-1)

    def sdistance(self, s):
        dis_s = (self.S - s) ** 2
        return np.array([np.sum(dis_s[i]) for i in range(dis_s.shape[0])])

    def update(self, u, s, a, dq):
        dis = self.a_s * self.sdistance(s) + self.a_u * self.udistance(u)
        index = np.argmin(dis)
        if dis[index] < dq:
            self.A[index] = self.A[index] + a.reshape(-1)
        else:
            self.A = np.concatenate((self.A, a.T), axis=0)
            self.phi = np.concatenate((self.phi, u[np.newaxis, :]), axis=0)
            self.S = np.concatenate((self.S, s[np.newaxis, :]), axis=0)
            self.center = ((self.center[0] * self.num + u) / (self.num + 1), (self.center[1] * self.num + s) / (self.num + 1))
            self.num += 1

    def m_k(self, u, s, a):
        dis = []
        p_k = []
        p_kk = []
        a_k = np.array([[]])
        m_0 = len(self.A)
        m_1 = len(u)
        for (uu, ss, aa) in zip(u, s, a):
            corr = np.exp(-self.a_s * self.sdistance(ss)) * np.exp(-self.a_u * self.udistance(uu))
            dis.append(1 - corr)

            p = np.zeros((m_0 * self.ns, self.ns))
            for mm in range(m_0):
                p[mm * self.ns:mm * self.ns + self.ns, :] = np.eye(self.ns) * corr[mm]
            p_k.append(p)

            dis_s_p = np.sum((s - ss) ** 2, axis=1)
            dis_u_p = (u - uu) ** 2
            dis_u_p = np.array([np.sum(dis_u_p[i]) for i in range(dis_u_p.shape[0])])
            # dis_u = np.sum(np.sum(dis_u, axis=1), axis=1)
            dis_u_p = dis_u_p.reshape(-1)
            corr_p = np.exp(-self.a_s * dis_s_p) * np.exp(-self.a_u * dis_u_p)

            p_p = np.zeros((m_1 * self.ns, self.ns))
            for mm in range(m_1):
                p_p[mm * self.ns:mm * self.ns + self.ns, :] = np.eye(self.ns) * corr_p[mm]
            p_kk.append(p_p)

            a_k = np.hstack((a_k, aa))

        self.p_k = np.hstack(p_k)
        p_kk = np.hstack(p_kk)
        self.a_k = a_k.T

        # print(np.size(self.a_k) / self.ns, len(u))
        # if np.size(self.a_k) / self.ns != len(u):
        #     print('ddddd')

        mk = self.alpha * self.a_k.T @ p_kk @ self.a_k - self.a_k.T @ self.p_k.T @ self.b @ self.p_k @ self.a_k

        self.dis = np.array(dis).T

        return mk

    def update_kalman(self, u, s, mk, e, dq):
        da = -self.b @ self.p_k @ self.a_k @ mk @ e
        da_p = self.alpha * self.a_k @ mk @ e

        db = -self.alpha * self.b @ self.p_k @ self.a_k @ mk @ self.a_k.T
        b = np.hstack((self.b, db))
        self.b = np.vstack((b, np.hstack((db.T, self.alpha ** 2 * self.a_k @ mk @ self.a_k.T))))

        da = da.reshape(-1, self.ns)
        da_p = da_p.reshape(-1, self.ns)
        self.A = self.A + da

        min_dis = np.min(self.dis, axis=0)
        index_A = np.argmin(self.dis, axis=0)
        index_merge = np.nonzero(min_dis <= dq)[0]
        index_keep = np.nonzero(min_dis > dq)[0]
        index_A = [index_A[pos] for pos in index_merge]

        if len(index_merge) == 0:
            self.A = np.concatenate((self.A, da_p), axis=0)
            self.phi = np.concatenate((self.phi, u), axis=0)
            self.S = np.concatenate((self.S, s), axis=0)

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
            self.phi = np.concatenate((self.phi, u[index_keep, :]), axis=0)
            self.S = np.concatenate((self.S, s[index_keep, :]), axis=0)
        self.num = len(self.A)


class NiceKAARMA:
    def __init__(self, ns, ny, a_s, a_u, u, dc, dq):
        self.ns = ns
        self.ny = ny
        self.a_s = a_s
        self.a_u = a_u

        self.dc = dc
        self.dq = dq

        self.II = np.zeros((ny, ns))
        self.II[:, ns - ny:] = np.eye(ny)

        self.clusters = []
        np.random.seed(0)
        s = np.random.random(ns)
        np.random.seed(1)
        a = np.random.random(ns)

        self.alpha_r = 1
        self.clusters.append(Cluster(a_s, a_u, u, s, a, self.alpha_r))

    def save_model(self, path):
        for (no, cluster) in enumerate(self.clusters):
            with open(path + '/%d_cluster.pkl' % no, 'wb+') as f:
                pickle.dump(cluster, f)

    def load_model(self, path):
        self.clusters.clear()
        for cc in os.walk(path):
            for c in cc[2]:
                if '.pkl' in c:
                    with open(path + '/' + c, 'rb+') as f:
                        self.clusters.append(pickle.load(f))

    def forward(self, u):
        ss = np.zeros((1, self.ns))
        for i in range(u.shape[0]):
            distance = [self.clusters[j].distance(u[i], ss) for j in range(len(self.clusters))]
            cluster = np.argmin(distance)
            di = self.clusters[cluster].S - ss
            k_s = np.exp(-self.a_s * np.sum(di ** 2, axis=1))[:, np.newaxis]
            diss = (self.clusters[cluster].phi - u[i]) ** 2
            k_u = np.exp(-self.a_u * np.array([np.sum(diss[j]) for j in range(self.clusters[cluster].num)]))[:,
                  np.newaxis]

            ki = k_s * k_u
            ss = self.clusters[cluster].A.T @ ki
            ss = ss.T
        pred = self.II @ ss.T
        return pred

    def test_one_sampe(self, x, y):
        pred = self.forward(x)
        # return np.sum((y - pred.reshape(-1)) ** 2), np.argmax(pred) == np.argmax(y)
        return np.sum((y - pred) ** 2), round(pred[0, 0]) == y

    def test(self, x, y):
        loss = []
        count = 0
        for (xx, yy) in zip(x, y):
            ls, result = self.test_one_sampe(xx, yy)
            loss.append(ls)
            count += result
        return np.mean(loss), count / len(loss)

    def train(self, x, y, lr, test_x=None, test_y=None):
        loss = []
        acc = []
        m = []
        loss_train = []
        step = 0
        for (u, d) in zip(x, y):
            step += 1
            print('\r', step, end='')
            # generate s-1
            # d = np.float64(d)
            s_p = np.zeros((u.shape[0], self.ns))
            phi = np.zeros(u.shape)
            v = np.zeros((u.shape[0], self.ns, self.ns))
            ss = np.zeros((1, self.ns))
            for j in range(u.shape[0]):
                distance = [self.clusters[i].distance(u[j], ss) for i in range(len(self.clusters))]
                cluster = np.argmin(distance)

                s_p[j] = ss.reshape(-1)
                phi[j] = u[j]
                di = self.clusters[cluster].S - ss
                # if di.dtype == 'object':
                #     print(di.dtype)
                k_s = np.exp(-self.a_s * np.sum(di ** 2, axis=1))[:, np.newaxis]
                diss = (self.clusters[cluster].phi - u[j]) ** 2
                k_u = np.exp(-self.a_u * np.array([np.sum(diss[i]) for i in range(self.clusters[cluster].num)]))[:, np.newaxis]
                ki = k_s * k_u
                # print(ki.tolist())
                ss = self.clusters[cluster].A.T @ ki
                ss = ss.T
                # print(ki.tolist())
                ki = np.diag(ki.reshape(-1))

                gamma_i = 2 * self.a_s * self.clusters[cluster].A.T @ ki
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
            loss_train.append(np.sum(e ** 2))
            # update weights
            start = max(0, len(s_p) - 10)
            num_steps = 0
            for (s, uu, vv) in zip(s_p, phi, v):
                num_steps += 1
                if num_steps > start:
                    distance = [self.clusters[i].distance(uu, s) for i in range(len(self.clusters))]
                    cluster = np.argmin(distance)
                    a = self.II @ vv
                    a = a.T @ e
                    if distance[cluster] > self.dc:
                        self.clusters.append(Cluster(self.a_s, self.a_u, uu, s, lr * a.reshape(-1), self.alpha_r))
                    else:
                        self.clusters[cluster].update(uu, s, lr * a, self.dq)

        if test_x is not None:
            loss_test = []
            num_test = 0
            for (test_xx, test_yy) in zip(test_x, test_y):
                ls, count = self.test_one_sampe(test_xx, test_yy)
                loss_test.append(ls)
                num_test = num_test + count

            print('\rloss_train: %05f' % loss_train[-1], 'loss_test: %05f' % np.mean(loss_test), ' acc_test: %05f' % (num_test / len(loss_test)), ' m:', [cc.num for cc in self.clusters])
            loss.append(np.mean(loss_test))
            acc.append(num_test / len(loss_test))
            m.append([cc.num for cc in self.clusters])

        return loss_train, loss, acc, m

    def train_kalman(self, x, y, test_x=None, test_y=None):
        loss = []
        acc = []
        m = []
        loss_train = []

        for (u, d) in zip(x, y):
            s_p = np.zeros((u.shape[0], self.ns))
            phi = np.zeros(u.shape)
            v = np.zeros((u.shape[0], self.ns, self.ns))
            ss = np.zeros((1, self.ns))
            for j in range(u.shape[0]):
                distance = [self.clusters[i].distance(u[j], ss) for i in range(len(self.clusters))]
                cluster = np.argmin(distance)

                s_p[j] = ss.reshape(-1)
                phi[j] = u[j]
                di = self.clusters[cluster].S - ss
                k_s = np.exp(-self.a_s * np.sum(di ** 2, axis=1))[:, np.newaxis]
                diss = (self.clusters[cluster].phi - u[j]) ** 2
                k_u = np.exp(-self.a_u * np.array([np.sum(diss[i]) for i in range(self.clusters[cluster].num)]))[:, np.newaxis]
                ki = k_s * k_u
                ss = self.clusters[cluster].A.T @ ki
                ss = ss.T
                ki = np.diag(ki.reshape(-1))

                gamma_i = 2 * self.a_s * self.clusters[cluster].A.T @ ki
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
            loss_train.append(np.sum(e ** 2))

            start = max(0, len(s_p) - 10)

            s_p = s_p[start:]
            phi = phi[start:]
            v = v[start:]
            a = []
            cluster = []

            for vv in v:
                a.append(self.II @ vv)

            for (s, uu) in zip(s_p, phi):
                distance = [self.clusters[i].distance(uu, s) for i in range(len(self.clusters))]
                if np.min(distance) <= self.dc:
                    cluster.append(np.argmin(distance))
                else:
                    cluster.append(-1)

            m_k = 0
            for idx in set(cluster):
                if idx != -1:
                    (pos, ) = np.nonzero(np.array(cluster) == idx)
                    m_k += self.clusters[idx].m_k(phi[pos], s_p[pos], np.array(a)[pos])
            m_k = np.linalg.inv(m_k + self.alpha_r * np.identity(self.ny))

            num_clusters = len(self.clusters)
            for idx in set(cluster):
                (pos, ) = np.nonzero(np.array(cluster) == idx)
                if idx == -1:
                    for nn in pos:
                        distance = [self.clusters[i].distance(uu, s) for i in range(num_clusters, len(self.clusters))]
                        if distance == []:
                            self.clusters.append(Cluster(self.a_s, self.a_u, phi[nn], s_p[nn], (a[nn]).reshape(-1), self.alpha_r, e, m_k))
                        else:
                            if np.min(distance) <= self.dc:
                                no = np.argmin(distance)
                                self.clusters[no + num_clusters].m_k(phi[[nn]], s_p[[nn]], np.array(a)[[nn]])
                                self.clusters[no + num_clusters].update_kalman(phi[[nn]], s_p[[nn]], m_k, e, self.dq)
                            else:
                                self.clusters.append(
                                    Cluster(self.a_s, self.a_u, phi[nn], s_p[nn], (a[nn]).reshape(-1), self.alpha_r, e,
                                            m_k))
                else:
                    self.clusters[idx].update_kalman(phi[pos], s_p[pos], m_k, e, self.dq)

            if test_x is not None:
                loss_test = []
                num_test = 0
                for (test_xx, test_yy) in zip(test_x, test_y):
                    ls, count = self.test_one_sampe(test_xx, test_yy)
                    loss_test.append(ls)
                    num_test = num_test + count

                print('\rloss_train: %05f' % loss_train[-1], 'loss_test: %05f' % np.mean(loss_test), ' acc_test: %05f' % (num_test / len(loss_test)), ' m:', [cc.num for cc in self.clusters])
                loss.append(np.mean(loss_test))
                acc.append(num_test / len(loss_test))
                m.append([cc.num for cc in self.clusters])

        return loss_train, loss, acc, m


if __name__ == '__main__':
    from tomita import generate_tomita4

    # x_train = []
    # y_train = []
    # for i in [6, 7, 8, 9, 10]:
    #     x, y = generate_tomita4(40, i)
    #     for xx, yy in zip(x, y):
    #         x_train.append(xx.T[:, np.newaxis])
    #         yyy = yy[-1]
    #         # yyy = np.zeros(2)
    #         # yyy[int(yy[-1])] = 1
    #         y_train.append(yyy)
    # x_train = np.array(x_train, dtype='object')
    # y_train = np.array(y_train, dtype='object')
    # x, y = generate_tomita4(100, 12)
    x_train = np.load('x_train_tomita.npy', allow_pickle=True)
    y_train = np.load('y_train_tomita.npy', allow_pickle=True)
    #

    # model = NiceKAARMA(5, 1, 2, 2, x_train[0][0], 0.5, 0.1)
    # ls_loss = []
    # ls_acc = []
    # lr = 0.1
    # for i in range(200):
    #     loss_train, loss, acc, m = model.train(x_train, y_train, 0.1, x_train, y_train)
    #     ls_loss.append(loss)
    #     ls_acc.append(acc)

    model = NiceKAARMA(5, 1, 2, 2, x_train[0][0], 3, 0.05)
    ls_loss = []
    ls_acc = []
    for i in range(200):
        loss_train, loss, acc, m = model.train_kalman(x_train, y_train, x_train, y_train)
        ls_loss.append(loss)
        ls_acc.append(acc)


############################################################
    # def toonehot(data, n):
    #     m = np.zeros((data.size, n))
    #     for i in range(data.size):
    #         m[i, int(data[i])] = 1
    #     return m
    #
    #
    # t = 0.025
    # fps = 100
    # folder = './data/small_01/'
    # test_y = np.load(folder + 'test_y.npy')
    # test_x = np.load(folder + 'test_x_final.npy', allow_pickle=True)
    # train_y = np.load(folder + 'train_y.npy')
    # train_x = np.load(folder + 'train_x_final.npy', allow_pickle=True)
    #
    # # train_x = np.concatenate((train_x, train_x, train_x))
    # # train_y = np.concatenate((train_y, train_y, train_y))
    #
    # num_data = np.size(train_y)
    # np.random.seed(0)
    # index = np.random.permutation(num_data)
    #
    # train_x_shuffle = []
    # train_y_shuffle = []
    # for i in index:
    #     train_x_shuffle.append(train_x[i])
    #     train_y_shuffle.append(train_y[i])
    #
    # train_x_shuffle = np.array(train_x_shuffle, dtype='object')
    #
    # train_y_oh = toonehot(np.array(train_y_shuffle), 2)
    # test_y_oh = toonehot(test_y, 2)
    #
    # model = NiceKAARMA(5, 2, 1, 4, train_x_shuffle[0][0], 0.1, 0.01)
    # print('50 5, 2, 1, 4, train_x_shuffle[0][0], 0.1, 0.01, on test')
    # tm = time.time()
    # loss_test = []
    # loss_train = []
    # acc_test = []
    # # pick = np.random.randint(576, size=30).tolist()
    # for i in range(5):
    #     print('epoch:', i)
    #     # model.train(train_x_shuffle, train_y_shuffle, 0.1, train_x_shuffle.tolist() + test_x.tolist(), train_y_shuffle + test_y.tolist())
    #     # model.train(train_x_shuffle, train_y_oh, 0.1, train_x_shuffle.tolist() + test_x.tolist(), train_y_oh.tolist() + test_y_oh.tolist())
    #     # model.train(train_x_shuffle, train_y_oh, 0.1, test_x.tolist(), test_y_oh.tolist())
    #
    #     model.train_kalman(train_x_shuffle, train_y_oh, test_x, test_y_oh)
    #     # loss_train, ls_test, ac_test, m = model.train(train_x, train_y, 1, test_x, test_y)
    # model.save_model('./model_50')
    # print('time: ', time.time() - tm)
