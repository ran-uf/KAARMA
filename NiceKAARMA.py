import numpy as np
import time


class Cluster:
    def __init__(self, a_s, a_u, u, s, a):

        self.a_s = a_s
        self.a_u = a_u

        self.S = s[np.newaxis, :]
        self.phi = u[np.newaxis, :]
        self.A = a[np.newaxis, :]

        self.center = (u, self.S[0])
        self.num = 1

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
            # print(self.S.shape, s.shap
            self.A = np.concatenate((self.A, a.T), axis=0)
            self.phi = np.concatenate((self.phi, u[np.newaxis, :]), axis=0)
            # self.phi = np.concatenate((self.phi, np.array([[uu]])), axis=0)
            self.S = np.concatenate((self.S, s[np.newaxis, :]), axis=0)
            self.center = ((self.center[0] * self.num + u) / (self.num + 1), (self.center[1] * self.num + s) / (self.num + 1))
            self.num += 1


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
        # np.random.seed(0)
        s = 2 * np.random.random(ns) - 1
        # np.random.seed(0)
        a = 2 * np.random.random(ns) - 1
        self.clusters.append(Cluster(a_s, a_u, u, a, s))

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
        return np.sum((y - pred.reshape(-1)) ** 2), np.argmax(pred) == np.argmax(y)
        # return np.sum((y - pred) ** 2), round(pred[0, 0]) == y

    def train(self, x, y, lr, test_x=None, test_y=None):
        loss = []
        acc = []
        m = []
        loss_train = []
        for (u, d) in zip(x, y):
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
            # print('\rerror:', e)

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
                        self.clusters.append(Cluster(self.a_s, self.a_u, uu, s, lr * a.reshape(-1)))
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

            # lr = 0.99 * lr
            # if lr < 0.01:
            #     lr = 0.1
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
    # x, y = generate_tomita4(40, 12)
    # # x_train = np.load('x_train_tomita.npy', allow_pickle=True)
    # # y_train = np.load('y_train_tomita.npy', allow_pickle=True)
    # #
    # model = NiceKAARMA(6, 1, 2, 2, x_train[0][0], 2, 0.01)
    # ls_loss = [1]
    # ls_acc = []
    # lr = 0.1
    # for i in range(200):
    #     # model.train_1(x_train, y_train, .05, 0.01)
    #     # loss, acc = model.train(x_train, y_train, lr, 0.1)
    #     loss_train, loss, acc, m = model.train(x_train, y_train, lr, x, y[:, -1])
    #     # if loss < 0.95 * ls_loss[-1]:
    #     #     lr = 0.95 * lr
    #     ls_loss.append(loss)
    #     ls_acc.append(acc)

    def toonehot(data, n):
        m = np.zeros((data.size, n))
        for i in range(data.size):
            m[i, int(data[i])] = 1
        return m


    t = 0.025
    fps = 100

    test_y = np.load('./data/small_01/test_y_small.npy')
    test_x = np.load('./data/small_01/test_x_final.npy', allow_pickle=True)
    train_y = np.load('./data/small_01/train_y_small.npy')
    train_x = np.load('./data/small_01/train_x_final.npy', allow_pickle=True)

    # train_x = np.concatenate((train_x, train_x, train_x))
    # train_y = np.concatenate((train_y, train_y, train_y))

    num_data = np.size(train_y)
    np.random.seed(0)
    index = np.random.permutation(num_data)

    train_x_shuffle = []
    train_y_shuffle = []
    for i in index:
        train_x_shuffle.append(train_x[i])
        train_y_shuffle.append(train_y[i])

    train_y_oh = toonehot(np.array(train_y_shuffle), 2)
    test_y_oh = toonehot(test_y, 2)

    model = NiceKAARMA(5, 2, 1, 4, train_x_shuffle[0][0], 0.1, 0.01)
    print('50 5, 2, 1, 4, train_x_shuffle[0][0], 0.1, 0.01, on test')
    tm = time.time()
    loss_test = []
    loss_train = []
    acc_test = []
    for i in range(1):
        print('epoch:', i)
        # model.train(train_x_shuffle[:50], train_y_oh[:50], 0.1, test_x, test_y_oh)
        loss_train, ls_test, ac_test, m = model.train(train_x_shuffle, train_y_oh, 0.1, test_x, test_y_oh)
    print('time: ', time.time() - tm)
