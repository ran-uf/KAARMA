import numpy as np


class Phi:
    def __init__(self, u):
        self.u = u


def heaviside(u1, u2, t, a):

    def pad(u, s, d):
        nu = np.zeros(s)
        aa = np.size(u)
        nu[:aa] = u
        for index in range(aa, s, 1):
            nu[index] = d
        return nu

    sz = np.max([np.size(u1), np.size(u2)])
    u1 = pad(u1, sz, t)
    u2 = pad(u2, sz, t)
    sm = 0
    for t1, t2 in zip(u1, u2):
        sm += (t1 - t2) ** 2
    return np.exp(- a * sm)


def heaviside_multi_channel(u1, u2, t, a):

    if isinstance(u1, Phi) & isinstance(u2, Phi):
        u1 = u1.u
        u2 = u2.u

    res = []
    for (u_1, u_2) in zip(u1, u2):
        res.append(heaviside(u_1, u_2, t, a))

    # pool = ThreadPool()
    # result = pool.map(heaviside, zip(u1, u2))
    # pool.close()
    # pool.join()
    return np.mean(res)


def gaussian(a1, a2, a):
    return np.exp(- a * np.sum((a1 - a2) ** 2))
