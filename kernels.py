import numpy as np


class Phi:
    def __init__(self, u):
        self.u = u


def norm_spikes(u1, u2):
    if np.size(u1) == 0:
        return np.sum((0 - u2) ** 2)
    return np.sum((u1 - u2) ** 2)


def norm_spikes_multi_channel(u1, u2):
    sm = 0
    for (u_1, u_2) in zip(u1, u2):
        sm += norm_spikes(u_1, u_2)
    return sm


def heaviside(u1, u2, a):
    # print('norm_spikes: ', norm_spikes(u1, u2))
    return norm_spikes(u1, u2)


def heaviside_multi_channel(u1, u2, a):
    if isinstance(u1, Phi) & isinstance(u2, Phi):
        u1 = u1.u
        u2 = u2.u

    res = []
    for (u_1, u_2) in zip(u1, u2):
        res.append(heaviside(u_1, u_2, a))

    # pool = ThreadPool()
    # result = pool.map(heaviside, zip(u1, u2))
    # pool.close()
    # pool.join()
    return np.exp(-a * np.mean(res))


def gaussian(a1, a2, a):
    # print('norm_states:', np.sum((a1 - a2) ** 2))
    a1 = a1.astype(np.float)
    a2 = a2.astype(np.float)
    return np.exp(- a * (a1 - a2) ** 2)


def gaussian_spikes(a1, a2, a):
    return np.exp(- a * np.sum(np.sum((a1 - a2) ** 2, axis=1), axis=1, keepdims=True))
