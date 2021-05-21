import numpy as np


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
