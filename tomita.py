import numpy as np


def label(x):
    num_0 = 0
    for a in x:
        if a == 0:
            num_0 = num_0 + 1
            if num_0 >= 3:
                return 0
        elif a == 1:
            num_0 = 0
    return 1


def tomita4(length, lbl):
    while 1:
        a = np.random.random(length) > 0.5
        a = a.astype(np.float32)
        if label(a) == lbl:
            return a


def generate_tomita4(num, length):
    strings = np.zeros((num, length))
    ll = np.zeros((num, length))
    for i in range(0, num, 2):
        ll[i] = 1
        ll[i + 1] = 0
        strings[i, :] = tomita4(length, 1)
        strings[i + 1, :] = tomita4(length, 0)

    return strings, ll


def tomita4_string(num, length):
    strings = np.zeros((num, length))
    ll = np.zeros((num, length))
    for i in range(num):
        a = np.random.random(length) > 0.5
        a = a.astype(np.float32)
        count = 0
        for j in range(length):
            if a[j] == 0:
                count += 1
                if count >= 3:
                    ll[i] = 0


if __name__ == '__main__':
    stringss = generate_tomita4(6, 10)
    print('done')
