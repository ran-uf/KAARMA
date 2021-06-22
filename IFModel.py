import numpy as np


def generate_impulse_train(data, fs, alpha=1, theta=0.05):
    impulse_train = []
    temp = 0

    for i in range(data.size):
        temp = np.exp(- alpha) * temp + data[i]
        if temp > theta:
            impulse_train.append(i / fs)
            temp = 0
    return impulse_train


def generate_impulse_train_multi_channels(data, fs, alpha, thetas):
    impulse_trains = []
    for i, theta in zip(data, thetas):
        impulse_trains.append(generate_impulse_train(i, fs, alpha, theta))
    return impulse_trains
