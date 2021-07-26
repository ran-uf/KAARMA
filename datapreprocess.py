import numpy as np
from gtfblib import Chen
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pickle
from IFModel import generate_impulse_train_multi_channels


DEBUG = False
folder = './data/full/'
# train_x_raw = np.load(folder + 'x_small.npy', allow_pickle=True)
train_x_raw = np.load(folder + 'train_x.npy', allow_pickle=True)
test_x_raw = np.load(folder + 'test_x.npy', allow_pickle=True)

train_x_normalize = []
test_x_normalize = []
for x in train_x_raw:
    train_x_normalize.append(x.astype(np.float64) / np.max(x))
for x in test_x_raw:
    test_x_normalize.append(x.astype(np.float64) / np.max(x))

fil = Chen(fs=8000)

train_x_gtf = []
test_x_gtf = []
num = 0
for x in train_x_normalize:
    a = fil.process(x).real
    a = a / np.max(abs(a), axis=1)[:, np.newaxis] * 4
    train_x_gtf.append(a)
    num += 1
for x in test_x_normalize:
    a = fil.process(x).real
    a = a / np.max(a, axis=1)[:, np.newaxis] * 4
    test_x_gtf.append(a)

train_impulse_trains = []
test_impulse_trains = []

alpha = 0.05
theta = 1000

freq = np.array([99, 192, 341, 501, 685, 908, 1134, 1473, 1821, 2200, 2720, 3326, 4075, 4990, 5977, 7283])

for i in train_x_gtf:
    train_impulse_trains.append(generate_impulse_train_multi_channels(i, 8000, alpha, theta / freq))

with open(folder + 'train_x_impulses_normalize.txt', 'wb') as f:
    pickle.dump(train_impulse_trains, f)
    f.close()

for i in test_x_gtf:
    test_impulse_trains.append(generate_impulse_train_multi_channels(i, 8000, alpha, theta / freq))

with open(folder + 'test_x_impulses_normalize.txt', 'wb') as f:
    pickle.dump(test_impulse_trains, f)
    f.close()

count = []
for i in range(len(train_impulse_trains)):
    for channel in range(16):
        mean_count = len(train_impulse_trains[i][channel]) / (train_x_gtf[i][channel].size / 8) * 25
        count.append(mean_count)

for i in range(len(test_impulse_trains)):
    for channel in range(16):
        mean_count = len(test_impulse_trains[i][channel]) / (test_x_gtf[i][channel].size / 8) * 25
        count.append(mean_count)

print('min:', np.mean(count), ' max:', np.max(count))


def extract_frame(t, t_s, t_e):
    ls = []
    for tt in t:
        if t_s < tt < t_e:
            if DEBUG:
                if tt < t_s:
                    print('extract_frame ERROR!')
            ls.append(tt - t_s)
    return ls


def split_frame(impulse_train, ms, fps):
    m = np.zeros(len(impulse_train))
    for (i, train) in enumerate(impulse_train):
        if train:
            m[i] = np.nanmax(np.array(impulse_train[i]))
    m = np.max(m)
    frames = []
    for t0 in np.arange(0, m, fps):
        channel = []
        for i in range(len(impulse_train)):
            channel.append(extract_frame(impulse_train[i], t0, t0 + ms))
        frames.append(channel)
    return frames


t = 0.025
fps = 100

train_x_frame = []
test_x_frame = []

for x in train_impulse_trains:
    train_x_frame.append(split_frame(x, t, 1 / fps))

for x in test_impulse_trains:
    test_x_frame.append(split_frame(x, t, 1 / fps))

max_num_spikes = 0
for x in train_x_frame:
    for f in x:
        for channel in f:
            length = len(channel)
            # print(length)
            if length > max_num_spikes:
                max_num_spikes = length
for x in test_x_frame:
    for f in x:
        for channel in f:
            length = len(channel)
            # print(length)
            if length > max_num_spikes:
                max_num_spikes = length

print('max number of spikes in one frame: ', max_num_spikes)


# pad
def pad(u, s, d):
    nu = np.ones(s) * d
    nu[:np.size(u)] = u
    return nu


def pad_one_sample(sample, s, d):
    n_frames = len(sample)
    n_channels = len(sample[0])
    res = np.ones((n_frames, n_channels, s)) * d
    for frame in range(n_frames):
        for chan in range(n_channels):
            data = sample[frame][chan]
            res[frame, chan, :len(data)] = np.array(data)
    return res


train_x = []
test_x = []
for x in train_x_frame:
    train_x.append(pad_one_sample(x, max_num_spikes, t))
for x in test_x_frame:
    test_x.append(pad_one_sample(x, max_num_spikes, t))

if DEBUG:
    for x in train_x:
        for frame in x:
            for chan in frame:
                if not all(x <= y for x, y in zip(chan, chan[1:])):
                    print('not ascend')
    # for x in test_x:
    #     for frame in x:
    #         for chan in frame:
    #             if not all(x <= y for x, y in zip(chan, chan[1:])):
    #                 print('not ascend')

np.save(folder + 'train_x_final.npy', np.array(train_x, dtype='object'))
np.save(folder + 'test_x_final.npy', np.array(test_x, dtype='object'))

# x = np.load(folder + 'x_final.npy')
# y = np.load(folder + 'y_small.npy')

