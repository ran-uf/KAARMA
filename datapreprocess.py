import numpy as np
from gtfblib import Chen
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pickle
from IFModel import generate_impulse_train_multi_channels


train_x_raw = np.load('train_x_raw.npy', allow_pickle=True)
test_x_raw = np.load('test_x_raw.npy', allow_pickle=True)

train_x_normalize = []
test_x_normalize = []
for x in train_x_raw:
    train_x_normalize.append(x / np.max(x))
for x in test_x_raw:
    test_x_normalize.append(x / np.max(x))

fil = Chen(fs=8000)

train_x_gtf = []
test_x_gtf = []
for x in train_x_normalize:
    train_x_gtf.append(fil.process(x).real)
for x in test_x_normalize:
    test_x_gtf.append(fil.process(x).real)

del train_x_raw, test_x_raw
# with open('train_x_filtered.txt', 'wb') as f:
#     pickle.dump(train_x_gtf, f)
#     f.close()
# with open('test_x_filtered.txt', 'wb') as f:
#     pickle.dump(test_x_gtf, f)
#     f.close()

# x = pickle.load(open('train_x_filtered.txt', 'rb'))

freq = []
impulse = np.zeros(8000)
impulse[0] = 1
res = fil.process(impulse).real
for i in range(16):
    fft = abs(np.fft.fft(res[i]))
    freq.append(np.argmax(fft[:4000]))


train_impulse_trains = []
test_impulse_trains = []

for i in train_x_gtf:
    # thetas = []
    # for (j, f) in zip(i, freq):
    #     #  thetas.append(1 / 50 / target_frequency[i] * m[i])
    #     thetas.append(0.02 / f * 4000)
    # m = 0.02 * np.max(i)
    train_impulse_trains.append(generate_impulse_train_multi_channels(i, 8000, 0.1, 0.02))

with open('train_x_impulses_normalize.txt', 'wb') as f:
    pickle.dump(train_impulse_trains, f)
    f.close()

# del train_impulse_trains
# del train_x_gtf

for i in test_x_gtf:
    # thetas = []
    # for (j, f) in zip(i, freq):
    #     #  thetas.append(1 / 50 / target_frequency[i] * m[i])
    #     thetas.append(0.02 / f * 4000)
    test_impulse_trains.append(generate_impulse_train_multi_channels(i, 8000, 0.01, 1000))

with open('test_x_impulses_normalize.txt', 'wb') as f:
    pickle.dump(test_impulse_trains, f)
    f.close()


