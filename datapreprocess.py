import numpy as np
from gtfblib import Chen
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pickle
from IFModel import generate_impulse_train_multi_channels


train_x_raw = np.load('train_x_raw.npy', allow_pickle=True)
train_y_raw = np.load('train_y_raw.npy')
test_x_raw = np.load('test_x_raw.npy', allow_pickle=True)
test_y_raw = np.load('test_y_raw.npy')

fil = Chen(fs=8000)

train_x_gtf = []
test_x_gtf = []
for x in train_x_raw:
    train_x_gtf.append(fil.process(x).real)
for x in test_x_raw:
    test_x_gtf.append(fil.process(x).real)

with open('train_x_filtered.txt', 'wb') as f:
    pickle.dump(train_x_gtf, f)
    f.close()
with open('test_x_filtered.txt', 'wb') as f:
    pickle.dump(test_x_gtf, f)
    f.close()

# x = pickle.load(open('train_x_filtered.txt', 'rb'))

train_impulse_trains = []
test_impulse_trains = []

for i in train_x_gtf:
    thetas = []
    for j in i:
        #  thetas.append(1 / 50 / target_frequency[i] * m[i])
        thetas.append(0.002 * np.max(abs(j)))
    train_impulse_trains.append(generate_impulse_train_multi_channels(i, 8000, 1, thetas))

for i in test_x_gtf:
    thetas = []
    for j in i:
        #  thetas.append(1 / 50 / target_frequency[i] * m[i])
        thetas.append(0.002 * np.max(abs(j)))
    test_impulse_trains.append(generate_impulse_train_multi_channels(i, 8000, 1, thetas))


