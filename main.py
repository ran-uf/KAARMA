from orgKAARMA import KAARMA
import numpy as np
import pickle


def extract_frame(t, t_s, t_e):
    ls = []
    for tt in t:
        if t_s < tt < t_e:
            ls.append(tt)
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


def toonehot(data, n):
    m = np.zeros((data.size, n))
    for i in range(data.size):
        m[i, int(data[i])] = 1
    return m


t = 0.025
fps = 100

model = KAARMA(20, 10, 0.05, 0.05, t)

test_y = np.load('test_y.npy')
train_y = np.load('train_y.npy')
train_x = pickle.load(open('train_x_impulses_simple.txt', 'rb'))
test_x = pickle.load(open('test_x_impulses_simple.txt', 'rb'))


train_x_frame = []
test_x_frame = []

for x in train_x:
    train_x_frame.append(split_frame(x, t, 1 / fps))
for x in test_x:
    test_x_frame.append(split_frame(x, t, 1 / fps))


num_data = np.size(train_y)
index = np.random.permutation(num_data)
train_y_shuffle = np.zeros(num_data)
train_y_shuffle[index] = train_y

train_y = toonehot(train_y_shuffle, 10)
test_oh_y = toonehot(test_y, 10)

train_x = []
for i in index:
    train_x.append(train_x_frame[i])

with open('train_x_frame.txt', 'wb') as f:
    pickle.dump(train_x_frame, f)
    f.close()

with open('test_x_frame.txt', 'wb') as f:
    pickle.dump(test_x_frame, f)
    f.close()


# x = [[[np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])], [np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])]]]
# y = [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])]
print('start training')
for i in range(1000):
    model.train(train_x, train_y, 1, 0.1)
    loss = model.test(test_x_frame, test_oh_y, test_y)

