from KAARMA import KAARMA
import numpy as np
import pickle
from kernels import gaussian_spikes


def toonehot(data, n):
    m = np.zeros((data.size, n))
    for i in range(data.size):
        m[i, int(data[i])] = 1
    return m


t = 0.025
fps = 100


test_y = np.load('./data/small/test_y_small.npy')
test_x = np.load('./data/small/test_x_final.npy', allow_pickle=True)
train_y = np.load('./data/small/train_y_small.npy')
train_x = np.load('./data/small/train_x_final.npy', allow_pickle=True)

num_data = np.size(train_y)
index = np.random.permutation(num_data)

train_x_shuffle = []
train_y_shuffle = []
for i in index:
    train_x_shuffle.append(train_x[i])
    train_y_shuffle.append(train_y[i])

# target_y = train_y_shuffle
train_y_oh = toonehot(np.array(train_y_shuffle), 10)
test_y_oh = toonehot(test_y, 10)

# with open('train_x_frame.txt', 'wb') as f:
#     pickle.dump(train_x_frame, f)
#     f.close()
#
# with open('test_x_frame.txt', 'wb') as f:
#     pickle.dump(test_x_frame, f)
#     f.close()


# x = [[[np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])], [np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])]]]
# y = [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])]
# print(train_x_shuffle[0][0].shape)
model = KAARMA(15, 10, 2, 4, train_x_shuffle[0][0], gaussian_spikes)
print('start training')
for i in range(100):
    # model.train(train_x_shuffle[:10], train_y_oh[:10], train_y_shuffle[:1], test_x, test_y_oh, test_y, 1000, 0.9, 0.01)
    model.train(train_x_shuffle[:50], train_y_oh[:50], 0.1, 0.01)

