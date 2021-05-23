import numpy as np


def __calculate_next_point(x0, y0, k, b, target):
    if k == 0:
        return target / y0 + x0
    else:
        if target > 0:
            if k > 0:
                return (- b + np.sqrt(b ** 2 + 2 * target * k + k * x0 * (y0 + b))) / k
            elif k < 0:
                return (- b + np.sqrt(b ** 2 + 2 * target * k + k * x0 * (y0 + b))) / k
        elif target < 0:
            if k > 0:
                return (- b - np.sqrt(b ** 2 + 2 * target * k + k * x0 * (y0 + b))) / k
            elif k < 0:
                return (- b - np.sqrt(b ** 2 + 2 * target * k + k * x0 * (y0 + b))) / k


def trains_from_interval(x0, y0, x1, y1, alpha, theta, temp):
    trains = []
    x_last = x0
    y_last = y0
    k = (y1 - y0) / (x1 - x0)
    b = y0 - k * x0
    temp = temp * np.exp(alpha * (x0 - x1))
    # print(abs((y1 + y_last) * (x1 - x_last) / 2 + temp))
    while abs((y1 + y_last) * (x1 - x_last) / 2 + temp) > theta:
        # print(y1, y_last)
        if y1 + y_last >= 0:
            x = __calculate_next_point(x_last, y_last, k, b, theta - temp)
            if x < x_last or x > x1:
                print('x0', x0, 'x1', x1, 'y0', y0, 'y1', y1, 'target', theta - temp, 'x', x)
            x_last = x
            trains.append([x_last, 1])
            y_last = k * x_last + b
            temp = 0
        else:
            x = __calculate_next_point(x_last, y_last, k, b, -theta - temp)
            if x < x_last or x > x1:
                print('x0', x0, 'x1', x1, 'y0', y0, 'y1', y1, 'target', -theta - temp)
            x_last = x
            trains.append([x_last, -1])
            y_last = k * x_last + b
            temp = 0
    temp = (y1 + y_last) * (x1 - x_last) / 2 + temp
    return trains, temp


def generate_impulse_train(data, fs, alpha=1, theta=0.05):
    impulse_train = []
    temp = 0
    a = alpha / fs
    for i in range(data.size - 1):
        # if i % 100 == 0:
        #     print("\r ",  i / data.size, end=" ", flush=True)
        # temp_data = data[temp_t:i + 1]
        # leaks = np.exp(a * np.arange(temp_t - i, 1, 1))
        # temp = temp_data @ leaks / fs
        # print(temp, i + 1 - temp_t)
        # if abs(temp) >= theta:
        #     temp_t = i + 1
        #     if temp >= theta:
        #         impulse_train[i] = 1
        #     elif temp <= -theta:
        #         impulse_train[i] = -1

        x0 = i / fs
        x1 = (i + 1) / fs
        y0 = data[i]
        y1 = data[i + 1]

        if y0 * y1 < 0:
            x_middle = (x0 * y1 - y0 * x1) / (y1 - y0)
            temp_trains, temp = trains_from_interval(x0, y0, x_middle, 0, alpha, theta, temp)
            impulse_train.extend(temp_trains)
            temp_trains, temp = trains_from_interval(x_middle, 0, x1, y1, alpha, theta, temp)
            impulse_train.extend(temp_trains)

        else:
            temp_trains, temp = trains_from_interval(x0, y0, x1, y1, alpha, theta, temp)
            impulse_train.extend(temp_trains)
    return impulse_train


def generate_impulse_train_multi_channels(data, fs, alpha, thetas):
    impulse_trains = []
    for (i, theta) in zip(data, thetas):
        impulse_trains.append(generate_impulse_train(i, fs, alpha, theta))
    return impulse_trains
