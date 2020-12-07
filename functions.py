import numpy as np
from math import exp

def sigmoid(x):
    res = np.zeros(x.shape)
    for i in range(len(x)):
        if x[i] >= 0:
            res[i] = 1.0 / (1 + exp(-x[i]))
        else:
            res[i] = exp(x[i]) / (1 + exp(x[i]))
    return res

def sigmoid_(x):
    # sigmoid的导数
    sx = sigmoid(x)
    return sx * (1 - sx)

def softmax(x):
    maxv = np.max(x)
    ex = np.exp(x - maxv)
    return ex / np.sum(ex)

def to_onehot(num):
    res = np.zeros((10, 1))
    res[num] = 1
    return res

def get_item(x):
    return x.flatten()[0]