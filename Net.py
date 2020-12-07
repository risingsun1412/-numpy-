import numpy as np
from functions import softmax, sigmoid, sigmoid_
from os.path import join

class Net:
    def __init__(self, n1 = 256, n2 = 64, n3 = 10, lr = 1e-3):
        self.lr = lr
        self.w1 = np.random.randn(n1, 28*28)
        self.b1 = np.random.randn(n1, 1)
        self.w2 = np.random.randn(n2, n1)
        self.b2 = np.random.randn(n2, 1)
        self.w3 = np.random.randn(n3, n2)
        self.b3 = np.random.randn(n3, 1)

        self.x = None
        self.y_pred = None

        self.a1 = None
        self.h1 = None
        self.a2 = None
        self.h2 = None
        self.a3 = None

        self.l_a3 = None

        self.l_w3 = None
        self.l_h2 = None
        self.l_a2 = None
        self.l_b3 = None

        self.l_w2 = None
        self.l_h1 = None
        self.l_a1 = None
        self.l_b2 = None

        self.l_w1 = None
        self.l_b1 = None

    def forward(self, x):
        self.x = x
        self.a1 = np.matmul(self.w1, x) + self.b1
        self.h1 = sigmoid(self.a1)

        self.a2 = np.matmul(self.w2, self.h1) + self.b2
        self.h2 = sigmoid(self.a2)

        self.a3 = np.matmul(self.w3, self.h2) + self.b3
        self.y_pred = softmax(self.a3)
        return self.y_pred

    def backward(self, y):
        self.l_a3 = self.y_pred - y
        self.l_w3 = np.matmul(self.l_a3, self.h2.T)
        self.l_h2 = np.matmul(self.w3.T, self.l_a3)
        self.l_a2 = self.l_h2 * sigmoid_(self.a2)
        self.l_b3 = self.l_a3
        self.l_w2 = np.matmul(self.l_a2, self.h1.T)
        self.l_h1 = np.matmul(self.w2.T, self.l_a2)
        self.l_a1 = self.l_h1 * sigmoid_(self.a1)
        self.l_b2 = self.l_a2
        self.l_w1 = np.matmul(self.l_a1, self.x.T)
        self.l_b1 = self.l_a1
        return -np.matmul(y.T, np.log(self.y_pred))

    def step(self):
        self.w3 = self.w3 - self.lr * self.l_w3
        self.b3 = self.b3 - self.lr * self.l_b3
        self.w2 = self.w2 - self.lr * self.l_w2
        self.b2 = self.b2 - self.lr * self.l_b2
        self.w1 = self.w1 - self.lr * self.l_w1
        self.b1 = self.b1 - self.lr * self.l_b1


    def save(self, path=''):
        np.savetxt(join(path, 'w3.csv'), self.w3, delimiter=',')
        np.savetxt(join(path, 'b3.csv'), self.b3, delimiter=',')
        np.savetxt(join(path, 'w2.csv'), self.w2, delimiter=',')
        np.savetxt(join(path, 'b2.csv'), self.b2, delimiter=',')
        np.savetxt(join(path, 'w1.csv'), self.w1, delimiter=',')
        np.savetxt(join(path, 'b1.csv'), self.b1, delimiter=',')
    def load(self, path=''):
        self.w3 = np.loadtxt(join(path, 'w3.csv'), delimiter=',', skiprows=0)
        self.b3 = np.loadtxt(join(path, 'b3.csv'), delimiter=',', skiprows=0)
        self.b3 = np.expand_dims(self.b3, axis=1)
        self.w2 = np.loadtxt(join(path, 'w2.csv'), delimiter=',', skiprows=0)
        self.b2 = np.loadtxt(join(path, 'b2.csv'), delimiter=',', skiprows=0)
        self.b2 = np.expand_dims(self.b2, axis=1)
        self.w1 = np.loadtxt(join(path, 'w1.csv'), delimiter=',', skiprows=0)
        self.b1 = np.loadtxt(join(path, 'b1.csv'), delimiter=',', skiprows=0)
        self.b1 = np.expand_dims(self.b1, axis=1)
if __name__ == '__main__':

    pass