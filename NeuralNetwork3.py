import csv
import sys
from math import sqrt

import numpy as np

try:
    train_image_path, train_label_path, test_image_path = sys.argv[1:4]
except:
    train_image_path, train_label_path, test_image_path = 'train_image.csv', 'train_label.csv', 'test_image.csv'
drop_probability = 0


def Relu(x, prime=False):
    if prime:
        return 1. * (x > 0)
    return x * (x > 0)


def Sigmoid(x, prime=False):
    if prime:
        return Sigmoid(x) * (1. - Sigmoid(x))
    return 1. / (1. + np.exp(-x))


def Tanh(x, prime=False):
    if prime:
        return 1. - x ** 2
    return np.tanh(x)


def Softmax(x):
    z = x - x.max()
    shape = x.shape
    z = np.atleast_2d(z)
    expz = np.exp(z)
    return (expz / expz.sum(axis=1, keepdims=True)).reshape(shape)


def dropout(X):
    keep_probability = 1 - drop_probability
    mask = np.random.random(X.shape) < keep_probability
    scale = (1 / keep_probability)
    return mask * X * scale


class NeuralNetwork:
    def __init__(self, layers, activations):
        self.Ws = []
        for i in range(1, len(layers) - 1):
            self.Ws.append(np.random.normal(loc=0, scale=sqrt(2 / (layers[0] + layers[-1])),
                                            size=(layers[i - 1] + 1, layers[i] + 1)))
        self.Ws.append(
            np.random.normal(loc=0, scale=sqrt(2 / (layers[0] + layers[-1])), size=(layers[-2] + 1, layers[-1])))
        self.activations = activations

    def fit(self, X: np.ndarray, y, alpha, n_epoch):
        global drop_probability
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        X = (X - X.mean()) / X.std()
        Y = np.eye(10)[y]
        s = 0
        for epoch in range(n_epoch):
            r = np.random.randint(X.shape[0], size=1)
            a = [dropout(X[r])]
            for i in range(len(self.Ws) - 1):
                a.append(dropout(self.activations[i](np.dot(a[i], self.Ws[i]))))
            a.append(self.activations[-1](np.dot(a[-1], self.Ws[-1])))
            delta = Y[r] - a[-1]
            s += np.sum(np.argmax(a[-1]) == np.argmax(Y[r]))
            deltas = [delta]
            for i in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.Ws[i].T) * self.activations[i - 1](a[i], prime=True))
            deltas.reverse()
            for i in range(len(self.Ws)):
                self.Ws[i] += alpha * np.dot(np.atleast_2d(a[i]).T, np.atleast_2d(deltas[i]))
            if epoch % 1000 == 0:
                acc = s / 1000
                print(acc)
                if acc > 0.90:
                    drop_probability = 0.2
                    alpha = 0.002
                s = 0

    def predict(self, X):
        a = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        a = (a - a.mean()) / a.std()
        for w, activation in zip(self.Ws, self.activations):
            a = activation(a.dot(w))
        return a.argmax(axis=1).astype('int8')


if __name__ == '__main__':
    with open(train_image_path) as train_image, open(train_label_path) as train_label, open(
            test_image_path) as test_image:
        train_X = np.asarray(list(csv.reader(train_image)), 'f')
        train_y = np.asarray(list(csv.reader(train_label)), 'int8').flatten()
        test_X = np.asarray(list(csv.reader(test_image)), 'f')
        # sample_i = np.random.choice(train_X.shape[0], size=10000, replace=False)
        # train_X = train_X[sample_i]
        # train_y = train_y[sample_i]

        nn = NeuralNetwork([784, 128, 256, 64, 32, 10], [Relu] * 4 + [Softmax])
        nn.fit(train_X, train_y, 0.001, 150000)
        pred_y = nn.predict(test_X)
        csv.writer(open('test_predictions.csv', 'w')).writerows(pred_y.reshape((-1, 1)))
        try:
            from test import precision

            precision()
        except:
            pass
