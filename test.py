import numpy as np
from sklearn.metrics import accuracy_score


def precision():
    pred_y = np.loadtxt('test_predictions.csv', dtype='int8')
    true_y = np.loadtxt('test_label.csv', dtype='int8')
    print(accuracy_score(true_y, pred_y))


if __name__ == '__main__':
    precision()
