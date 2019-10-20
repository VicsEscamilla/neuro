import gzip
import numpy as np
from matplotlib import pyplot as plt

import json

def show_img(img):
    plt.imshow(img.reshape(28,28))
    plt.show()


def load_mnist():
    ''' load mnist data, return training data and test data '''
    def _load():
        import pickle
        with gzip.open('mnist.pkl.gz', 'rb') as f:
            return (pickle.load(f, encoding='latin1'))

    def _to_vector(x):
        e = np.zeros((10, 1))
        e[x] = 1.0
        return e

    train, _, test = _load()

    train_X = np.array([np.reshape(x, (784, 1)) for x in train[0]], dtype=np.float32)
    train_y = np.array([_to_vector(y) for y in train[1]], dtype=np.float32)

    test_X = np.array([np.reshape(x, (784, 1)) for x in test[0]], dtype=np.float32)
    test_y = np.array([_to_vector(y) for y in test[1]], dtype=np.float32)

    return train_X, train_y, test_X, test_y



print("START")
train_X, train_y, test_X, test_y = load_mnist()
np.save("mnist_train_X.npy", train_X.flatten())
np.save("mnist_train_y.npy", train_y.flatten())
np.save("mnist_test_X.npy", test_X.flatten())
np.save("mnist_test_y.npy", test_y.flatten())
