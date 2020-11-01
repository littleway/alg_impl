import os
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


if not os.path.exists('data'):
    os.mkdir('data')
if not os.path.exists('log'):
    os.mkdir('log')


def download_boston():
    dataset = datasets.load_boston()
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=5)

    np.savez("data/boston", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    # arr = np.load("data/boston.npz")


def download_iris():
    dataset = datasets.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=5)

    np.savez("data/iris", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def watermelon():
    dataset = np.loadtxt("data/watermelon.data", dtype=int, delimiter=",")
    data = dataset[:, :-1]
    target = dataset[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=5)
    np.savez("data/watermelon", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


if __name__ == "__main__":
    # download_boston()
    # download_iris()
    watermelon()

