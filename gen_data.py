import numpy as np
from sklearn import datasets


dataset = datasets.load_boston()

x_train = np.insert(dataset.data[0:400, :-1], 0, values=np.ones(400), axis=1)
y_train = dataset.data[0:400, -1]
x_test = np.insert(dataset.data[400:, :-1], 0, values=np.ones(106), axis=1)
y_test = dataset.data[400:, -1]

x_train = dataset.data[0:400, :-1]
y_train = dataset.data[0:400, -1]
x_test = dataset.data[400:, :-1]
y_test = dataset.data[400:, -1]

np.savez("data/boston", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
# arr = np.load("data/boston.npz")


print("sdfs")

