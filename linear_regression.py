import numpy as np
import time
import matplotlib.pyplot as plt

# sklearn库的实现使用了最小二乘法
#from sklearn.linear_model import LinearRegression

'''
线性回归，使用梯度下降法
    learning_rate：学习率确实是个重要问题，不仅涉及到迭代效率，还会影响到模型loss收拢
    standard：如果不做标准化，会使得特征之间差距不同，导致梯度下降变慢（梯度较大的变量还容易导致loss发散）
使用了波斯顿房价的数据集，收敛后，结果并不太好，总归线性回归的模型泛化能力还是不足，太简单了
'''


class LinearReg(object):
    def __init__(self):
        self.fea_dim = 0
        self.train_num = 0
        self.weights = None
        self.bias = 0.0
        self.iterations = 10000
        self.rate = 0.001
        self.gradient = None
        self.loss_arr = []
        self.is_standard = False
        self.means = None
        self.stds = None

    def init(self, dim):
        self.weights = np.random.normal(size=dim).reshape(-1, 1)

    def mse(self, X, y):
        return np.linalg.norm(np.matmul(X, self.weights) - y, ord=2)

    def fit(self, X, y):
        y = y.reshape(X.shape[0], 1)
        self.train_num = X.shape[0]
        if self.is_standard:
            self.means = np.mean(X, axis=0)
            self.stds = np.std(X, axis=0)+0.0000000000001
            X = (X-self.means) / self.stds
        X = np.insert(X, 0, 1, axis=1)
        dim = X.shape[1]
        self.init(dim)
        self.loss_arr.append(self.mse(X, y))
        for i in range(self.iterations):
            y_pred = np.matmul(X, self.weights)
            w_grad = 2.0 / self.train_num * np.matmul(X.T, y_pred - y)
            self.weights -= self.rate * w_grad
            self.loss_arr.append(self.mse(X, y))

    def predict(self, X):
        if self.is_standard:
            X = (X-self.means) / self.stds
        X = np.insert(X, 0, 1, axis=1)
        return np.matmul(X, self.weights)


def main():
    arr = np.load("data/boston.npz")
    x_train, y_train = arr['x_train'], arr['y_train']
    model = LinearReg()
    model.is_standard = True
    model.iterations = 50
    model.rate = 0.1
    start_ts = time.time()
    model.fit(x_train, y_train)
    end_ts = time.time()
    with open('lr_loss', 'a+') as writer:
        writer.write("is_standard={}, iteration={}, rate={}, last_loss={}, time_cost={}\n".format(
            model.is_standard, model.iterations, model.rate, model.loss_arr[-1], end_ts-start_ts))
        x = np.arange(len(model.loss_arr))
        # plt.plot(x, model.loss_arr)

    x_test, y_test = arr['x_test'], arr['y_test']
    y_pred = model.predict(x_test)
    x = list(range(y_pred.shape[0]))
    plt.plot(x, y_test.tolist(), label='y_test')
    plt.plot(x, y_pred.tolist(), label='y_pred')
    plt.show()


main()