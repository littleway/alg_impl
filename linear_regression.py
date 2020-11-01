import numpy as np
import time
import matplotlib.pyplot as plt

# sklearn库的实现使用了最小二乘法
#from sklearn.linear_model import LinearRegression

'''
线性回归，使用梯度下降法
    learning_rate：学习率确实是个重要问题，不仅涉及到迭代效率，还会影响到模型loss收拢
    standard：如果不做标准化，会使得特征之间差距不同，导致梯度下降变慢（梯度较大的变量还容易导致loss发散）
    regularization: 正则化
使用了波斯顿房价的数据集，收敛后，结果并不太好，总归线性回归的模型泛化能力还是不足，太简单了
TODO: 
    用kernel解决非线性问题
'''

'''
实验结果
只展示数据结果，感兴趣的可以把图打出来更直观
TODO: !!!下面的数据都是错的，因为当初目标结果使用错了。。。但是思路还是可以借鉴的，后面有空再更新
1. learning_rate的飘走
学习率需要很低，才能不飘走，飘走是二次优化问题的通病，需要小心
学习率低了，导致训练迭代太多，耗时太久
is_standard=False, regular_type=0, l2_rate=0.0001, iteration=100, rate=1e-05, train_loss=1.1563479704661123e+69, test_loss=7.513108362618583e+69, time_cost=0.007646799087524414
is_standard=False, regular_type=0, l2_rate=0.0001, iteration=100, rate=1e-06, train_loss=1305.1279963000202, test_loss=4130.407657941945, time_cost=0.007445812225341797
is_standard=False, regular_type=0, l2_rate=0.0001, iteration=100, rate=1e-07, train_loss=1409.2123184351706, test_loss=3707.289829132579, time_cost=0.007420063018798828
is_standard=False, regular_type=0, l2_rate=0.0001, iteration=10000, rate=1e-06, train_loss=149.55140390859813, test_loss=1551.8673054378435, time_cost=0.5667829513549805
is_standard=False, regular_type=0, l2_rate=0.0001, iteration=100000, rate=1e-06, train_loss=103.40184627569423, test_loss=630.2777906050305, time_cost=5.792619943618774
is_standard=False, regular_type=0, l2_rate=0.0001, iteration=1000000, rate=1e-06, train_loss=87.95984060912868, test_loss=696.251102317927, time_cost=56.70756411552429

2. 标准化的结果
主要关注iterator或者训练时间，顺带关注train_loss
标准化后，可以采用较大的学习率，使得训练速度加快
is_standard=True, regular_type=0, l2_rate=0.0001, iteration=100, rate=0.1, train_loss=80.78850314671664, test_loss=738.2944303290418, time_cost=0.007305145263671875
is_standard=True, regular_type=0, l2_rate=0.0001, iteration=50, rate=0.1, train_loss=80.83438482427387, test_loss=726.8206803402682, time_cost=0.004673004150390625
is_standard=True, regular_type=0, l2_rate=0.0001, iteration=20, rate=0.1, train_loss=81.6420332347977, test_loss=687.7957173412783, time_cost=0.0025920867919921875
is_standard=True, regular_type=0, l2_rate=0.0001, iteration=10, rate=0.1, train_loss=86.2113274600426, test_loss=779.0287319283707, time_cost=0.001878976821899414

3. 正则化的结果
主要关注test_loss
is_standard=True, regular_type=0, l2_rate=0.01, iteration=1000, rate=0.01, train_loss=80.78792864766321, test_loss=737.166531574196, time_cost=0.06296229362487793
is_standard=True, regular_type=2, l2_rate=0.1, iteration=1000, rate=0.01, train_loss=80.80716469290293, test_loss=735.4706291854548, time_cost=0.052866220474243164
is_standard=True, regular_type=2, l2_rate=0.01, iteration=1000, rate=0.01, train_loss=80.7874310904788, test_loss=737.5423105231204, time_cost=0.05828094482421875
is_standard=True, regular_type=2, l2_rate=0.001, iteration=1000, rate=0.01, train_loss=80.79370424422028, test_loss=734.7827263681078, time_cost=0.053295135498046875
is_standard=True, regular_type=2, l2_rate=0.0001, iteration=1000, rate=0.01, train_loss=80.79503015923372, test_loss=733.9937985918065, time_cost=0.0640420913696289
'''


class LinearReg(object):
    def __init__(self):
        self.fea_dim = 0
        # 这里没有bias，作为weight的第一维，所以会对输入样本新增第一维，值为1
        self.weights = None
        self.iterations = 100
        self.rate = 0.001
        self.loss_record = []
        # 是否标准化，均值、方差记录
        self.is_standard = False
        self.means = None
        self.stds = None
        # 正则化类型，0-none, 1-L1，2-L2，3-elastic net. 暂时支持l2
        self.regular_type = 0
        self.l2_rate = 0.01

    def init(self):
        self.weights = np.random.normal(size=self.fea_dim).reshape(-1, 1)

    # 私有函数，X已经预处理过，包括添加bias维，标准化等
    def _mse(self, X, y):
        return np.linalg.norm(np.matmul(X, self.weights) - y, ord=2)

    def mse(self, X, y):
        if self.is_standard:
            X = (X-self.means) / self.stds
        X = np.insert(X, 0, 1, axis=1)
        return np.linalg.norm(np.matmul(X, self.weights) - y, ord=2)

    def regular_grad(self):
        if self.regular_type == 0:
            return 0
        elif self.regular_type == 1:
            return 0
        elif self.regular_type == 2:
            return self.l2_rate / self.fea_dim * self.weights

    def fit(self, X, y):
        y = y.reshape(X.shape[0], 1)
        train_num = X.shape[0]
        if self.is_standard:
            self.means = np.mean(X, axis=0)
            self.stds = np.std(X, axis=0)+0.0000000000001
            X = (X-self.means) / self.stds
        X = np.insert(X, 0, 1, axis=1)
        self.fea_dim = X.shape[1]
        self.init()
        self.loss_record.append(self._mse(X, y))
        for i in range(self.iterations):
            y_pred = np.matmul(X, self.weights)
            w_grad = 2.0 / train_num * np.matmul(X.T, y_pred - y) + self.regular_grad()
            self.weights -= self.rate * w_grad
            self.loss_record.append(self._mse(X, y))

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
    model.iterations = 20
    model.rate = 0.1
    model.regular_type = 0
    model.l2_rate = 0.0001
    start_ts = time.time()
    model.fit(x_train, y_train)
    end_ts = time.time()

    # train的预测图片
    # y_train_pred = model.predict(x_train)
    # x = list(range(y_train_pred.shape[0]))
    # plt.plot(x, y_train.tolist(), label='y_train')
    # plt.plot(x, y_train_pred.tolist(), label='y_train_pred')
    # plt.show()

    x_test, y_test = arr['x_test'], arr['y_test']
    y_pred = model.predict(x_test)
    x = list(range(y_pred.shape[0]))
    plt.plot(x, y_test.tolist(), label='y_test')
    plt.plot(x, y_pred.tolist(), label='y_pred')
    plt.show()
    loss = model.mse(x_test, y_test)

    with open('log/lr_loss', 'a+') as writer:
        writer.write("is_standard={}, regular_type={}, l2_rate={}, iteration={}, rate={},"
                     " train_loss={}, test_loss={}, time_cost={}\n".format(
            model.is_standard, model.regular_type, model.l2_rate, model.iterations, model.rate,
            model.loss_record[-1], loss, end_ts - start_ts))
        x = np.arange(len(model.loss_record))
        # plt.plot(x, model.loss_arr)


if __name__ == "__main__":
    main()

