import numpy as np
from enum import Enum, unique
from functools import reduce  # py3
from common.log import logger

'''
ID3
深度优先
离散特征
缺省值
分类问题
'''
'''
TODO:
剪枝
C4.5
cart树，分类、回归
二分类
类别平衡
连续特征
回归问题
正则项
更多的数据集测试数据

广度优先？
可扩展性？
'''

# 信息增益截止条件，小于该值则停止分裂
minimum_info_gain = 0.0001
# 如果Node节点的信息熵足够小了，就不分裂了（Node所有类别都是一个类别是，entropy最小，为0）
minimum_entropy = 0.00001


@unique
class SplitType(Enum):
    INFO_GAIN = 1
    INFO_GAIN_RATE = 2
    GINI = 3


class Node(object):
    def __init__(self, x_train, y_train, data_idx, fea_value=None, split_type=SplitType.INFO_GAIN):
        self.children = []
        self.children_fea_value = []
        self.x_train, self.y_train = x_train, y_train
        self.fea_num = self.x_train.shape[1]
        # 临时存储使用，表明属于当前节点的数据的索引数组
        self.data_idx = data_idx
        self.split_type = split_type
        # 当前节点的信息熵，表示拥有所有数据时候的信息熵
        stat = {}
        for idx in self.data_idx:
            value = self.y_train[idx]
            if value not in stat:
                stat[value] = 0
            stat[value] += 1
        self.train_num = len(data_idx)
        entropy_list = [-v / self.train_num * np.log(v / self.train_num) for k, v in stat.items()]
        self.entropy = reduce(lambda x, y: x + y, entropy_list)
        # 是否叶子节点
        self.is_leaf = True
        # 若该节点属于叶子节点，则该节点的分类为最大概率的那个类
        self.obj = reduce(lambda x, y : x if x[1] > y[1] else x, stat.items())[0]
        # Node的特征值，也就是上级在选择child的时候，根据child的fea_value选择
        # 和split_fea_idx区分开来，split_fea_idx是说在当前Node进行分裂的时候，使用哪个特征进行分类；
        #   而fea_value则是指Node作为搜索路径上，应该是哪个特征值
        self.fea_value = fea_value
        # 非叶子节点时，表明使用哪个特征进行分裂
        self.split_fea_idx = None

    @staticmethod
    def cal_entropy(stat):
        map(lambda p: p*np.log(p), stat.values())

    @staticmethod
    def create_root(x_train, y_train):
        return Node(x_train, y_train, list(range(x_train.shape[0])))

    def __gain(self, children):
        # 之所以叫y_fea_entropy，其实是在fea特征下，对于每个特征值下的类别情况的熵，再对所有特征值的熵求和
        y_fea_entropy = \
            reduce(lambda x, y: x + y, [child.entropy * child.train_num for child in children]) / self.train_num
        info_gain = self.entropy - y_fea_entropy
        # ID3, info_gain
        if self.split_type is SplitType.INFO_GAIN:
            return info_gain
        # C4.5, info_gain_rate
        fea_entropy_list = [-(c.train_num / self.train_num) * np.log((c.train_num / self.train_num)) for c in children]
        fea_entropy = reduce(lambda x, y: x + y, fea_entropy_list)
        info_gain_rate = info_gain / fea_entropy
        return info_gain_rate

    def build(self):
        if len(self.data_idx) < 2 or self.entropy < minimum_entropy:
            logger.info("not need to construct children node")
            return
        max_gain = 0
        max_gain_fea_idx = 0
        max_gain_children = None
        for i in range(self.fea_num):
            children = self.split(i)
            gain = self.__gain(children)
            if gain > max_gain:
                max_gain = gain
                max_gain_fea_idx = i
                max_gain_children = children
        if max_gain < minimum_info_gain:
            logger.info("信息熵增益太少，不进行分裂: gain=%f", max_gain)
            return
        logger.info("进行该节点的分裂: fea=%d, gain=%f, children_num=%d",
                    max_gain_fea_idx, max_gain, len(max_gain_children))
        self.is_leaf = False
        self.children = max_gain_children
        self.split_fea_idx = max_gain_fea_idx
        for child in self.children:
            child.build()

    # 按照fea_index生成子树
    def split(self, fea_idx):
        fea_data = {}
        for idx in self.data_idx:
            fea = self.x_train[idx][fea_idx]
            if fea not in fea_data:
                fea_data[fea] = []
            fea_data[fea].append(idx)
        return [Node(self.x_train, self.y_train, v, k, split_type=self.split_type) for k, v in fea_data.items()]

    def predict(self, one_sample):
        if self.is_leaf:
            return self.obj
        else:
            fea_value = one_sample[self.split_fea_idx]
            match_child = self.__find_child(fea_value)
            if match_child is None:
                logger.warn("没找到相应的子节点，用当前节点的值替代预测值")
                return self.obj
            return match_child.predict(one_sample)

    def __find_child(self, fea_value):
        for child in self.children:
            if fea_value == child.fea_value:
                return child
        logger.error("cannot find child for fea_value")
        return None


class DecisionTree(object):
    def __init__(self):
        # 根节点，是Node类型
        self.root = None
        self.split_type = SplitType.INFO_GAIN
        # 训练数据
        self.x_train, self.y_train = None, None
        self.train_num, self.fea_num = None, None

    def fit(self, x_train, y_train):
        self.x_train, self.y_train = x_train, y_train
        self.train_num, self.fea_num = x_train.shape
        self.root = Node.create_root(x_train, y_train)
        self.root.build()
        logger.info("完成决策树构建")

    def predict(self, x_test):
        y_predict = []
        for item in x_test:
            y_predict.append(self.root.predict(item))
        return y_predict


def main():
    arr = np.load("data/watermelon.npz")
    x_train, y_train = arr['x_train'], arr['y_train']
    model = DecisionTree()
    model.fit(x_train, y_train)

    x_test, y_test = arr['x_test'], arr['y_test']
    y_pred = model.predict(x_test)
    print(y_test)
    print(y_pred)


if __name__ == "__main__":
    main()


