# Title     : Multi Layer Perceptron
# Objective : Implementation of a multi-layer perceptron using numpy
# Created by: Wu Shangbin
# Created on: 2021/11/6
import numpy as np


class MLPnet:

    def __init__(self, x, y, lr=0.003):
        """
        :param x: data
        :param y: labels
        :param lr: learning rate
        yh: predicted labels
        """
        self.X = x
        self.Y = y
        self.yh = np.zeros((1, self.Y.shape[1]))
        self.lr = lr
        self.dims = [13, 20, 1]  # 不同层的结点个数

        self.param = {}  # 需要训练的参数（w, b）
        self.ch = {}  # 将一些结果存在这里，以便在反向传播时使用
        self.loss = []  # 存放每个epoch的loss
        self.batch_size = 64

    def nInit(self):
        """对神经网络中的参数进行随机初始化"""
        np.random.seed(1)
        self.param['theta1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0])
        self.param['b1'] = np.zeros((self.dims[1], 1))
        self.param['theta2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1])
        self.param['b2'] = np.zeros((self.dims[2], 1))

    def Relu(self, u):
        return np.maximum(0, u)

    def Tanh(self, u):
        return (np.exp(u) - np.exp(-u)) / (np.exp(u) + np.exp(-u))

    def forward(self, x):
        if self.param == {}:
            self.nInit()
        u1 = np.matmul(self.param['theta1'], x) + self.param['b1']
        o1 = self.Tanh(u1)
        u2 = np.matmul(self.param['theta2'], o1) + self.param['b2']
        o2 = self.Relu(u2)

        self.ch['X'] = x
        self.ch['u1'], self.ch['o1'] = u1, o1
        self.ch['u2'], self.ch['o2'] = u2, o2
        return o2

    def nloss(self,y, yh):
        """
        :param y: 1*n, 列表
        :param yh: 1*N, 列表
        :return: 1*1 标量
        """
        n = y.shape[1]
        error = []
        squaredError = []
        for i in range(n):
            error.append(y[0][i] - yh[0][i])
        for val in error:
            squaredError.append(val * val)
        result = sum(squaredError) / (2 * n)
        return result

    def dRelu(self, u):
        """
        :param u: u of any dimension
        :return: dRelu(u) """
        u[u<=0] = 0
        u[u>0] = 1
        return u

    def dTanh(self, u):
        """
        :param u: u of any dimension
        :return: dTanh(u)
        """
        o = np.tanh(u)
        return 1-o**2

    def backward(self, y, yh):
        n = y.shape[1]
        dLoss_o2 = (yh - y) / n
        dLoss_u2 = dLoss_o2 * self.dRelu(self.ch['o2'])  # (1,379)
        dLoss_theta2 = np.matmul(dLoss_u2, self.ch['o1'].T) / n
        dLoss_b2 = np.sum(dLoss_u2) / n

        dLoss_o1 = np.matmul(self.param["theta2"].T, dLoss_u2)  # (20*1) mul (1*379)
        dLoss_u1 = dLoss_o1 * self.dTanh(self.ch['u1'])  # (20*379)
        dLoss_theta1 = np.matmul(dLoss_u1, self.X.T)  # (20*379) mul (379*13)
        dLoss_b1 = np.sum(dLoss_u1, axis=1, keepdims=True) / n
        # parameters update:
        self.param["theta2"] = self.param["theta2"] - self.lr * dLoss_theta2
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2
        self.param["theta1"] = self.param["theta1"] - self.lr * dLoss_theta1
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
        return dLoss_theta2, dLoss_b2, dLoss_theta1, dLoss_b1

    def predict(self, x):
        yh = self.forward(x)
        return yh

    def gradient_descent(self, x, y, iter=60000):
        """
        每次跑全部数据，跑iter次，每2000次存储一次loss
        :param x: data
        :param y: labels
        :param iter: 迭代次数
        """
        for i in range(iter):
            pre_y = self.predict(x)
            this_loss = self.nloss(y, pre_y)
            self.loss.append(this_loss)
            if i % 2000 == 0:
                print("Loss after iteration", i, ":", this_loss)
            self.backward(y, pre_y)

    def batch_gradient_descent(self, x, y, iter=60000):
        """
        这里的迭代次数，依然是backward的次数，并非epoch（epoch是看全部数据的次数）
        """
        n = y.shape[1]
        begin = 0
        for k in range(iter):
            index_list = [i % n for i in range(begin, begin + self.batch_size)]
            x_batch = x[:, index_list]
            y_batch = y[:, index_list]
            pre_y = self.predict(x_batch)
            self.X = x_batch
            this_loss = self.nloss(y_batch, pre_y)
            if k % 1000 == 0:
                self.loss.append(this_loss)
                print("Loss after iteration", k, ":", this_loss)
            self.backward(y_batch, pre_y)
            begin = begin + self.batch_size


from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# 梯度下降：
'''
'''
if __name__ == '__main__':
    dataset = load_boston()  # load the dataset
    x, y = dataset.data, dataset.target
    y = y.reshape(-1, 1)

    x = MinMaxScaler().fit_transform(x)  # normalize data
    y = MinMaxScaler().fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)  # split data
    x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.reshape(1, -1), y_test  # condition data

    nn = MLPnet(x_train, y_train, lr=0.001)  # initalize neural net class
    nn.batch_gradient_descent(x_train, y_train, iter = 60000) #train

    # create figure
    fig = plt.plot(np.array(nn.loss).squeeze())
    plt.title(f'Training: MLPnet')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    y_predicted = nn.predict(x_test)  # predict
    y_test = y_test.reshape(1, -1)
    print("Mean Squared Error (MSE)", (np.sum((y_predicted - y_test) ** 2) / y_test.shape[1]))
