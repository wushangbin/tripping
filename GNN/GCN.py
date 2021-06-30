# Title     : Graph Convolution Network
# Objective : 使用Cora数据集，通过一个完整的例子来理解GCN, 代码来自《深入浅出图神经网络》
# Created by: Wu Shangbin
# Created on: 2021/6/24
from collections import namedtuple
import os
import pickle
import urllib
import numpy as np
import itertools
import scipy.sparse as sp
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])
"""
我们使用的cora数据集是一个论文数据集。
Data: class-namedtuple
x: 1708*1433, 每篇论文通过词袋模型得到的特征
y: 论文的类别，共有7类
adjacency： 2708*2708 这些论文之间的引用关系，共5429条边
train_mask, val_mask, test_mask: 与结点数相同，用来标记节点是训练集，验证集或测试集
"""


class CoraData(object):
    download_url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    filenames = ["ind.cora.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="cora", rebuild=False):
        """
        从一个链接中下载数据集，并且把数据集构造成cora/raw/preocessed_cora.pkl的格式
        :param data_root: string, optional 
        :param rebuild: 是否需要重新构建数据集
        """
        self.data_root = data_root
        save_file = os.path.join(self.data_root, "processed_cora.pkl")
        if os.path.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            self.maybe_download()
            self._data = self.process_data()
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)
            print("Cached file: {}".format(save_file))

    @property
    def data(self):
        """
        :return: 数据对象，包括x,y,adjacency,train_mask,test_mask,val_mask
        """
        return self._data

    def maybe_download(self):
        save_path = os.path.join(self.data_root, "raw")
        for name in self.filenames:
            if not os.path.exists(os.path.join(save_path, name)):
                self.download_data("{}/{}".format(self.download_url, name), save_path)

    @staticmethod
    def download_data(url, save_path):
        """
        每次下载'x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index'中的一种数据并保存
        :param url: cora/raw
        :param save_path: https://github.com/kimiyoung/planetoid/raw/master/data/ + $NAME
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data = urllib.request.urlopen(url)
        filename = os.path.basename(url)
        with open(os.path.join(save_path, filename), 'wb') as f:
            f.write(data.read())
        return True

    @staticmethod
    def build_adjacency(adj_dict):
        """根据邻接表（字典）创建邻接矩阵,我们构造的邻接矩阵是n*n的格式，n为结点个数"""
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        # 删掉重复的边
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])),
                                  shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

    def process_data(self):
        print("Process data...")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(os.path.join(self.data_root, "raw", name))
                                                       for name in self.filenames]
        """
        _ (140, 1433) 01矩阵  没用上
        tx (1000, 1433) 01矩阵 
        allx (1708, 1433)
        y (140, 7) 没用上，后面我们对y重新赋值了
        ty (1000, 7)
        ally (1708, 7)
        graph: 一个长度为2708的字典，描述每个节点之间的边
        test_index 长度为1000列表，代表测试集的索引（均小于2708）
        """
        train_index = np.arange(y.shape[0])
        val_index = np.arange(y.shape[0], y.shape[0] + 500)
        sorted_test_index = sorted(test_index)

        x = np.concatenate((allx, tx), axis=0)
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)
        # 下面这个赋值感觉没什么用
        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        adjacency = self.build_adjacency(graph)
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())
        return Data(x=x, y=y, adjacency=adjacency, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def read_data(path):
        name = os.path.basename(path)
        if name == 'ind.cora.test.index':
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            # hasattr: 查看对象（out）是否有某个属性(toarray)
            out = out.toarray() if hasattr(out, "toarray") else out
            return out


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """
        :param input_dim:
        :param output_dim:
        :param use_bias:
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """
        :param adjacency:
        :param input_feature:
        :return:
        """
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output


class GcnNet(nn.Module):
    """
    一个包含两层GraphConvolution的模型
    """
    def __init__(self, input_dim=1433, hidden_dim=16, output_dim=7):
        """
        :param input_dim: 输入结点的矢量，1433维
        :param hidden_dim: 隐藏层节点个数
        :param output_dim: 输出维度，因为我们最终是把论文分为7类，所以是7
        """
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, hidden_dim,)
        self.gcn2 = GraphConvolution(hidden_dim, output_dim)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        return logits


def normalization(adjacency):
    """计算L=D^-0.5 * (A+I) * D^-0.5"""
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))  # 横向求和，算度数
    d_hat = sp.diags(np.power(degree, -0.5).flatten())  # 把 degree开-0.5次方，然后变成一个对角阵
    return d_hat.dot(adjacency).dot(d_hat).tocoo()


learning_rate = 0.1
weight_decay = 5e-4
epochs = 200
# 模型定义
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GcnNet().to(device)
# 损失函数使用交叉熵
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# -- 数据
dataset = CoraData().data
x = dataset.x / dataset.x.sum(1, keepdims=True)  # 求和,但维度不变:对所有结点的特征进行归一化
tensor_x = torch.from_numpy(x).to(device)
tensor_y = torch.from_numpy(dataset.y).to(device)
tensor_train_mask = torch.from_numpy(dataset.train_mask).to(device)
tensor_val_mask = torch.from_numpy(dataset.val_mask).to(device)
tensor_test_mask = torch.from_numpy(dataset.test_mask).to(device)
normalize_adjacency = normalization(dataset.adjacency)
indices = torch.from_numpy(np.asarray([normalize_adjacency.row, normalize_adjacency.col]).astype('int64')).long() # n*n Tensor
values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))
# 把normalize_adjacency构造成稀疏张量，其中indices是索引，values是每个索引的值
tensor_adjacency = torch.sparse.FloatTensor(indices, values, (2708, 2708)).to(device)


def train():
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(epochs):
        logits = model(tensor_adjacency, tensor_x)
        train_mask_logits = logits[tensor_train_mask]
        loss = criterion(train_mask_logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = test(tensor_train_mask)
        val_acc = test(tensor_val_mask)
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()))
    return loss_history, val_acc_history


def test(mask):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]  # 对每一行取最大值，并求最大值所在的索引
        accuracy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuracy


train()
