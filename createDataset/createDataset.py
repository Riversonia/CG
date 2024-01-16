import random

import dgl
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

strPathRead = "D:\CG\CoalGangueCode\ReadDataset\sliceC_1.csv"
strPathSave = "D:\CG\CoalGangueCode\SaveDataset\sliceC_1.dataset"

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def CreateDataset(readFile, CCfactor):   # 将文件readFile构建成为数据集

    # 读取指定文件file 第一行不做列名header=None (不添加header=None会默认第一行为行名称数据)
    df = pd.read_csv(readFile, header=0, index_col=0)

    # 初始化一个空数据集，接受生成的数据
    graph = []
    # 每行的数据格式为 content channel groupNumber feature*1024 (index from 0 ~ 1026)
    # 利用循环从文件中每行的第3个数据开始
 #  # 这里记得要改range(0, len(df))
    for row in range(0, len(df)):

        # 特征矩阵建立为x 8*128 8个节点，每个节点有128维的特征向量
        x = torch.zeros(8, 128, dtype=torch.float)

        for num_nodes in range(0, 8):
            x[num_nodes, :] = torch.tensor(df.values[row, (num_nodes*127 + 3 + num_nodes): ((num_nodes+1)*127 + 3 + num_nodes + 1)], dtype=torch.float)

        # 建立标签列y content含量列
        y = torch.tensor(df.values[row, 1], dtype=torch.float)


        edge_index = [[], []]
        edge_attr = []
        for i in range(0, 8):
            for j in range(i+1, 8):
                v1, v2 = x[:, i], x[:, j]

                # pearson correlation coefficient matrix [[C(v1, v1) C(v1, v2)], [C(v2, v1) C(v2, v2)]]
                corr = np.corrcoef(v1, v2)

                # 皮尔逊相关系数，np.corrcoef 生成了一个v1和v2的相关系数矩阵，形状如上，对角相等，取C(v1, v2)或C(V2, V1)
                pCorrCoef = corr[0, 1]
                # print(corr)

                # 如果某两个节点的特征向量v1, v2的相关系数大于相关系数CCfactor
                # 则会在两个节点建立边索引，并将相关系数pCorrCoef的值作为边权重赋值给edge_attr
                # 为了保证是无向图，在边索引建立时需要建立双向边[i, j]和[j, i]
                if pCorrCoef >= CCfactor:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
                    edge_index[1].append(i)
                    edge_index[0].append(j)
                    edge_attr.append(pCorrCoef)
                    edge_attr.append(pCorrCoef)

        # 将计算后的edge_index, edge_attr变量转换为tensor量
        edge_index = torch.tensor(edge_index, dtype=torch.float)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # 将计算后的x, y, edge_index, edge_attr整理成为Data数据集，并附在graph后
        graph.append(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr))
        """
        如果edge_index 的格式为起始点与终点的索引对数组
            形如tensor([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [4, 5], [4, 6], [4, 7], [5, 6], [5, 7], [6, 7]])
        需要通过edge_index.t().contiguous()方法将索引对转化为边序列索引
            形如tensor([[0, 0, 1, 1, 2, 4, 4, 4, 5, 5, 6],
                        [1, 2, 2, 3, 3, 5, 6, 7, 6, 7, 7]])
                        
        或者在添加边索引时，先声明一个二维空数组
            edge_index = [[], []]
            在后面的方法中添加两条边的索引值，后面不需要用到转化的方法
            edge_index[0].append(sourceIndexValue)
            edge_index[1].append(targetIndexValue）
        """

        # 进度显示
        if (row + 1) % 100 == 0:
            print(readFile, row + 1, "/12000 finished")

    # 最后函数将返回整理好的graph数据集
    # len(graph) = 2400
    # graph[i] = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

    # 随机打乱数据集的索引，并返回整理好的数据集
    # random.shuffle(graph)
    return graph

    # print(len(edge_attr))
    # print(graph[0])
    # print(len(graph))
    # print(edge_index)
    # print(edge_attr)
    # print(graph)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(128, 64)
        self.conv2 = GCNConv(64, 16)
        self.out = torch.nn.Linear(16, 1)

        # 创建损失函数
        self.loss_function = torch.nn.MSELoss()

        # 创建优化器
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=5e-4)

        # 训练次数计数
        self.counter = 0

        # 训练过程损失记录
        self.progress = []

        # 创建池化层
        # self.global_mean_pool = global_mean_pool(x=x, batch=batch)

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x=x, batch=batch)
        out = self.out(x)

        return out

    def train_(self, data):
        outputs = self.forward(data.x, data.edge_index, data.batch)

        # 计算损失
        y = data.y.to(device)
        loss = self.loss_function(outputs, y)
        # print(loss) # debug

        # 训练次数+1
        self.counter += 1

        # 每10次训练记录损失
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())

        # 每1000次训练输出训练次数
        if (self.counter % 1000 == 0):
            print(f"counter = {self.counter}, loss = {loss.item()}")

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def test(self, data):
        outputs = self.forward(data.x, data.edge_index, data.batch)
        y = data.y
        # acc = sum(torch.abs(y - outputs) <= 0.05) / len(data.y)
        acc = torch.abs(y - outputs) # / y *100
        for num in range (0, len(acc)):
            sum = acc[num, 1]
            # print(acc[num, 1])

        print(len(acc))
        return sum / len(acc) * 100

    def plot_progress(self):
        plt.plot(range(len(self.progress)),self.progress)


def main():
    data = pd.read_csv(strPathRead, header=0, index_col=0)
    # print(data.groupby('label'))
    # print(data)
    # print(len(data))
    # print(data.values[0, 0: 4])

    # 这里记得启动制作数据集
    dataset = CreateDataset(strPathRead, 0.4)
    # print(dataset[0].x)
    # print(dataset[0].edge_index)
    # print(dataset[0].edge_attr)
    # print(dataset[0].y)
    # print(dataset[0])

    # print(dataset[0].validate())
    # print(f'graph_is_undirected: {dataset[0].is_undirected()}')
    print(f'graph_keys: {dataset[0].keys()}')
    print(f'graph_num_nodes: {dataset[0].num_nodes}')
    print(f'graph_num_edges: {dataset[0].num_edges}')
    print(f'graph_num_nodes_features: {dataset[0].num_node_features}')
    print(f'graph_num_edges_features: {dataset[0].num_edge_features}')

    trainDataset = dataset[ : 9600]
    # print(trainDataset[0: 5])
    # random.shuffle(trainDataset)
    # print(trainDataset[0: 5])

    testDataset = dataset[9600: ]
    # print(testDataset[0: 5])
    # random.shuffle(testDataset)
    # print(testDataset[0: 5])


    trainLoader = DataLoader(trainDataset, batch_size=16, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=len(testDataset))

    # 开始训练
    print(device)
    model = GCN().to(device)
    for i in range(0, len(trainDataset)):
        trainDataset[i].to(device)
    for i in range(0, len(testDataset)):
        testDataset[i].to(device)

    for i in range(10):
        for data in trainLoader:
            # try:  # debug
            data.edge_index = torch.tensor(data.edge_index, dtype=torch.int64)
            model.train_(data)
            # print(data[0])
            # except Exception as e: # debug
            # print(e)  # debug

    for data in testLoader:
        data.edge_index = torch.tensor(data.edge_index, dtype=torch.int64)
        acc = model.test(data)
        print(acc)

    # 损失率变化趋势画图
    model.plot_progress()
    plt.show()

main()

