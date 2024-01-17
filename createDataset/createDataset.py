import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import warnings

import pandas as pd
import numpy as np
import random


warnings.filterwarnings("ignore", category=Warning)
strReadFile = "F:\Gitrepo\Python\CG\CG\createDataset\data\\raw\sliceCM_1.csv"

from torch_geometric.datasets import TUDataset
# 这里给出大家注释方便理解
# 程序只要第一次运行后，processed文件生成后就不会执行proces函数，而且只要不重写download()和process()方法，也会直接跳过下载和处理。
class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # print(self.data) # 输出torch.load加载的数据集data
        # print(root) # MYdata
        # print(self.data) # Data(x=[3, 1], edge_index=[2, 4], y=[3])
        # print(self.slices) # defaultdict(<class 'dict'>, {'x': tensor([0, 3, 6]), 'edge_index': tensor([ 0,  4, 10]), 'y': tensor([0, 3, 6])})
        # print(self.processed_paths[0]) # MYdata\processed\datas.pt

    # 返回数据集源文件名，告诉原始的数据集存放在哪个文件夹下面，如果数据集已经存放进去了，那么就会直接从raw文件夹中读取。
    @property
    def raw_file_names(self):
        # pass # 不能使用pass，会报join() argument must be str or bytes, not 'NoneType'错误
        return ["sliceCM_1.csv"]

    # 首先寻找processed_paths[0]路径下的文件名也就是之前process方法保存的文件名
    @property
    def processed_file_names(self):
        return ["sliceCM_1_seed115"]

    # 用于从网上下载数据集，下载原始数据到指定的文件夹下，自己的数据集可以跳过
    def download(self):
        pass

    # 生成数据集所用的方法，程序第一次运行才执行并生成processed文件夹的处理过后数据的文件，否则必须删除已经生成的processed文件夹中的所有文件才会重新执行此函数
    def process(self):

        # 读取指定文件file 第一行不做列名header=None (不添加header=None会默认第一行为行名称数据)
        df = pd.read_csv(strReadFile, header=0, index_col=0)

        # 这里用于构建data

        # 初始化一个空数据集，接受生成的数据
        graph = []
        # 每行的数据格式为 content channel groupNumber feature*1024 (index from 0 ~ 1026)
        # 利用循环从文件中每行的第3个数据开始
        # 这里记得要改range(0, len(df))
        for row in range(0, len(df)):

            # 特征矩阵建立为x 8*128 8个节点，每个节点有128维的特征向量
            x = torch.zeros(8, 128, dtype=torch.float)

            for num_nodes in range(0, 8):
                x[num_nodes, :] = torch.tensor(
                    df.values[row, (num_nodes * 127 + 3 + num_nodes): ((num_nodes + 1) * 127 + 3 + num_nodes + 1)],
                    dtype=torch.float)

            # 建立标签列y content含量列
            y = torch.tensor(df.values[row, 0], dtype=torch.float)

            edge_index = [[], []]
            edge_attr = []
            for i in range(0, 8):
                for j in range(i + 1, 8):
                    v1, v2 = x[:, i], x[:, j]

                    # pearson correlation coefficient matrix [[C(v1, v1) C(v1, v2)], [C(v2, v1) C(v2, v2)]]
                    corr = np.corrcoef(v1, v2)

                    # 皮尔逊相关系数，np.corrcoef 生成了一个v1和v2的相关系数矩阵，形状如上，对角相等，取C(v1, v2)或C(V2, V1)
                    pCorrCoef = corr[0, 1]
                    # print(corr)

                    # 如果某两个节点的特征向量v1, v2的相关系数大于相关系数CCfactor
                    # 则会在两个节点建立边索引，并将相关系数pCorrCoef的值作为边权重赋值给edge_attr
                    # 为了保证是无向图，在边索引建立时需要建立双向边[i, j]和[j, i]
                    if pCorrCoef >= 0.4: # CCfactor = 0.4
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
                print(strReadFile, row + 1, f"/ {len(df)} finished")

            # 随机打乱数据集的索引，并返回整理好的数据集
        random.seed(115)
        random.shuffle(graph)

        if self.pre_filter is not None:  # pre_filter函数可以在保存之前手动过滤掉数据对象。用例可能涉及数据对象属于特定类的限制。默认None
            graph = [data for data in graph if self.pre_filter(data)]

        if self.pre_transform is not None:  # pre_transform函数在将数据对象保存到磁盘之前应用转换(因此它最好用于只需执行一次的大量预计算)，默认None
            graph = [self.pre_transform(data) for data in graph]

        data, slices = self.collate(graph) # 直接保存list可能很慢，所以使用collate函数转换成大的torch_geometric.data.Data对象
        # print(data)
        torch.save((data, slices), self.processed_paths[0])

# 数据集对象操作
b = MyOwnDataset("F:\Gitrepo\Python\CG\CG\createDataset\data") # 创建数据集对象
data_loader = DataLoader(b, batch_size=1, shuffle=False) # 加载数据进行处理，每批次数据的数量为1
for data in data_loader:
    print(data) # 按批次输出数据