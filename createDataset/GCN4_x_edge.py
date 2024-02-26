import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from createDataset import MyOwnDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

delta = 0.01
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(128, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.conv4 = GCNConv(16, 4)
        self.out = torch.nn.Linear(4, 1)

        # 创建损失函数
        self.loss_function = torch.nn.MSELoss()

        # 创建优化器
        self.optimiser = torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=5e-4)

        # 训练次数计数
        self.counter = 0

        # 训练过程损失记录
        self.progress = []

        # 创建池化层
        # self.global_mean_pool = global_mean_pool(x=x, batch=batch)

    def forward(self, x, edge_index, batch):

        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
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
        if (self.counter % 500 == 0):
            print(f"counter = {self.counter}, loss = {loss.item()}")

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def test(self, data):
        labels_, outputs_ = [], []
        outputs = self.forward(data.x, data.edge_index, data.batch)
        outputs = outputs.permute(1, 0)
        outputs = outputs.squeeze(0)
        y = data.y.to(device)

        outputs = outputs.detach().cpu().numpy()
        labels = data.y.detach().cpu().numpy()

        for index in range (0, len(labels)):
            if labels[index] == 0:
                labels_.append(delta)
            else:
                labels_.append(labels[index])

        for index in range (0, len(outputs)):
            outputs_.append(outputs[index])

        labels_ = np.array(labels_)
        outputs_ = np.array(outputs_)

        return labels_, outputs_

    def plot_progress(self):
        plt.plot(range(len(self.progress)), self.progress)

def mse(y, y_hat):
    diff2 = []
    for index in range(0, len(y)):
        diff2.append( (y[index] - y_hat[index]) ** 2 )
    mse = np.sum(diff2) / len(y)
    return mse

def mae(y, y_hat):
    diff = []
    for index in range(0, len(y)):
        diff.append( abs(y[index] - y_hat[index]) )
    mae = np.sum(diff) / len(y)
    return mae

if __name__ == "__main__":
    datas = MyOwnDataset("F:\Gitrepo\Python\CG\CG\createDataset\data")
    DataLoader(datas, batch_size=1, shuffle=False)

    trainDataset = datas[0: int(len(datas)* 0.8)]
    testDataset = datas[int(len(datas)* 0.8): len(datas)]

    trainLoader = DataLoader(trainDataset, batch_size=128, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=len(testDataset))

    # 开始训练
    print(device)
    model = GCN().to(device)

    for i in range(500):
        for data in trainLoader:
            # try:  # debug
            data.edge_index = torch.tensor(data.edge_index, dtype=torch.int64)
            model.train_(data)
            # print(data[0])
            # except Exception as e: # debug
            # print(e)  # debug

    for data in testLoader:
        data.edge_index = torch.tensor(data.edge_index, dtype=torch.int64)
        testLabels, testOutputs = model.test(data)
        mseErr = mse(testLabels, testOutputs)
        maeErr = mae(testLabels, testOutputs)

        print(f"MSE = {mseErr} \n"
              f"MAE = {maeErr} \n"
              f"diff = |{maeErr/2}|")
    # 损失率变化趋势画图
    model.plot_progress()
    plt.show()



