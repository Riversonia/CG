import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from createDataset import MyOwnDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        if (self.counter % 1000 == 0):
            print(f"counter = {self.counter}, loss = {loss.item()}")

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def test(self, data):
        outputs = self.forward(data.x, data.edge_index, data.batch)
        outputs = outputs.permute(1, 0)

        y = data.y.to(device)
        delta = torch.abs(y - outputs)

        # acc代表误差率，越小越好
        # print(f"testdata_y = {data.y}, \n "
        #       f"output = {outputs}, \n "
        #       f"delta = {delta}")
        sum = delta.sum().item()
        return sum / len(data)

    def plot_progress(self):
        plt.plot(range(len(self.progress)), self.progress)


def main():
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
        deltaRate = model.test(data)
        print(deltaRate)

    # 损失率变化趋势画图
    model.plot_progress()
    plt.show()

main()


