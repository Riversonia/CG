"""
Training.py



"""

# Public import
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from createDataset import MyOwnDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Script in folder
from GCN4 import GCN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
delta = 0.01

# Main
if __name__ == "__main__":
    datas = MyOwnDataset("F:\Gitrepo\Python\CG\CG\dataset\data")
    DataLoader(datas, batch_size=1, shuffle=False)

    trainDataset = datas[0: int(len(datas) * 0.8)]
    testDataset = datas[int(len(datas) * 0.8): len(datas)]

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
            mseErr = mean_squared_error(testLabels, testOutputs)
            maeErr = mean_absolute_error(testLabels, testOutputs)
            R2 = r2_score(testLabels, testOutputs)


            print(f"MSE = {mseErr} \n"
                  f"MAE = {maeErr} \n"
                  f"R^2 = {R2} \n"
                  f"diff = |{maeErr / 2}|")

    # 损失率变化趋势画图
    model.plot_progress()
    plt.show()