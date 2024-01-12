import dgl
import pandas as pd


strPathRead = "D:\CoalGangueCode\ReadDataset\sliceCM_1.csv"
strPathSave = "D:\CoalGangueCode\SaveDataset\sliceCM_1.dataset"

def main():
    data = pd.read_csv(strPathRead, header=0, index_col=0)
    print(data)

main()
