import requests
import re
import csv
import pandas as pd

# set read file path
#
# strPathRead = "D:\研0\组会\\20231220 煤矸数据裁剪处理完成 GNN模型确立\Coal gangue datset cropped and integrated\sliceFinished\sliceCM_1.csv"
# strPathSave = "D:\CoalGangueCode\ReadDataset\sliceCM_1.csv"

strPathRead = "D:\研0\组会\\20231220 煤矸数据裁剪处理完成 GNN模型确立\Coal gangue datset cropped and integrated\sliceFinished\sliceC_1.csv"
strPathSave = "D:\CoalGangueCode\ReadDataset\sliceC_1.csv"

def DatasetHeaderAdd(strPath, strSave, header):
    # add header to data.csv
    data = pd.read_csv(strPath, header=0, names=header)
    data.to_csv(strSave)
    print("Dataset Header add finished")

def CreateHeader(num_feature):
    # create header
    header = []
    header.append("label")
    header.append("channel")
    header.append("group")
    print(header)

    for i in range(0, num_feature):
        header.append(f"feature_{i + 1}")
    print(header)
    print(len(header))
    return header

def main():
    header = CreateHeader(1024)
    DatasetHeaderAdd(strPathRead, strPathSave, header)
    data = pd.read_csv(strPathSave, header=0, index_col=0)
    print(data)

main()




