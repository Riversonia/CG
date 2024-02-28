import requests
import re
import csv
import pandas as pd

# set read file path
#
# strPathRead = "D:\研0\组会\\20231220 煤矸数据裁剪处理完成 GNN模型确立\Coal gangue datset cropped and integrated\sliceFinished\sliceCM_1.csv"
# strPathSave = "D:\CoalGangueCode\ReadDataset\sliceCM_1.csv"

# strPathRead = "D:\研0\组会\\20231220 煤矸数据裁剪处理完成 GNN模型确立\Coal gangue datset cropped and integrated\sliceFinished\sliceC_1.csv"
# strPathSave = "D:\研0\组会\\20231220 煤矸数据裁剪处理完成 GNN模型确立\Coal gangue datset cropped and integrated\sliceFinished\labDatas(2400x5)groups\sliceC_1.csv"

def CombineCSV():
    strReadPath = "D:\研0\组会\\20231220 煤矸数据裁剪处理完成 GNN模型确立\Coal gangue datset cropped and integrated\sliceFinished\\8\\"
    strPathSave = "D:\研0\组会\\20231220 煤矸数据裁剪处理完成 GNN模型确立\Coal gangue datset cropped and integrated\sliceFinished\labDatas(2400x5)groups\sliceC_8.csv"
    f1 = pd.read_csv(strReadPath + "sliceC000_8.csv", header=None)
    f2 = pd.read_csv(strReadPath + "sliceC025_8.csv", header=None)
    f3 = pd.read_csv(strReadPath + "sliceC050_8.csv", header=None)
    f4 = pd.read_csv(strReadPath + "sliceC075_8.csv", header=None)
    f5 = pd.read_csv(strReadPath + "sliceC100_8.csv", header=None)
    f = pd.concat([f1, f2, f3, f4, f5])
    print(f)
    f.to_csv(strPathSave)
    return f

def DatasetHeaderAdd(strPath, strSave, header):
    # add header to data.csv
    data = pd.read_csv(strPath, header=0, names=header)
    print(data)
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

if __name__ == "__main__":
    CombineCSV()
    header = CreateHeader(1024)
    strPathRead = "D:\研0\组会\\20231220 煤矸数据裁剪处理完成 GNN模型确立\Coal gangue datset cropped and integrated\sliceFinished\labDatas(2400x5)groups\sliceC_8.csv"
    strPathSave = "D:\研0\组会\\20231220 煤矸数据裁剪处理完成 GNN模型确立\Coal gangue datset cropped and integrated\sliceFinished\labDatas(2400x5)groups\sliceCR_8.csv"
    DatasetHeaderAdd(strPathRead, strPathSave, header)
    data = pd.read_csv(strPathRead, header=0, index_col=0)
    print(data)





