import os
import pandas as pd
import glob

def CombineCSV():

    strReadPath = "D:\研0\组会\\20231220 煤矸数据裁剪处理完成 GNN模型确立\Coal gangue datset cropped and integrated\sliceFinished\\1\\"
    f1 = pd.read_csv(strReadPath + "sliceC000_1.csv", header=None)
    f2 = pd.read_csv(strReadPath + "sliceC025_1.csv", header=None)
    f3 = pd.read_csv(strReadPath + "sliceC050_1.csv", header=None)
    f4 = pd.read_csv(strReadPath + "sliceC075_1.csv", header=None)
    f5 = pd.read_csv(strReadPath + "sliceC100_1.csv", header=None)
    f = pd.concat([f1, f2, f3, f4, f5])

    print(f)
