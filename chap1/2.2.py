import os
import pandas as pd


os.makedirs(os.path.join('..', 'tmp'), exist_ok=True)
data_file = os.path.join('..', 'tmp', 'house_tiny.csv')
if True or not os.path.exists(data_file):
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')  # 列名
        f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
        f.write("2,NA,106000\n")
        f.write("4,NA,178100\n")
        f.write("NA,NA,140000\n")


data = pd.read_csv(data_file)
print(data)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print('inputs:')
print(inputs)
print('outputs:')
print(outputs)

