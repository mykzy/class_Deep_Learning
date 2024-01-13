"""
 $ @Author: jx like kzy
 $ @Date: 2024-01-11 18:00:33
 $ @LastEditTime: 2024-01-13 19:30:31
 $ @FilePath: \实训\main.py
 $ @Description: 实训的项目 手写数字识别
 $ @
 """
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import torch
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
# print("你好，kaggle2")
data_train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
data_test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
# print(data_train.shape)
print(data_test.shape)
# print(data_train.head)

X_test_data = data_test# 移除label标签
# X_test_label=data_test["label"]

X_train_data = data_train.drop("label" ,axis=1)# 移除label标签
X_train_label=data_train["label"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch  

X_train_tensor = torch.tensor(X_train_data.values, dtype=torch.float32)  
y_train_tensor = torch.tensor(X_train_label.values, dtype=torch.long)
from torch.utils.data import Dataset, DataLoader  
# print(X_train_tensor)
# print(y_train_tensor)
# print(X_train_tensor.shape)
# print(y_train_tensor.shape)
class CustomDataset(Dataset):  
    def __init__(self, X, y):  
        self.X = X  
        self.y = y  
 
    def __len__(self):  
        return len(self.X)  
 
    def __getitem__(self, idx):  
        return self.X[idx], self.y[idx]  

class CustomDatasetOne(Dataset):  
    def __init__(self, X):  
        self.X = X  
 
    def __len__(self):  
        return len(self.X)  
 
    def __getitem__(self, idx):  
        return self.X[idx]  

train_dataset = CustomDataset(X_train_tensor,y_train_tensor)  
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)



X_test_tensor = torch.tensor(X_test_data.values, dtype=torch.float32)  


test_dataset = CustomDatasetOne(X_test_tensor)  
test_loader = DataLoader(test_dataset, batch_size=64)

import torch.nn as nn
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
learning_rate = 0.0001
num_epochs = 30
sequence_length=28
end_data=[]
losses=[]
modle=RNN(input_size, hidden_size, num_layers, num_classes)
print(modle)
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modle.parameters(), lr=learning_rate)
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        outputs = modle(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item()}')
    modle.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images in test_loader:
            images = images.reshape(-1, sequence_length, input_size)#.to(device)
            # labels = labels#.to(device)
            outputs = modle(images)
            # print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted.data)
            end_data.append(predicted.data)          
import csv
with open('/kaggle/working/sample_submission.csv',mode='w',newline='',encoding='utf8') as cf:
    wf=csv.writer(cf)
    wf.writerow(["ImageId","Label"])
    for i in range(len(end_data)):
        for j in range(len(end_data[i])):
            data=[(i)*64+j+1,int(end_data[i][j])]
            wf.writerow(data)
import matplotlib.pyplot as plt
plt.plot(range(len(losses)),losses)
plt.show()
