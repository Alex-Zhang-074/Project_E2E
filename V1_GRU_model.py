import numpy as np
import pandas as pd
import os
from multiprocessing import Pool
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# from data_api import DataFeeder, DataAPI
# from factor_base import FactorBase

# # Instantiate the data interface
# d = DataFeeder() 
# api = DataAPI()

def classify_y(y):
    if y >= 0.001:
        return 1
    else:
        return 0

def process_rolling_window(window):
    # print(window)
    if window.isnull().any().any():  # Check if any NaN in the window
        return None
    else:
        return ((window.iloc[:-1,:5].to_numpy(), np.array([[classify_y(window.iloc[-1,5])]])), str(window.index[-2])[:10])  # Convert DataFrame to numpy array

fac_path = "/mnt/research/data/temp/zhangsurui/E2E_NN/Round1/train/"
stock_pool = [file.split("_")[0] for file in os.listdir(
    fac_path) if file.endswith('_train.feather')]
window_size = 10
train_sample_list = []
train_index_list = []
test_sample_list = []
test_index_list = []
for stock in tqdm(stock_pool):
    data_train = pd.read_feather("/mnt/research/data/temp/zhangsurui/E2E_NN/Round1/train/"+stock+"_train.feather")
    data_train.set_index('date',inplace=True)
    train_list = [process_rolling_window(data_train.iloc[i:i+window_size+1]) for i in range(len(data_train) - window_size)]
    train_index = [(stock, res[1]) for res in train_list if res is not None]
    train_list = [res[0] for res in train_list if res is not None]
    train_sample_list.extend(train_list)
    train_index_list.extend(train_index)
    
    data_test = pd.read_feather("/mnt/research/data/temp/zhangsurui/E2E_NN/Round1/test/"+stock+"_test.feather")
    data_test.set_index('date',inplace=True)
    test_list = [process_rolling_window(data_test.iloc[i:i+window_size+1]) for i in range(len(data_test) - window_size)]
    test_index = [(stock, res[1]) for res in test_list if res is not None]
    test_list = [res[0] for res in test_list if res is not None]
    test_sample_list.extend(test_list)
    test_index_list.extend(test_index)
whole_index_list = train_index_list + test_index_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(train_sample_list, batch_size=1, shuffle=False)
test_loader = DataLoader(test_sample_list, batch_size=1, shuffle=False)

class GRUNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1, dropout_rate=0.4,random_seed=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        # self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        # out = self.batch_norm(out)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

model = GRUNet(input_size=5, hidden_size=50, num_layers=3, output_size=1, dropout_rate=0.4)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

num_epochs = 3
model.to(device)
train_losses = []
train_acc = []
test_losses = []
test_acc = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    train_correct = 0
    total_train = 0

    for seq, labels in train_loader:
        seq, labels = seq.to(device), labels.to(device)
        optimizer.zero_grad()
        y_pred = model(seq.float())
        single_loss = criterion(y_pred.squeeze(), labels.float().squeeze())
        single_loss.backward()
        optimizer.step()
        
        total_train_loss += single_loss.item()
        predicted = (y_pred > 0.5).float()
        total_train += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
    train_loss = total_train_loss / len(train_loader)
    train_accuracy = train_correct / total_train
    train_losses.append(train_loss)
    train_acc.append(train_accuracy)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')
    
    model.eval()
    total_test_loss = 0
    test_correct = 0
    total_test = 0

    for seq, labels in test_loader:
        seq, labels = seq.to(device), labels.to(device)
        y_pred = model(seq.float())
        single_loss = criterion(y_pred.squeeze(), labels.float().squeeze())
        
        total_test_loss += single_loss.item()
        predicted = (y_pred > 0.5).float()
        total_test += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
    test_loss = total_test_loss / len(test_loader)
    test_accuracy = test_correct / total_test
    test_losses.append(test_loss)
    test_acc.append(test_accuracy)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')

torch.save(model.state_dict(), '/mnt/research/data/temp/zhangsurui/E2E_NN/Round1/model_state_dict.pth')
print('Finished Training')

factor_name = "R1V2"
model.eval()
train_pred = []
for seq, labels in train_loader:
    seq, labels = seq.to(device), labels.to(device)
    y_pred = model(seq.float())
    y_pred = y_pred.to('cpu')
    train_pred.append(y_pred.squeeze().detach().numpy().tolist())
train_df = pd.DataFrame({"stock_code":[index[0] for index in train_index_list], "date":[index[1] for index in train_index_list], "factor_"+factor_name+"_train":train_pred})
train_df.to_feather("/mnt/research/data/temp/zhangsurui/E2E_NN/Round1/factor_"+factor_name+"_train.feather")
test_pred = []
for seq, labels in test_loader:
    seq, labels = seq.to(device), labels.to(device)
    y_pred = model(seq.float())
    y_pred = y_pred.to('cpu')
    test_pred.append(y_pred.squeeze().detach().numpy().tolist())
test_df = pd.DataFrame({"stock_code":[index[0] for index in test_index_list], "date":[index[1] for index in test_index_list], "factor_"+factor_name+"_test":test_pred})
test_df.to_feather("/mnt/research/data/temp/zhangsurui/E2E_NN/Round1/factor_"+factor_name+"_test.feather")
whole_df = pd.DataFrame({"stock_code":[index[0] for index in whole_index_list], "date":[index[1] for index in whole_index_list], "factor_"+factor_name:train_pred+test_pred})
whole_df.to_feather("/mnt/research/data/temp/zhangsurui/E2E_NN/Round1/factor_"+factor_name+".feather")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.title("Training and Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(test_acc, label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.legend()

plt.savefig("/mnt/research/data/temp/zhangsurui/E2E_NN/Round1/"+factor_name+".jpg")