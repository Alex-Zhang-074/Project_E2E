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

from data_api import DataFeeder, DataAPI
from factor_base import FactorBase

# Instantiate the data interface
d = DataFeeder() 
api = DataAPI()

data = pd.read_feather("/mnt/data/stocks/basic/processed_stock_trade_data.feather")
data = data[data.date >= "2016-01-01"][['date', 'stock_code', 'open', 'high', 'low', 'close', 'volume', 'total_turnover']]
data['vwap'] = data.total_turnover / data.volume
date_list = data.date.unique().tolist()
def process_stock_group(group):
    group = group[['date', 'vwap']].copy()
    group.set_index('date', inplace=True)
    group = group.reindex(date_list)
    group["return_5"] = (group.vwap.shift(-4) / group.vwap - 1).shift(-1)
    return group.iloc[:-5,:]
df_return = data.groupby('stock_code').apply(process_stock_group).reset_index()
df_return.rename(columns={'level_1': 'date'}, inplace=True)
df_return = df_return[['date', 'stock_code', 'return_5']]
# df_return.to_feather("/mnt/research/data/temp/zhangsurui/E2E_NN/Round2/return_data.feather")
print("Loading return data finished.")

def ts_pre_process(group):
    price_adj = group.open.tolist()[0]
    group.open = group.open / price_adj
    group.high = group.high / price_adj
    group.low = group.low / price_adj
    group.close = group.close / price_adj
    group.vwap = group.vwap / price_adj

    volume_adj = group.volume.mean()
    if volume_adj < 1e-14:
        volume_adj = 1
    group.volume = group.volume / volume_adj

    return group

def cs_pre_process(group):
    if group.open.std() < 1e-14:
        group.open = 0
    else:
        group.open = (group.open - group.open.mean()) / group.open.std()
    group.high = (group.high - group.high.mean()) / group.high.std()
    group.low = (group.low - group.low.mean()) / group.low.std()
    group.close = (group.close - group.close.mean()) / group.close.std()
    group.vwap = (group.vwap - group.vwap.mean()) / group.vwap.std()

    if group.volume.std() < 1e-14:
        group.volume = 0
    else:
        group.volume = (group.volume - group.volume.mean()) / group.volume.std()

    return group

def process_stock(stock):
    if stock in df_rtn.index:
        stock_rtn = df_rtn.loc[stock, "return_5"]
        if ~np.isnan(stock_rtn):
            x = torch.Tensor([df_window.loc[df_window.stock_code == stock, ["open", "high", "low", "close", "vwap", "volume"]].to_numpy()])
            y = stock_rtn
            index = (date_current, stock)
            return x, y, index
    return None, None, None

n_jobs = 150
window_size = 5
train_sample_list = []
train_index_list = []
test_sample_list = []
test_index_list = []

for date_seq in tqdm(range(window_size-1,len(date_list))):
    
    date_current = date_list[date_seq]
    print("Loading data at "+date_current.strftime("%Y-%m-%d"))
    df_rtn = df_return.loc[df_return.date == date_current, ["stock_code", "return_5"]].set_index("stock_code")
    data_type = "train" if date_current.strftime("%Y-%m-%d") < "2022-01-01" else "test"
    
    date_window = date_list[date_seq-window_size+1:date_seq+1]
    window_stack = []
    for date in date_window:
        data_mink = pd.read_feather("/mnt/data/stocks/mink/by_date/"+date.strftime("%Y%m%d")+".feather")
        data_mink["stock_code"] = data_mink.order_book_id.map(lambda x:api.code_transf(rqcode=x,verse=False))
        data_mink["vwap"] = data_mink.total_turnover / data_mink.volume
        data_mink = data_mink[["stock_code", "datetime", "date", "time", "open", "high", "low", "close", "vwap", "volume"]]
        data_mink = data_mink[(data_mink.time != "14:58:00") & (data_mink.time != "14:59:00")]
        data_mink.loc[data_mink.volume == 0, "vwap"] = data_mink.loc[data_mink.volume == 0, "open"]
        window_stack.append(data_mink)
    df_window = pd.concat(window_stack, axis=0)
    if df_window.isna().any().any():
        print("The window " + date_current.strftime("%Y%m%d") + " contains unexpected NaN")
        continue
    df_window = df_window.groupby("stock_code").apply(ts_pre_process).reset_index(drop=True)
    df_window = df_window.groupby("datetime").apply(cs_pre_process).reset_index(drop=True)
    # df_window.to_feather("/mnt/research/data/temp/zhangsurui/E2E_NN/Round2/"+data_type+"/"+date_current.strftime("%Y%m%d")+"_"+data_type+".feather")
    
    pool = Pool(n_jobs)
    results = pool.map(process_stock, df_window.stock_code.unique())
    pool.close()
    pool.join()
    batch_X, batch_y, batch_index = [], [], []
    for result in results:
        x, y, index = result
        if x is not None:
            batch_X.append(x)
            batch_y.append(y)
            batch_index.append(index)

    if data_type == "train":
        train_sample_list.append((batch_X, batch_y))
        train_index_list.append(batch_index)
    else:
        test_sample_list.append((batch_X, batch_y))
        test_index_list.append(batch_index)

print("Data loading finished.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PearsonCorrelationLoss(nn.Module):
    def forward(self, y_pred, y_true):
        y_pred_mean = torch.mean(y_pred)
        y_true_mean = torch.mean(y_true)
        y_pred_std = torch.std(y_pred)
        y_true_std = torch.std(y_true)

        cov = torch.mean((y_pred - y_pred_mean) * (y_true - y_true_mean))
        IC = cov / (y_pred_std * y_true_std)

        return -IC

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
        return out

model = GRUNet(input_size=6, hidden_size=50, num_layers=2, output_size=1, dropout_rate=0.4)
criterion = PearsonCorrelationLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

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
