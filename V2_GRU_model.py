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

def process_stock(stock_seq):
    stock = stack_index[stock_seq]
    if stock in df_rtn.index:
        stock_rtn = df_rtn.loc[stock, "return_5"]
        if ~np.isnan(stock_rtn):
            y = stock_rtn
            return y, stock_seq, stock
    return None, None, None

n_jobs = 200
window_size = 5
train_loader = []
train_index_list = []
test_loader = []
test_index_list = []

for date_seq in tqdm(range(window_size-1,len(date_list))):
    
    date_current = date_list[date_seq]
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

    stack_index = []
    stack_numpy = []
    for df in df_window.groupby("stock_code"):
        if len(df[1]) == 1190:
            stack_index.append(df[0])
            stack_numpy.append(df[1][["open", "high", "low", "close", "vwap", "volume"]].to_numpy())
    stack_numpy = np.array(stack_numpy)
    # ts_pre_process
    price_adj = np.tile(stack_numpy[:,0,0][:,np.newaxis], (1,stack_numpy.shape[1]))
    for price in range(5):
        stack_numpy[:,:,price] /= price_adj
    volume_adj = stack_numpy[:,:,5].mean(axis=1)
    volume_adj = np.tile(np.where(volume_adj < 1e-14, 1, volume_adj)[:,np.newaxis], (1,stack_numpy.shape[1]))
    stack_numpy[:,:,5] /= volume_adj
    # cs_pre_process
    for feat in range(6):
        cs_mean = np.tile(stack_numpy[:,:,feat].std(axis=0)[np.newaxis,:], (stack_numpy.shape[0],1))
        cs_std = np.tile(stack_numpy[:,:,feat].std(axis=0)[np.newaxis,:], (stack_numpy.shape[0],1))
        stack_numpy[:,:,feat] = np.where(cs_std < 1e-14, np.zeros_like(stack_numpy[:,:,feat]), (stack_numpy[:,:,feat] - cs_mean) / cs_std)

    pool = Pool(n_jobs)
    results = pool.map(process_stock, range(len(stack_index)))
    pool.close()
    pool.join()
    batch_list = []
    batch_y = []
    batch_code = []
    for result in results:
        y, index, code = result
        if y is not None:
            batch_y.append(y)
            batch_list.append(index)
            batch_code.append((date_current, code))
    batch_X = torch.Tensor(stack_numpy[batch_list])
    batch_y = torch.Tensor(np.array(batch_y)[:,np.newaxis])

    if data_type == "train":
        train_loader.append((batch_X, batch_y))
        train_index_list.append(batch_code)
    else:
        test_loader.append((batch_X, batch_y))
        test_index_list.append(batch_code)
whole_index_list = train_index_list + test_index_list

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
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
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

    for seq, labels in train_loader:
        seq, labels = seq.to(device), labels.to(device)
        optimizer.zero_grad()
        y_pred = model(seq.float())
        loss = criterion(y_pred.float(), labels.float())
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()

    train_loss = total_train_loss / len(train_loader)
    train_losses.append(train_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}')
    
    model.eval()
    total_test_loss = 0

    for seq, labels in test_loader:
        seq, labels = seq.to(device), labels.to(device)
        y_pred = model(seq.float())
        loss = criterion(y_pred.float(), labels.float())
        total_train_loss += loss.item()

    train_loss = total_train_loss / len(train_loader)
    train_losses.append(train_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}')

torch.save(model.state_dict(), '/mnt/research/data/temp/zhangsurui/E2E_NN/Round2/model_state_dict.pth')
print('Finished Training')

factor_name = "R2V1"
model.eval()
train_pred = []
for seq, labels in train_loader:
    seq, labels = seq.to(device), labels.to(device)
    y_pred = model(seq.float())
    y_pred = y_pred.to('cpu')
    train_pred.append(y_pred.detach().numpy().reshape(-1).tolist())
train_df = pd.DataFrame({"stock_code":[index[0] for index in train_index_list], "date":[index[1] for index in train_index_list], "factor_"+factor_name+"_train":train_pred})
train_df.to_feather("/mnt/research/data/temp/zhangsurui/E2E_NN/Round2/factor_"+factor_name+"_train.feather")
test_pred = []
for seq, labels in test_loader:
    seq, labels = seq.to(device), labels.to(device)
    y_pred = model(seq.float())
    y_pred = y_pred.to('cpu')
    test_pred.append(y_pred.detach().numpy().reshape(-1).tolist())
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

plt.savefig("/mnt/research/data/temp/zhangsurui/E2E_NN/Round2/"+factor_name+".jpg")