# This code is  LSTM model by torch

import numpy as np
import pandas as pd

from helper import *
from dataprocess import DataProcessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

__author__ = 'Wenter'

__all__ = ['LSTMDataset',
          'LSTM_Model_for_sintosin',
          'LSTM_Model_for_seqtosin',
          'LSTM_Process']


class LSTMDataset(Dataset):
    """
    :class: dataset
    """

    # every day length
    PAD = 4794

    def __init__(self,
                 data: pd.DataFrame,
                 pad: int = PAD,
                 time_interval: int = 0):
        """
        :param data: whole data set
        :param pad: the data length for train, if < will be padded
        :param time_interval: how much time features for a response
        """

        self.data = data
        self.date_index = self.data.index.get_level_values(0).unique()
        self.pad = pad
        self.time_interval = time_interval

    def __len__(self):
        """
        :return: the length of data (day)
        """
        return len(self.date_index)

    def __getitem__(self, idx):
        """
        :func: data to tensor gpu and padding the data
        """
        idx = self.date_index[idx]
        data = torch.tensor(self.data.loc[idx].values, dtype=torch.float, device=device)
        if data.shape[0] < self.pad:
            data = F.pad(data, (0, 0, self.pad - data.shape[0], 0), "constant", 0)

        features = data[:, :-1]

        if self.time_interval:
            response = data[self.time_interval - 1:, -1:]
        else:
            response = data[:, -1:]

        return features, response


class LSTM_Model_for_sintosin(nn.Module):
    """
    :class:LSTM MODEL for single time index to single time index
    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, drop_out):
        """
        :param input_dim: feature dim
        :param hidden_dim: hidden dim of linear dense of lstm model
        :param layer_dim: layer of lstm dense vertical
        :param output_dim: output dim
        :param drop_out: drop out dense
        """

        super(LSTM_Model_for_sintosin, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=drop_out)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        """
        :func:initialize the fc for train
        """
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.fc.bias, std=1e-6)

    def forward(self, x):
        """
        :param x: input
        h: output
        c: hidden output sum of update gate and forget gate
        :return: all the output h[1]
        """
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(F.sigmoid(out))
        return out


class LSTM_Model_for_seqtosin(nn.Module):
    """
    :class:LSTM MODEL for sequence time index to single time index
    """
    def __init__(self, input_dim:int, hidden_dim:int, layer_dim, output_dim, drop_out, time_interval):
        """
        init
        :param input_dim: feature dim
        :param hidden_dim: hidden dim of linear dense of lstm model
        :param layer_dim: layer of lstm dense vertical
        :param output_dim: output dim
        :param drop_out: drop out dense
        :param time_interval: how much time index for one time index
        """
        super(LSTM_Model_for_seqtosin, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.time_interval = time_interval

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=drop_out)
        self.fc1 = nn.Linear(hidden_dim, 1)
        self.fc2 = nn.Linear(time_interval, output_dim)
        self._init_weights()

    def _init_weights(self):
        """
        :func:initialize the fc for train
        """
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.fc.bias, std=1e-6)

    def forward(self, x):
        """
        :param x: input
        h: output
        c: hidden output sum of update gate and forget gate
        Using unfold to get much time interval
        :return: all the output h[1]
        """
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc1(F.sigmoid(out))
        out = out.unfold(1, self.time_interval, 1)
        out = self.fc2(F.sigmoid(out))
        out = out.view(out.shape[0], -1, 1)

        return out


class LSTM_Process:
    """
    :class:LSTM train test process
    CONVERGE:converge rate
    *can optimize to apply all nn model
    """
    DROP_OUT: int = 0.2
    BATCH_SIZE: int = 10
    EPOCHS: int = 100
    ITER: int = 32
    LR: int = 0.01
    CONVERGE: int = 0.001
    SAVE: list = []

    def __init__(self,):
        """
        using MSE to train
        """
        self.model = None
        self.criterion = nn.MSELoss()
        self.optimizer = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.time_interval = 0

    @staticmethod
    def flog(epoch, content):
        print(f"------epoch{epoch}----{content}------")

    def initialize(self,
                   new: bool = True,
                   all_dates: list = ALL_DATES,
                   input_dim: int = 1,
                   hidden_dim: int = 20,
                   layer_dim: int = 1,
                   output_dim: int = 1,
                   time_interval: int = 0

    ):
        """
        :func: get the data and initialize the model
               can choose new or load and time interval
        :param new: whether creat new or load
        :param input_dim: feature dim
        :param hidden_dim: hidden dim of linear dense of lstm model
        :param layer_dim: layer of lstm dense vertical
        :param output_dim: output dim
        :param time_interval: how much time index for one time index
        :optimizer: adam
        """

        '''
        for new we get data
        dp = DataProcessing(all_dates,validate=0.2,train=0.6
        train,val,test = dp.data_for_nn()
        save_gzip(train, './ML/lstm/lstm_train_data.pkl')
        save_gzip(val,'./ML/lstm/lstm_val_data.pkl')
        save_gzip(test,'./ML/lstm/lstm_test_data.pkl')

        '''
        train = load_gzip('./ML/lstm/lstm_train_data.pkl')
        val = load_gzip('./ML/lstm/lstm_val_data.pkl')
        test = load_gzip('./ML/lstm/lstm_test_data.pkl')

        self.time_interval = time_interval
        if not new:
            try:
                self.model = torch.load(f"./ML/lstm/lstm_model_{self.time_interval}_train.pth")
            except ValueError as e:
                print('----ERROR! no existing model----')

        else:
            if time_interval:
                self.model = LSTM_Model_for_seqtosin(input_dim, hidden_dim, layer_dim, output_dim, self.DROP_OUT,
                                                     time_interval)
            else:
                self.model = LSTM_Model_for_sintosin(input_dim, hidden_dim, layer_dim, output_dim, self.DROP_OUT)

        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)

        train_dataset = LSTMDataset(train,time_interval = time_interval)
        val_dataset = LSTMDataset(val, time_interval=time_interval)
        test_dataset = LSTMDataset(test, time_interval=time_interval)

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.BATCH_SIZE)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=self.BATCH_SIZE)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.BATCH_SIZE)

        self.flog(0,f"LSTM_time{self.time_interval}_model initialize successfully")
        self.flog(0,"LSTM MODEL")

        sum_ = 0
        for name, param in self.model.named_parameters():
            mul = 1
            for size_ in param.shape:
                mul *= size_  # 统计每层参数个数
            sum_ += mul  # 累加每层参数个数
            print(f"{name}:{param.shape}")
        print('参数个数：', sum_)

    def _base_line(self):
        """
        :func: use shift(1) as baseline
        """
        loss = []
        for _,val in enumerate(self.val_loader):
                response = val[1]
                val_loss = self.criterion(response[:,:-1,:],response[:,1:,:])
                loss.append(val_loss)
        loss = torch.tensor(loss, dtype=torch.float)
        self.flog(0,f"the base line loss is {torch.mean(loss)}")
        return torch.mean(loss)

    def _validate(self,epoch):
        loss = []
        with torch.no_grad():
            for _, val in enumerate(self.val_loader):
                feature = val[0]
                response = val[1]
                output = self.model(feature)
                loss.append(self.criterion(output, response))
            loss = torch.tensor(loss, dtype=torch.float)
            self.flog(epoch,f"the val loss is {torch.mean(loss)}")

        return torch.mean(loss)

    def train(self):
        """
        :func: model train function
               train each batch and judge whether validate better and whether train loss nan

        *can have some learning rate change and more accurate validate
        """
        self.model.train()
        ##epoch_loss = 0
        base_line_loss = self._base_line()
        validate_loss = self._validate(0)
        self.flog(0,"train beginning")

        for i in range(self.EPOCHS):
            sum_train_loss = []
            for _, train in enumerate(self.train_loader):
                feature = train[0]
                response = train[1]

                for _ in range(self.ITER):
                    self.optimizer.zero_grad()
                    output = self.model(feature)

                    loss = self.criterion(output,response)
                    loss.backward()

                    self.optimizer.step()

                sum_train_loss.append(loss)
            sum_train_loss = torch.tensor(sum_train_loss, dtype=torch.float)
            '''if abs(sum_train_loss-epoch_loss)/epoch_loss <= self.CONVERGE:
                print(f"----Epoch{i} Loss Converge----")
                break

            epoch_loss = sum_train_loss'''
            self.flog(i+1,f"the train loss is {torch.mean(sum_train_loss)}")

            if (val_l := self._validate(i+1)) <= base_line_loss:
                if val_l < validate_loss:
                    self.flog(i+1,"have improvement")
                    self.flog(i + 1, "save model parameters")

                    torch.save(self.model, f"./ML/lstm/lstm_model_{self.time_interval}_train.pth")
                    validate_loss = val_l
            else:
                self.flog(i+1,"model is not good")

            if np.isnan(val_l):
                self.model = torch.load(f"./ML/lstm/lstm_model_{self.time_interval}_train.pth")
                validate_loss = self._validate(i+1)
                self.flog(i+1,"the model nan retrain")



            '''if (i+1) % 5 == 0:
                print(f"----save model parameters ----")
                torch.save(self.model,'./ML/lstm/lstm_model_train.pth')'''

        self.flog('final', f"the final loss is {torch.mean(sum_train_loss)}")

    def test(self):
        """
        :func: test the model on test data

        *need to backtest
        """
        loss = []
        sign = []
        with torch.no_grad():
            for _, test in enumerate(self.test_loader):
                feature = test[0]
                response = test[1]
                output = self.model(feature)

                loss.append(self.criterion(output, response))

                direction_right = ((torch.sign(output) == torch.sign(response)).sum())/(response.shape[0]*response.shape[1])
                sign.append(direction_right)
            loss = torch.tensor(loss,dtype=torch.float)
            sign = torch.tensor(sign,dtype=torch.float)

            self.flog("test", f"the test loss is {torch.mean(loss)}")
            self.flog("test", f"the test sign accurate is {torch.mean(sign)}")

    def _adjust(self):
        """
        :func: for the learning rate ..etc
        """
        lr = [0.1,0.01,0.001,0.001]
        pass


if __name__ == "__main__":
    lstm = LSTM_Process()
    lstm.initialize(new=False,input_dim=13,hidden_dim=100,layer_dim=2,output_dim=1,  time_interval=0)
    #lstm.train()
    lstm.test()



