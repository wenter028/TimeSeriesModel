# This code is for the CNN model by torch

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

__all__ = ['CNNDataset',
          'CNN_model_for_sintosin',
          'CNN_model_for_seqtosin',
          'CNN_Process'
          ]


class CNNDataset(Dataset):
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


# 1d cnn model not use time series
class CNN_model_for_sintosin(nn.Module):
    """
    :class:CNN MODEL for single time index to single time index
    """

    def __init__(self, input_dim, kernel_size, output_dim, drop_out):
        """
        :param input_dim: feature dim
        :param kernel_size: convolution kernel
        :param output_dim: output dim
        :param drop_out: drop out dense
        Conv1d( input chanel ,output chanel)  ex. (1,2,kernel=3)  (2,1,20) -> (2,2,18)
        """
        super(CNN_model_for_sintosin, self).__init__()
        ##out_dim1 = kernal num
        out_dim1 = int(input_dim / 2)
        out_dim2 = int((out_dim1 - 1) / 2)

        self.cnn1d1 = nn.Conv1d(input_dim, out_dim1, kernel_size=kernel_size)
        self.cnn1d2 = nn.Conv1d(out_dim1 - 1, out_dim2, kernel_size=kernel_size)
        self.maxpool1 = nn.MaxPool1d(2, 1)
        self.fc = nn.Linear(out_dim2, output_dim)
        self.drop = nn.Dropout(drop_out)

        self._init_weights()

    def _init_weights(self):
        """
        :func:initialize the fc for train
        """
        nn.init.xavier_uniform_(self.cnn1d1.weight)
        nn.init.xavier_uniform_(self.cnn1d2.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.cnn1d1.bias, std=1e-6)
        nn.init.normal_(self.cnn1d2.bias, std=1e-6)
        nn.init.normal_(self.fc.bias, std=1e-6)


    def forward(self, x):
        """
        :param x: input
        (batch, 4794,13) -> (b,13,4794) ->(b,6,4794)
        :return: single time index response
        """
        x = torch.transpose(x, 1, 2)
        x = self.cnn1d1(x)

        x = F.relu(x)
        x = torch.transpose(x, 1, 2)
        x = self.maxpool1(x)
        x = torch.transpose(x, 1, 2)

        x = self.drop(x)
        x = self.cnn1d2(x)
        x = F.relu(x)
        x = torch.transpose(x, 1, 2)

        x = self.fc(x)

        return x


class CNN_model_for_seqtosin(nn.Module):
    """
    :class:CNN MODEL for sequence time index to single time index
    """

    def __init__(self, input_dim, kernel_size, output_dim, time_interval, drop_out):
        """
        :param input_dim: feature dim
        :param kernel_size: convolution kernel
        :param output_dim: output dim
        :param time_interval: how much time for a time index
        :param drop_out: drop out dense
        Conv1d( input chanel ,output chanel)  ex. (1,2,kernel=3)  (2,1,20) -> (2,2,18)
        """

        super(CNN_model_for_seqtosin, self).__init__()
        out_dim1 = int(input_dim / 2)
        out_dim2 = int((out_dim1 - 1) / 2)
        self.cnn1d1 = nn.Conv1d(input_dim, out_dim1, kernel_size=kernel_size)
        self.cnn1d2 = nn.Conv1d(out_dim1 - 1, out_dim2, kernel_size=time_interval-kernel_size+1)
        self.maxpool1 = nn.MaxPool1d(2, 1)
        self.fc = nn.Linear(out_dim2, output_dim)
        self.drop = nn.Dropout(drop_out)

        self._init_weights()

    def _init_weights(self):
        """
        :func:initialize the fc for train
        """

        nn.init.xavier_uniform_(self.cnn1d1.weight)
        nn.init.xavier_uniform_(self.cnn1d2.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.cnn1d1.bias, std=1e-6)
        nn.init.normal_(self.cnn1d2.bias, std=1e-6)
        nn.init.normal_(self.fc.bias, std=1e-6)

    def forward(self, x):
        """
        :func: similar as the sintosin, this convo more time index
        """
        x = torch.transpose(x, 1, 2)
        x = self.cnn1d1(x)

        x = F.relu(x)
        x = torch.transpose(x, 1, 2)
        x = self.maxpool1(x)
        x = torch.transpose(x, 1, 2)

        x = self.drop(x)
        x = self.cnn1d2(x)
        x = F.relu(x)
        x = torch.transpose(x, 1, 2)

        x = self.fc(x)

        return x


class CNN_Process:
    """
    :class:CNN train test process
    CONVERGE:converge rate
    *can optimize to apply all nn model
    """
    BATCH_SIZE: int = 10
    DROP_OUT: int = 0.2
    EPOCHS: int = 100
    ITER: int = 32
    LR: int = 0.01
    CONVERGE: int = 0.001
    SAVE = []

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

        self.time_interval = None

    def initialize(self,
                   input_dim: int,
                   kernel_size: int,
                   output_dim: int,
                   new: bool = True,
                   all_dates: list = ALL_DATES,
                   time_interval: int=0):

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

        train = load_gzip('./ML/cnn/cnn_train_data.pkl')
        val = load_gzip('./ML/cnn/cnn_val_data.pkl')
        test = load_gzip('./ML/cnn/cnn_test_data.pkl')

        self.time_interval = time_interval

        if not new:
            try:
                self.model = torch.load(f"./ML/cnn/cnn_model_{self.time_interval}_train.pth")
            except ValueError as e:
                print('----ERROR! no existing model----')

        else:
            if time_interval:
                self.model = CNN_model_for_seqtosin(input_dim,kernel_size,output_dim,time_interval,self.DROP_OUT)
            else:
                self.model = CNN_model_for_sintosin(input_dim,kernel_size,output_dim,self.DROP_OUT)

        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)

        train_dataset = CNNDataset(train,time_interval = time_interval)
        val_dataset = CNNDataset(val, time_interval=time_interval)
        test_dataset = CNNDataset(test, time_interval=time_interval)

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.BATCH_SIZE)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=self.BATCH_SIZE)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.BATCH_SIZE)

        print(f"----CNN_time{self.time_interval}_model initialize successfully")
        print("--------CNN MODEL--------")
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
        print(f"----the base line loss is {torch.mean(loss)}")
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
            print(f"----epoch{epoch} ------the val loss is {torch.mean(loss)}")

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
        print("----train beginning------")
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
            print(f"----Epoch{i+1} ------the train loss is {torch.mean(sum_train_loss)}")

            if (val_l := self._validate(i+1)) <= base_line_loss:
                if val_l < validate_loss:
                    print("----have improvement------")
                    print("----save model parameters ------")
                    torch.save(self.model, f"./ML/cnn/cnn_model_{self.time_interval}_train.pth")
                    validate_loss = val_l
            else:
                print("----model is not good------")

            if np.isnan(val_l):
                self.model = torch.load(f"./ML/cnn/cnn_model_{self.time_interval}_train.pth")
                validate_loss = self._validate(i+1)
                print(f"----Epoch{i+1} ------the model nan retrain")


            '''if (i+1) % 5 == 0:
                print(f"----save model parameters ----")
                torch.save(self.model,'./ML/lstm/lstm_model_train.pth')'''



        print(f"----the train is finished ------the final loss is {torch.mean(sum_train_loss)}")

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
            print(f"---- the test loss is {torch.mean(loss)}")
            print(f"---- the test sign accurate is {torch.mean(sign)}")

    def _adjust(self):
        """
        :func: for the learning rate ..etc
        """
        lr = [0.1,0.01,0.001,0.001]
        pass


if __name__ == "__main__":
    cnn = CNN_Process()
    cnn.initialize(new=True,input_dim=13,kernel_size=3,output_dim=1,time_interval=5)
    cnn.train()
    cnn.test()