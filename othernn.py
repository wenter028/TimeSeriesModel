# this code is for some combination of cnn and lstm
import numpy as np

from helper import *
from dataprocess import DataProcessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lstm import LSTMDataset

__author__ = 'Wenter'

__all__ = ['LSTM_CNN_OUTPUT_Model_for_seqtosin',
           'LSTM_CNN_INPUT_Model_for_seqtosin',
           'LSTnet_seqtosin',
           'convGRU_sintosin']


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class LSTM_CNN_OUTPUT_Model_for_seqtosin(nn.Module):
    """
    :class:LSTM CNN MODEL for sequence time index to single time index
    CNN output instead of fc
    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, drop_out, time_interval):
        """
        init
        :param input_dim: feature dim
        :param hidden_dim: hidden dim of linear dense of lstm model
        :param layer_dim: layer of lstm dense vertical
        :param output_dim: output dim
        :param drop_out: drop out dense
        :param time_interval: how much time index for one time index
        """
        super(LSTM_CNN_OUTPUT_Model_for_seqtosin, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.time_interval = time_interval

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=drop_out)

        self.cnn2d1 = nn.Conv2d(1, 1, kernel_size=(3, 20), stride=(1, 5))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 4), stride=(1, 1))
        self.cnn2d2 = nn.Conv2d(1, 1, kernel_size=(2, 5), stride=(1, 3))
        self.fc = nn.Linear(4, output_dim)
        self.dropout = nn.Dropout(drop_out)

        self._init_weights()

    def _init_weights(self):
        """
        :func:initialize the fc for train
        """
        nn.init.xavier_uniform_(self.cnn2d1.weight)
        nn.init.xavier_uniform_(self.cnn2d2.weight)
        nn.init.xavier_uniform_(self.fc.weight)

        nn.init.normal_(self.cnn2d1.bias, std=1e-6)
        nn.init.normal_(self.cnn2d2.bias, std=1e-6)
        nn.init.normal_(self.fc.bias, std=1e-6)

    def forward(self, x):
        """
        :param x: input
        h: output
        c: hidden output sum of update gate and forget gate
        Using convolution to get time interval combine
        :return: single time index
        """
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = out.unsqueeze(1)
        out = self.cnn2d1(out)
        out = self.maxpool(F.relu(out))
        out = self.dropout(out)
        out = self.cnn2d2(out)

        out = F.relu(out.squeeze(1))
        out = self.fc(out)

        return out


class LSTM_CNN_INPUT_Model_for_seqtosin(nn.Module):
    """
    :class:LSTM CNN MODEL for sequence time index to single time index
    CNN to convo the input to get feature
    """
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, drop_out, time_interval):
        """
        init
        :param input_dim: feature dim
        :param hidden_dim: hidden dim of linear dense of lstm model
        :param layer_dim: layer of lstm dense vertical
        :param output_dim: output dim
        :param drop_out: drop out dense
        :param time_interval: how much time index for one time index
        """
        super(LSTM_CNN_INPUT_Model_for_seqtosin, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.time_interval = time_interval

        self.cnn2d1 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 2))
        self.cnn2d2 = nn.Conv2d(1, 1, kernel_size=(2, 2), stride=(1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 3), stride=(1, 1))
        self.dropout = nn.Dropout(drop_out)

        self.lstm = nn.LSTM(3, hidden_dim, layer_dim, batch_first=True, dropout=drop_out)
        self.fc = nn.Linear(hidden_dim, 1)

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
        Using cnn to convo input to get features
        h: output
        c: hidden output sum of update gate and forget gate
        :return: all the output h[1]
        """
        x = x.unsqueeze(1)
        x = self.cnn2d1(x)
        x = self.maxpool(F.relu(x))
        x = self.dropout(x)
        x = self.cnn2d2(x)
        x = F.relu(x)
        x = x.squeeze(1)

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(F.sigmoid(out))

        return out


class LSTnet_seqtosin(nn.Module):
    """
    :class:LSTnet MODEL for sequence time index to single time index
    :CNN for input
     Rnn and Rnn skip for hidden
     add AR on the final

    :论文名称：Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks
    :论文下载：https://dl.acm.org/doi/abs/10.1145/3209978.3210006
    :论文年份：SIGIR 2018
    :论文被引：594（2022/04/21）
    :论文代码：https://github.com/laiguokun/LSTNet
    :论文数据：https://github.com/laiguokun/multivariate-time-series-data

    """

    def __init__(self, input_dim, hidden_dim, layer_dim, drop_out, time_interval):
        """
        init
        :param input_dim: feature dim
        :param hidden_dim: hidden dim of linear dense of lstm model
        :param layer_dim: layer of lstm dense vertical
        :param drop_out: drop out dense
        :param time_interval: how much time index for one time index
        Gru instead of Rnn for quick
        * can use args.input_dim instead
        """
        super(LSTnet_seqtosin, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.drop_out = drop_out

        self.time_interval = time_interval

        self.conv1d = nn.Conv1d(self.input_dim, 6, kernel_size=5)
        self.rnn = nn.GRU(6, self.hidden_dim, self.layer_dim, batch_first=True, dropout=0.2)
        self.rnn_skip = nn.GRU(6, self.hidden_dim, self.layer_dim, batch_first=True, dropout=0.2)

        self.drop = nn.Dropout(self.drop_out)

        self.fc1 = nn.Linear(self.hidden_dim*2, 1)
        self.fc2 = nn.Linear(self.time_interval, 1)
        self.fc3 = nn.Linear(self.input_dim, 1)

    def forward(self, x):
        """
        LSTnet process
        conv input
        rnn,rnn skip fit add autoregressive
        :param x: input
        hf init gru
        hc init gru skip
        :return: all the output h[1]
        """
        batch_size = x.shape[0]
        hf = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_().to(device)
        hs = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_().to(device)

        # Conv1d
        z = F.relu(x)
        z = z.transpose(1, 2).contiguous()
        z = self.conv1d(z)
        z = z.transpose(1, 2).contiguous()
        z = self.drop(z)
        z = F.tanh(z)

        # Rnn
        fr, _ = self.rnn(z, hf)

        # Rnn_skip
        s = z.unfold(1, 1, self.time_interval).view(batch_size, -1, z.shape[2])

        fk, _ = self.rnn_skip(s, hs)

        s = torch.tensor([]).to(device)
        for i in range(fk.shape[1]):
            p = fk[:, i:i + 1, :]
            p = p.repeat(1, self.time_interval, 1)
            s = torch.cat((s, p), 1)

        # Cat
        r = torch.cat((fr, s), 2)
        r = self.fc1(r)
        r = F.sigmoid(r)

        # Autoregressive
        a = x.unfold(1, self.time_interval, 1)
        a = self.fc2(a).view(batch_size, -1, self.input_dim)
        a = self.fc3(a)

        res = a + r

        return res

class convGRU_sintosin(nn.Module):
    """
    :class:convGru MODEL for single time index to single time index
    Conv dense instead hidden fc dense of GRU(LSTM)
    :论文地址：https://aaai.org/ojs/index.php/
    :知乎：人人都能看懂的GRU - 陈诚的文章 - 知乎 https://zhuanlan.zhihu.com/p/32481747
    """
    def __init__(self):
        """
        2 layers
        """
        super(convGRU_sintosin, self).__init__()
        self.reset1 = nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=2)
        self.reset2 = nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=2)
        self.update1 = nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=2)
        self.update2 = nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=2)

        self.n1 = nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=2)
        self.n2 = nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=2)

        self.mid = nn.Linear(16, 13)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        """
        :param x: input
        ht: each time index output
        :return: all the output h[1]
        """

        ht = torch.zeros(2,x.shape[0], 16).requires_grad_().to(device)
        output = torch.tensor([]).requires_grad_().to(device)

        for i in range(x.shape[1]):
            out, ht = self.convGru_forward(x[:, i:i + 1, :], ht)
            output = torch.cat((output, out), 1)

        return output

    def convGru_forward(self, x, h_t_1):
        """
        :func: process of conv gru,iter first layer and second layer
        :param x: input
        :param h_t_1: ht from last time
        :return: output(every time index fit),ht(this time index hidden layer)
        """

        h_t1_1 = h_t_1[0, :, :].unsqueeze(1)
        h_t1_2 = h_t_1[1, :, :].unsqueeze(1)

        # First layer
        f1 = torch.cat((x, h_t1_1), 2)

        r1 = F.sigmoid(self.reset1(f1))
        z1 = F.sigmoid(self.update1(f1))

        h_t1_hat_1 = r1.mul(h_t1_1)

        f1 = torch.cat((x, h_t1_hat_1), 2)

        h_hat_1 = F.tanh(self.n1(f1))

        h_t_1 = (1 - z1).mul(h_t1_1) + z1.mul(h_hat_1)

        # Second layer
        k = self.mid(h_t_1)
        f2 = torch.cat((k, h_t1_2), 2)

        r2 = F.sigmoid(self.reset2(f2))
        z2 = F.sigmoid(self.update2(f2))

        h_t1_hat_2 = r2.mul(h_t1_2)

        f2 = torch.cat((x, h_t1_hat_2), 2)

        h_hat_2 = F.tanh(self.n2(f2))

        h_t_2 = (1 - z2).mul(h_t1_2) + z2.mul(h_hat_2)

        output = self.out(h_t_2)

        h_t_1 = h_t_1.squeeze().unsqueeze(0)
        h_t_2 = h_t_2.squeeze().unsqueeze(0)
        ht = torch.cat((h_t_1, h_t_2), 0)

        return output, ht


# same as the lstm process
# * need to optimize
class LSTM_Process:
    DROP_OUT: int = 0.2
    BATCH_SIZE: int = 10
    EPOCHS: int = 10
    ITER: int = 32
    LR: int = 0.01
    CONVERGE: int = 0.001
    SAVE: list = []

    def __init__(self,):
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

        '''
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
                self.model = LSTnet_seqtosin(input_dim, hidden_dim, layer_dim,  self.DROP_OUT,
                                                     time_interval)
            else:
                self.model = convGRU_sintosin()
                pass

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
        loss = []
        for _,val in enumerate(self.val_loader):
                response = val[1]
                val_loss = self.criterion(response[:,:-1,:],response[:,1:,:])
                loss.append(val_loss)
        loss = torch.tensor(loss, dtype=torch.float)
        self.flog(0,f"the base line loss is {torch.mean(loss)}")
        return torch.mean(loss)

    def train(self):
        self.model.train()
        ##epoch_loss = 0
        base_line_loss = self._base_line()
        validate_loss = self._validate(0)
        self.flog(0,"train beginning")

        for i in range(self.EPOCHS):
            sum_train_loss = []
            for j, train in enumerate(self.train_loader):
                feature = train[0]
                response = train[1]
                print(j)

                for _ in range(3):
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

                    #torch.save(self.model, f"./ML/lstm/lstm_model_{self.time_interval}_train.pth")
                    validate_loss = val_l
            else:
                self.flog(i+1,"model is not good")

            if np.isnan(val_l):
                #self.model = torch.load(f"./ML/lstm/lstm_model_{self.time_interval}_train.pth")
                validate_loss = self._validate(i+1)
                self.flog(i+1,"the model nan retrain")



            '''if (i+1) % 5 == 0:
                print(f"----save model parameters ----")
                torch.save(self.model,'./ML/lstm/lstm_model_train.pth')'''



        self.flog('final', f"the final loss is {torch.mean(sum_train_loss)}")

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

    def test(self):
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

    def adjust(self):
        lr = [0.1,0.01,0.001,0.001]
        pass



if __name__ == "__main__":
    lstm = LSTM_Process()
    lstm.initialize(new=True,input_dim=13,hidden_dim=100,layer_dim=2,output_dim=1,  time_interval=0)
    lstm.train()
    lstm.test()





