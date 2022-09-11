# This code is whole ddpg process

import numpy as np
from helper import *
from dataprocess import DataProcessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical


from Env import BasisEnv
from ddpg_model import DDPG

__author__ = 'Wenter'

__all__ = ['LSTMDataset',
           'DDPG_Process']

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# same as lstm
class LSTMDataset(Dataset):
    def __init__(self,
                 data,
                 pad: int = 4794,
                 time_interval: int = 0):
        self.data = data
        self.date_index = self.data.index.get_level_values(0).unique()
        self.pad = pad
        self.time_interval = time_interval

    def __len__(self):
        return len(self.date_index)

    def __getitem__(self, idx):
        idx = self.date_index[idx]
        data = torch.tensor(self.data.loc[idx].values, dtype=torch.float,device=device)

        if data.shape[0] < self.pad:
            data = F.pad(data, (0, 0, self.pad - data.shape[0], 0), "constant", 0)

        features = data[:, :-1]

        if self.time_interval:
            response = data[self.time_interval - 1:, -1:]
        else:
            response = data[:, -1:]

        return features, response

# similar to the lstm but need to create env each iter
# * can be more accurate
class DDPG_Process:
    DROP_OUT: int = 0.2
    BATCH_SIZE: int = 10
    EPOCHS: int = 10
    ITER: int = 32
    LR: int = 0.01
    CONVERGE: int = 0.001
    SAVE: list = []

    def __init__(self, ):
        self.model = None
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
        # self.model = DDPG([13, 1], 1, 10, 100, 1, 7, 0.1)
        self.model = torch.load(f"./ML/rl/rl_model_{self.time_interval}_train.pth")



        train_dataset = LSTMDataset(train, time_interval=time_interval)
        val_dataset = LSTMDataset(val, time_interval=time_interval)
        test_dataset = LSTMDataset(test, time_interval=time_interval)

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.BATCH_SIZE)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=self.BATCH_SIZE)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.BATCH_SIZE)

        self.flog(0, f"DDPG_time{self.time_interval}_model initialize successfully")
        self.flog(0, "DDPG MODEL")

    def _create_env(self, data):
        self.env = BasisEnv(data, self.time_interval)

    def _train_on_env(self, batch_size):
        train_door = 0

        s = self.env.reset()
        d = None
        while not d:
            a = self.model.take_action(s)
            s1, r, d = self.env.step(a)
            for b in range(batch_size):
                self.model.perceive([s[0][b], s[1][b]], [s1[0][b], s1[1][b]], a[b], r[b])

            train_door += 1

            if train_door % 100 == 0:
                self.model.train()

    def train(self):
        self.flog(0, "train beginning")

        for i in range(self.EPOCHS):

            for c, train in enumerate(self.train_loader):
                print(c)
                self._create_env(train)
                self._train_on_env(train[0].shape[0])

            self._validate(i)
            self._save()
        self._test()

    def _validate(self,epoch):
        account = 1
        for _, val in enumerate(self.val_loader):
            env = BasisEnv(val, self.time_interval)
            s = env.reset()
            d = None
            while not d:
                a = self.model.take_action(s)
                s1, _, d = env.step(a)
            account *= torch.prod(s1[1])

        return self.flog(epoch, f'validate account{account}')

    def _test(self):
        account = 1
        for _, val in enumerate(self.val_loader):
            env = BasisEnv(val, self.time_interval)
            s = env.reset()
            d = None
            while not d:
                a = self.model.take_action(s)
                s1, _, d = env.step(a)
            account *= torch.prod(s1[1])

        return self.flog('final', f'test account{account}')

    def _save(self):
        torch.save(self.model, f"./ML/rl/rl_model_{self.time_interval}_train.pth")




if __name__ == "__main__":
    ddpg = DDPG_Process()
    ddpg.initialize(time_interval=5)
    ddpg._test()


