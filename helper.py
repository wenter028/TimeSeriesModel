import datetime
import pandas as pd
import _pickle as cPickle
import gzip
import dask
import os
import functools
from dask import compute, delayed
from datetime import time
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import warnings


warnings.filterwarnings("ignore")

CORE_NUM = int(os.environ['NUMBER_OF_PROCESSORS'])

PROJ_PATH = 'D:/Code/Work/GTJA/spread basis'
DATA_PATH = 'D:/Code/Work/GTJA/spread basis/wwt_data'
ETF_PATH = DATA_PATH+'/'+'510300'
FUTURE_PATH_NOW = DATA_PATH +'/'+'IF'
FUTURE_PATH_NEXT = DATA_PATH+'/'+'IF_1'

#the data used to train model
ETF_DATA_PATH = 'D:/Code/Work/GTJA/spread basis/ETF'
FUTURE_DATA_PATH = 'D:/Code/Work/GTJA/spread basis/major contract'


#parell
def parLapply(iterable, func, *args, **kwargs):
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(func, *args, **kwargs)
        result = compute([delayed(f_par)(item) for item in iterable])[0]
        return result



# load pickle data
def load(path):
    with open(path,'rb') as file_object:
        raw_data = file_object.read()
    return cPickle.loads(raw_data)


def load_gzip(path):
    with gzip.open(path, 'rb', compresslevel=1) as file_object:
        raw_data = file_object.read()
    return cPickle.loads(raw_data)


def save_gzip(data, path):
    serialized = cPickle.dumps(data)
    with gzip.open(path, 'wb', compresslevel=1) as file_object:
        file_object.write(serialized)



# plot the basis


class basis_plot:
    def __init__(self):
        self._data_etf = None
        self._data_future = None
        self.data = None

        self.day_basis = None

    @staticmethod
    def _get_data(date):
        data_etf = load_gzip(ETF_DATA_PATH + '/' + date)
        data_future = load_gzip(FUTURE_DATA_PATH + '/' + date)

        return data_etf, data_future

    @staticmethod
    def _handle_etf(data):
        data.reset_index(inplace=True)
        data.time = data.time.apply(lambda x: x.time())
        data['wpr_price'] = (data.BidPrice1 * data.AskVolume1 + data.AskPrice1 * data.BidVolume1) * 1000 / (
                    data.BidVolume1 + data.AskVolume1)

        return data

    @staticmethod
    def _handle_future(data):
        data['time'] = data.Time.apply(lambda x: time(x.hour, x.minute, x.second) if x.microsecond < 500000 else 0)
        data['wpr_price'] = (data.bid_price1 * data.ask_vol1 + data.ask_price1 * data.bid_vol1) / (
                    data.bid_vol1 + data.ask_vol1)

        return data

    def _get_basis(self, date):
        self._data_etf = self._handle_etf(self._get_data(date)[0])
        self._data_future = self._handle_future(self._get_data(date)[1])
        data = pd.merge(self._data_etf, self._data_future, on=['time'], suffixes=['_etf', '_future'])
        data.set_index('time', inplace=True)
        data['basis'] = data['wpr_price_future'] - data['wpr_price_etf']

        return data[['wpr_price_etf', 'wpr_price_future', 'basis']]

    def plot_basis_inday(self, date):
        plt.figure(1, figsize=(16, 10))
        plt.title('%s,basis' % date[:10])
        self.data = self._get_basis(date)

        self.data.iloc[np.where(np.arange(self.data.shape[0])%20 == 0)].basis.plot()

    def plot_basis_onday(self, all_dates):
        self.day_basis = pd.DataFrame(columns=['Open', 'Close', 'High', 'Low'])
        for i in all_dates:
            data = self._get_basis(i)
            self.day_basis.loc[i[:10]] = [data.basis.iloc[0], data.basis.iloc[-1], data.basis.max(), data.basis.min()]

        return self.day_basis

    @staticmethod
    def plot_basis_candle(data):
        data.set_index(pd.to_datetime(data.index))
        mpf.plot(data, type='candle')



# get data
class DataLoader:
    def __init__(self):
        self.data_etf = None
        self.data_future = None

        self.second_type = ['half', 'three']

    def _load(self, date):
        self.data_etf = load_gzip(ETF_DATA_PATH + '/' + date)
        self.data_future = load_gzip(FUTURE_DATA_PATH + '/' + date)

        # future data
        self.data_future = self.data_future[
            (self.data_future.Time >= time(9, 30, 0)) & (self.data_future.Time <= time(15, 0, 0))]
        self.data_future['wpr_price'] = ((
                                                 self.data_future.bid_price1 * self.data_future.ask_vol1 + self.data_future.ask_price1 * self.data_future.bid_vol1) / (
                                                 self.data_future.bid_vol1 + self.data_future.ask_vol1)).round(2)

        future_microsecond = self.data_future.iloc[0].Time.microsecond
        # etf data
        self.data_etf.reset_index(inplace=True)
        self.data_etf['Time'] = self.data_etf.time.apply(lambda x: time(x.hour, x.minute, x.second, future_microsecond))
        self.data_etf['wpr_price'] = ((
                                                  self.data_etf.BidPrice1 * self.data_etf.AskVolume1 + self.data_etf.AskPrice1 * self.data_etf.BidVolume1) * 1000 / (
                                                  self.data_etf.BidVolume1 + self.data_etf.AskVolume1)).round(2)

    def get_second_data(self, date,second_type='half'):
        try:
            if second_type not in self.second_type:
                raise ValueError
        except ValueError as e:
            print("get some error of second_type",e)

        self._load(date)

        self.data_etf = self.data_etf[(self.data_etf.Time >= time(9, 30, 0)) & (self.data_etf.Time <= time(15, 0, 0))][
            ['Time', 'wpr_price']]
        self.data_future = \
        self.data_future[(self.data_future.Time >= time(9, 30, 0)) & (self.data_future.Time <= time(15, 0, 0))][
            ['Time', 'wpr_price']]


        if second_type == 'half':
            data = pd.merge(self.data_etf, self.data_future, on='Time', how='outer', suffixes=('_etf', '_future'))
        elif second_type =='three':
            data = pd.merge(self.data_etf, self.data_future, on='Time', how='inner', suffixes=('_etf', '_future'))

        try:
            data = data.sort_values(by='Time').set_index('Time').fillna(method='ffill').dropna()
        except NameError as e:
            print('SECOND_TYPE error',e)

        data['basis'] = data.wpr_price_future - data.wpr_price_etf
        data = data.drop(data[(data.index>time(11,30,0))&(data.index<time(13,0,0))].index)

        return data


