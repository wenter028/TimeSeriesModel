# This code used for all the model data preprocessing
import pandas as pd
from helper import *
import torch
from datetime import time
import numpy as np
import math

__author__ = 'Wenter'
__all__ = ['DataProcessing']


# get the data processing from original data
class DataProcessing:
    """
    :class: Data process for all model
    """

    TIME_INTERVAL: int = 5
    TRAIN: float = 0.6
    VALIDATE: float = 0.2

    def __init__(self,
                 all_dates: list,
                 second_type: str = 'three',
                 train: float = TRAIN,
                 validate: float = VALIDATE,
                 ) -> None:

        """
        :param: TIME_INTERVAL: for predict interval
        :param: TRAIN: train set divided
        :param: VALIDATE: validate set divided
        """
        self.train = train
        self.validate = validate

        self.second_type = second_type
        self.all_dates = all_dates
        self.time_index = None

    @staticmethod
    def data_for_arima(time_dim) -> tuple:
        """
        :func:data for arima
              train and val data are the mean of each train day basis
              train and test selected by time interval
        :param time_dim: how much time mean
        :return: train,val,test
        """
        data = load_gzip('D:/Code/Work/GTJA/spread basis/data/wholedata.pkl')

        train_date = load_gzip('D:/Code/Work/GTJA/spread basis/data/train_date.pkl')
        val_date = load_gzip('D:/Code/Work/GTJA/spread basis/data/val_date.pkl')
        test_date = load_gzip('D:/Code/Work/GTJA/spread basis/data/test_date.pkl')

        train_data = data.loc[train_date].basis.groupby("Date").apply(
            lambda x: x.rolling(time_dim).mean().iloc[np.where(~np.arange(4800) % time_dim == 0)]).groupby(
            'Time').mean()
        val_data = data.loc[val_date].basis.groupby("Date").apply(
            lambda x: x.rolling(time_dim).mean().iloc[np.where(~np.arange(4800) % time_dim == 0)]).groupby(
            'Time').mean()
        test_data = data.loc[test_date]

        return train_data, val_data, test_data

    @staticmethod
    def data_for_nn() -> tuple:
        """
        :func: data for nn
               get the time index of three second
               data type: Multiindex(date,daytime),13 features
               minmax by train,
               train,val,test by random split
               :return: train,val,test
        """
        data = load_gzip('D:/Code/Work/GTJA/spread basis/data/wholedata.pkl')
        features = ['High_etf', 'Low_etf', 'Volume_diff_etf', 'Wpr_price_etf', 'Wpr_price_rate_etf', 'High_future',
                    'Low_future', 'Volume_diff_future', 'Wpr_price_future', 'Wpr_price_rate_future',
                    'Open_interest_future', 'Open_interest_rate_future', 'basis']

        data = data.loc[:, features]
        train_date = load_gzip('D:/Code/Work/GTJA/spread basis/data/train_date.pkl')
        val_date = load_gzip('D:/Code/Work/GTJA/spread basis/data/val_date.pkl')
        test_date = load_gzip('D:/Code/Work/GTJA/spread basis/data/test_date.pkl')

        # for min max
        train_min = data.loc[train_date].iloc[:, :-1].min()
        train_max = data.loc[train_date].iloc[:, :-1].max()
        data.iloc[:, :-1] = (data.iloc[:, :-1] - train_min) / (train_max - train_min)

        train_data = data.loc[train_date]
        val_data = data.loc[val_date]
        test_data = data.loc[test_date]

        return train_data, val_data, test_data

    def _data_processing(self) -> pd.Index:

        """
        preprocess for the time index
        for each three second
        train data and test data are same treatment
        :return: train,test
        """
        all_data = pd.DataFrame(columns=['Time'])
        dr = DataReader()
        data_columns = ['basis']
        for i in self.all_dates:
            data = dr.get_second_data(i, self.second_type).loc[:, data_columns]
            data.reset_index(inplace=True)
            data.Time = data.Time.apply(lambda x: time(x.hour, x.minute, int(np.floor(x.second / 3)) * 3))
            data = data.drop_duplicates(subset='Time')

            all_data = pd.merge(all_data, data[['Time', 'basis']], on='Time', how='outer')

        all_data.sort_values(by='Time', inplace=True)
        all_data.fillna(method='ffill', inplace=True)
        all_data.set_index('Time', inplace=True)

        return all_data.index

    def whole_data(self) -> None:
        """
        :func:the whole dataset
              handle as log rate
        :return: save gzip file to quick load
        """
        time_index = self._data_processing()

        all_data = pd.DataFrame(index=pd.MultiIndex.from_product([[], []], names=['Date', 'Time']))
        dr = DataReader()

        for i in self.all_dates:
            data = dr.get_second_data(i, self.second_type)
            data.reset_index(inplace=True)
            data.Time = data.Time.apply(lambda x: time(x.hour, x.minute, int(np.floor(x.second / 3)) * 3))
            fix = pd.DataFrame(index=time_index)
            fix.reset_index(inplace=True)
            data = pd.merge(fix, data, on='Time', how='outer')
            data.drop_duplicates(subset='Time', inplace=True)
            data.sort_values(by='Time', inplace=True)
            data.fillna(method='ffill', inplace=True)
            data.set_index(pd.MultiIndex.from_product([[i[:10]], data.Time], names=['Date', 'Time']),
                           inplace=True)

            df = pd.DataFrame(
                index=pd.MultiIndex.from_product([[i[:10]], time_index], names=['Date', 'Time']))

            df['High_etf'] = (data.High_etf / data.PreClose_etf).apply(math.log)
            df['Low_etf'] = (data.Low_etf / data.PreClose_etf).apply(math.log)

            df['Volume_etf'] = data.Volume_etf
            df['Volume_diff_etf'] = data.Volume_etf.diff(1)
            df['Volume_diff_etf'].iloc[0] = data.Volume_etf.iloc[0]

            df['Wpr_price_etf'] = data.wpr_price_etf
            df['Wpr_price_rate_etf'] = (data.wpr_price_etf / 1000 / data.PreClose_etf).apply(math.log)
            df.loc[:, data.columns[9:29]] = data.loc[:, data.columns[9:48]]

            df['High_future'] = (data.High_future / data.PreClose_future).apply(math.log)
            df['Low_future'] = (data.Low_future / data.PreClose_future).apply(math.log)

            df['Volume_future'] = data.Volume_future
            df['Volume_diff_future'] = data.Volume_future.diff(1)
            df['Volume_diff_future'].iloc[0] = data.Volume_future.iloc[0]

            df['Wpr_price_future'] = data.wpr_price_future
            df['Wpr_price_rate_future'] = (data.wpr_price_future / data.PreClose_future).apply(math.log)
            df.loc[:, data.columns[72:92]] = data.loc[:, data.columns[72:92]]

            df['Open_interest_future'] = data.OpenInterest_future
            df['Open_interest_rate_future'] = (data.OpenInterest_future / data.PreOpenInterest_future).apply(math.log)
            df['basis'] = data.basis

            all_data = pd.concat([all_data, df])

        all_data.fillna(0, inplace=True)

        return save_gzip(all_data, 'D:/Code/Work/GTJA/spread basis/data/wholedata.pkl')

    def split_data(self) -> None:
        """
        :func: split data to the train,val,test
        """
        data = load_gzip('D:/Code/Work/GTJA/spread basis/data/wholedata.pkl')
        date = data.index.get_level_values(0).unique()
        date_len = len(date)
        tr, val, te = torch.utils.data.random_split(date, [round(date_len * self.train),
                                                           round(date_len * self.validate),
                                                           date_len - round(date_len * self.train) - round(
                                                               date_len * self.validate)])
        train_date = list(iter(tr))
        val_date = list(iter(val))
        test_date = list(iter(te))

        save_gzip(train_date, 'D:/Code/Work/GTJA/spread basis/data/train_date.pkl')
        save_gzip(val_date, 'D:/Code/Work/GTJA/spread basis/data/val_date.pkl')
        save_gzip(test_date, 'D:/Code/Work/GTJA/spread basis/data/test_date.pkl')


if __name__ == "__main__":
    dp = DataProcessing(ALL_DATES, 'three')
    dp.whole_data()
    dp.split_data()


