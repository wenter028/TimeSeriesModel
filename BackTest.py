# This code is for the backtest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__author__ = 'Wenter'

__all__ = ['BackTest',
           ]


class BackTest:
    """
    :class: backtest all the model
    # for simple use 1 basis to backtest (account and fee have been divided)
    # and not use the event driven just use vector backtest
    """
    ACCOUNT = 5000000 / 900
    ETF = 900000
    FUTURE = 3

    ETF_FEE = 0.00025 * 4500000 / 900
    FUTURE_FEE = 0.0023 * 500000 / 900

    FEE = round(ETF_FEE + FUTURE_FEE, 2)

    def __init__(self,
                 model,
                 test_data: pd.DataFrame,
                 time_dim: int) -> None:
        """
        :func: init
        :param model:.
        :param test_data:.
        :param time_dim: time dim to change pos
        """
        self.model = model
        self.test_data = test_data.sort_index()
        self.date = self.test_data.index.get_level_values(0).unique()
        self.time_dim = time_dim
        self.change_pos_time = None

    def _get_change_time(self) -> None:
        """
        :func: get change pos time index
        """
        self.change_pos_time = self.test_data.loc[self.date[0]].iloc[
            np.where(~np.arange(4800) % self.time_dim == 0)].index

    def _out_signal(self) -> None:
        """
        :func: for model to output signal for each time
        """
        self.test_data['signal'] = np.nan

        for date in self.date:
            sig = self.model.signal(self.test_data.loc[date])
            self.test_data.loc[(date, sig.index), 'signal'] = sig.values

    def _plot_net(self, data) -> None:
        """
        :func: plot the net pic
               plot the net and price rate corr pic
        :param data: trade data

        """
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes()
        # price rate
        price = self.test_data.groupby('Date').apply(lambda x: x.iloc[-1]).Wpr_price_rate_etf
        positive = price > 0
        plt.bar(np.arange(len(price)), price, color=positive.map({True: 'r', False: 'c'}), label='price_rate')

        ax1 = plt.twinx()
        # net
        ax1.plot(data.groupby('Date').apply(lambda x: x.iloc[-1]).account, color='0.3', label='net')

        xtick = data.index.get_level_values(0).unique()[
            np.where(np.arange(len(data.index.get_level_values(0).unique())) % 8 == 0)]
        ax.set_xticks(xtick)
        ax.set_xlabel('Date', fontsize=15)
        ax.set_ylabel('Net', fontsize=15)
        ax1.set_ylabel('price rate', fontsize=15)

        sharpe = round(np.mean(data.profit) / np.std(data.profit), 2)
        win_rate = round(len(data.loc[data.profit > 0]) * 100 / len(data), 2)
        net = round(data.iloc[-1].account, 2)
        plt.title(
            f'Backtest {self.model.__class__.__name__} Model   (Sharpe {sharpe}    WinRate {win_rate}%    Net {net})',
            fontsize=12)
        ax.legend()
        ax1.legend()
        plt.show()

    def backtest_simple(self) -> pd.DataFrame:
        """
        :func: not include fee and slippage simple backtest
               signal 1  long ,-1  short ,0 empty
               vector backtest
        :return: backtest result
        """

        self._out_signal()
        self.test_data['mean_basis'] = self.test_data.basis.rolling(window=self.time_dim,
                                                                    min_periods=self.time_dim).mean()

        backtest = self.test_data[['mean_basis', 'signal']].groupby('Date', group_keys=False).apply(
            lambda x: x.iloc[np.where(~np.arange(4800) % self.time_dim == 0)])

        backtest['signal'] = backtest.signal.replace(0, np.nan).groupby('Date', group_keys=False).fillna(
            method='pad').fillna(0)
        backtest['basis_diff'] = backtest['mean_basis'].groupby('Date', group_keys=False).diff(1).shift(-1).fillna(0)

        backtest['profit'] = backtest.basis_diff * backtest.signal

        backtest['account'] = backtest.profit
        backtest['account'].iloc[0] = self.ACCOUNT
        backtest['account'] = backtest.account.cumsum() / self.ACCOUNT

        print('Simple_Backtest')
        self._plot_net(backtest)
        return backtest

    def backtest_real(self):
        """
        :func: include fee and slippage  backtest
               signal 1  long ,-1  short ,0 empty
               vector backtest
        :return: backtest result
        # simple the volume to trade successful, just suppose we can definitely trade by opposite price
        """

        self._out_signal()
        self.test_data['mean_basis'] = self.test_data.basis.rolling(window=self.time_dim,
                                                                    min_periods=self.time_dim).mean()
        col = ['mean_basis', 'signal', 'BidPrice1_etf', 'AskPrice1_etf', 'bid_price1_future', 'ask_price1_future']
        backtest = self.test_data[col].groupby('Date', group_keys=False).apply(
            lambda x: x.iloc[np.where(~np.arange(4800) % self.time_dim == 0)])
        backtest['long'] = backtest.ask_price1_future - backtest.BidPrice1_etf * 1000
        backtest['short'] = backtest.bid_price1_future - backtest.AskPrice1_etf * 1000

        backtest['basis'] = np.nan
        backtest.basis.loc[backtest.signal == 1] = backtest.long
        backtest.basis.loc[backtest.signal == -1] = backtest.short
        backtest.basis.loc[backtest.signal == 0] = backtest.mean_basis

        pos = backtest.signal.groupby('Date', group_keys=False).shift(1)

        backtest['change'] = False
        backtest['change'].loc[pos != backtest.signal] = True

        backtest = backtest.loc[backtest['change']]

        backtest['basis_diff'] = backtest['basis'].groupby('Date', group_keys=False).diff(1).shift(-1).fillna(0)
        backtest['profit'] = backtest.basis_diff * backtest.signal
        backtest.profit -= self.FEE

        backtest['account'] = backtest.profit
        backtest['account'].iloc[0] = self.ACCOUNT
        backtest['account'] = backtest.account.cumsum() / self.ACCOUNT

        print('Real_Backtest')
        self._plot_net(backtest)

        return backtest

