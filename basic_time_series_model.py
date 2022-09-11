# This Code is for the basic time series model
# ARIMA and Garch by using python or R(rpy2)

from helper import *
import numpy as np
import pandas as pd
from dataprocess import DataProcessing
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from arch import arch_model
import itertools
import torch

# R
from rpy2.robjects import r, pandas2ri,numpy2ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

from BackTest import *

__author__ = 'Wenter'

__all__ = ['Basic_TS_Py',
           'Basic_TS_R']


# ARIMA and GARCH by using python
class Basic_TS_Py:

    def __init__(self,
                 train_data:pd.DataFrame,
                 test_data:pd.DataFrame):
        self.train_data = train_data
        self.test_data = test_data
        self.arima_model = None
        self.garch_model = None

    @staticmethod
    def _get_stationary(data):
        """
        :func: judge whether stationary(confidence interval 0.05)
        :param data: data for test
        :return: stationary
        """

        if ts.adfuller(data)[1] <= 0.05:
            if ts.kpss(data)[1] >= 0.05:
                print('data stationary')
                return True

        print('data not stationary')
        return False

    @staticmethod
    def _white_noise(data):
        """
        :func: judge whether whitenoise (confidence interval 0.05)
        :param data: data for test
        :return: WN
        """
        if acorr_ljungbox(data, lags=[10]).values[0][1] >= 0.05:
            print('data WN')
            return True

        print('data not Wn')
        return False

    @staticmethod
    def _plot_acf(data):
        """
        :func: acf plot
        """
        sm.graphics.tsa.plot_acf(data, lags=20)

    def _select_best_model(self,
                           max_ar:int = 8,
                           max_ma:int = 8):
        """
        :func: select best arima model
        :param max_ar: max ar parameter
        :parma max_ma: max ma parameter
        :return: p for ar,q for ma
        """

        if self._get_stationary(self.train_data):
            if not self._white_noise(self.train_data):
                p, q = sm.tsa.arma_order_select_ic(self.train_data, max_ar=max_ar, max_ma=max_ma, ic='bic')[
                    'bic_min_order']
                return p, q
        return 0

    def arima_fit(self,
                  max_ar: int = 4,
                  max_ma: int = 4):
        """
        :func: fit arima
        :param max_ar: max ar parameter
        :param max_ma: max ma parameter
        """

        try:
            p, q = self._select_best_model(max_ar, max_ma)

        except TypeError as e:
            print('p,q not define')

        self.arima_model = sm.tsa.arima.ARIMA(self.train_data, order=(p, 0, q)).fit()

        print('arima fit done')
        print(self.arima_model.summary())

    def arima_test(self):
        """
        :func:test whether arima model fit well
        """

        resid = self.arima_model.resid
        if self._white_noise(resid):
            print('arima fit mean well')
            if self._white_noise(abs(resid)):
                print('volatility cluster fit well')
            else:
                print('volatility cluster need to fit')
        else:
            print('arima fit mean bad')

        sm.qqplot(resid, line='q', fit=True)
        # self._plot_acf(resid)
        # self._plot_acf(abs(resid))
        self.arima_model.plot_diagnostics(figsize=(15, 10))
        plt.show()

    def garch_fit(self,
                  pv: int = 1,
                  qv: int = 1):
        """
        :func: only fit garch cannot fit arima+garch
        :param pv: garch p
        :param qv: garch q
        """
        distribution = ['normal', 't']
        vol = ['GARCH', 'EGARCH']
        bic = 0
        for cartesian in itertools.product(distribution, vol):
            garch_model = arch_model(self.arima_model.resid, mean='Zero', dist=cartesian[0], vol=cartesian[1], p=pv,
                                     q=qv).fit()
            if garch_model.bic < bic:
                self.garch_model = garch_model
                bic = garch_model.bic
        print('garch fit done')
        print(self.garch_model)

    def garch_test(self):
        """
        :func: test garch fit well
        """
        resid = self.garch_model.resid

        self._white_noise(resid)

        garch_std_resid = pd.Series(self.garch_model.resid / self.garch_model.conditional_volatility)
        fig = plt.figure(figsize=(15, 8))

        # Residual
        garch_std_resid.plot(ax=fig.add_subplot(3, 1, 1), title='GARCH Standardized-Residual', legend=False)

        # ACF/PACF
        self._plot_acf(garch_std_resid)

        # QQ-Plot & Norm-Dist
        sm.qqplot(garch_std_resid, line='s', ax=fig.add_subplot(3, 2, 5))
        plt.title("QQ Plot")
        fig.add_subplot(3, 2, 6).hist(garch_std_resid, bins=40)
        plt.title("Histogram")

        plt.tight_layout()
        plt.show()


# ARIMA and GARCH by using R
class Basic_TS_R:
    """
    :class: Because python package can't fit arima+garch together, using R instead
    """

    def __init__(self,
                 train: pd.Series,
                 val: pd.Series,
                 test: pd.DataFrame,
                 time_dim:int) -> None:

        self.train_data = train
        self.val_data = val
        self.test_data = test

        self.time_dim = time_dim

        self.limit = None
        # importr means  using the R package
        self.arima_R = importr('forecast')
        self.garch_R = importr('rugarch')

        self.arima_coefficients = None
        self.garch_coefficients = None

    @staticmethod
    def _get_stationary(data) -> bool:
        """
        :func: judge whether stationary(confidence interval 0.05)
        :param data: data for test
        :return: stationary
        """
        if ts.adfuller(data)[1] <= 0.05:
            if ts.kpss(data)[1] >= 0.05:
                print('data stationary')
                return True

        print('data not stationary')
        return False

    @staticmethod
    def _white_noise(data) -> bool:
        """
        :func: judge whether whitenoise (confidence interval 0.05)
        :param data: data for test
        :return: WN
        """
        if acorr_ljungbox(data, lags=[10]).values[0][1] >= 0.05:
            print('data WN')
            return True

        print('data not Wn')
        return False

    @staticmethod
    def _plot_acf(data) -> None:
        """
        :func: plot acf
        :param data: .
        """
        sm.graphics.tsa.plot_acf(data, lags=20)

    @staticmethod
    def mse(data_original, data_predict) -> torch.tensor:
        """
        :func: cal mse
        :param data_original: test data
        :param data_predict: fit data
        """
        dat = torch.tensor(data_original)
        pre = torch.tensor(data_predict)
        return torch.nn.MSELoss(reduction='sum')(dat, pre)

    @staticmethod
    def get_sign(dat ,pre) -> int:
        if dat > 0:
            if pre > 0:
                return 1
        else:
            if pre <= 0:
                return 1
        return 0

    def arima_fit(self,
                  max_ar: int = 8,
                  max_ma: int = 8) -> None:
        """
        :func: using R to fit arima
        auto_arima can get the best model

        :param max_ar： max ar parameter
        :param max_ma： max ma parameter
        :result: get the arima coefficient
        """
        if self._get_stationary(self.train_data):
                pandas2ri.activate()

                arima_model = self.arima_R.auto_arima(self.train_data.values, max_p=max_ar, max_q=max_ma, ic="bic",
                                                      seasonal=False, trace=False)
                p, q = arima_model[6][0], arima_model[6][1]
                resid = arima_model[7]

                self.arima_coefficients = {'p': arima_model[0][:p], 'q': arima_model[0][p:p + q], 'resid': resid}
                print('arima fit done')
                print(f"arima coefficient is{self.arima_coefficients['p'],self.arima_coefficients['q']}")
                pandas2ri.deactivate()


        else:
            print('train data not stationary')

    def arima_test(self) -> None:
        """
        :func: test arima whether fit well
        """
        resid = self.arima_coefficients['resid']
        if self._white_noise(resid):
            print('arima fit mean well')
            if self._white_noise(abs(resid)):
                print('volatility cluster fit well')
            else:
                print('volatility cluster need to fit')
        else:
            print('arima fit mean bad')

        # sm.qqplot(resid, line='q', fit=True)
        # self._plot_acf(resid)
        # self._plot_acf(abs(resid))
        # plt.hist(resid,figsize=(15, 10))
        # plt.show()

    def val(self, coefficients) -> torch.tensor:
        """
        :func: validate to choose the model
        :param coefficients: model coefficients
        :return: mse of validate
        """

        val_diff = (self.val_data/self.val_data.shift(1)).apply(np.log).dropna()
        mu = coefficients['mu']
        p = coefficients['p']
        q = coefficients['q']
        p_len = len(p)
        q_len = len(q)

        ar = np.zeros(p_len)
        resid = np.zeros(q_len)
        pred_list = []
        for j, _ in enumerate(val_diff):
            pred = mu
            for p_index in range(p_len):
                pred += p[p_index] * (ar[-p_index - 1])
            for q_index in range(q_len):
                pred += q[q_index] * resid[-q_index - 1]

            pred_list.append(pred)

            if p_len:
                ar = np.append(np.delete(ar, 0), val_diff.iloc[j] - mu)
            if q_len:
                resid = np.append(np.delete(resid, 0), val_diff.iloc[j] - pred)

        return self.mse(val_diff, pred_list), np.mean(pred_list),np.std(pred_list)

    def garch_fit(self) -> None:
        """
        :func: fit arima + garch by R
               validate various model to find better
        :result: get arima + garch coefficient
        # directly use series.values
        # by .slot to get the result
        """
        pandas2ri.activate()
        p = len(self.arima_coefficients['p'])
        q = len(self.arima_coefficients['q'])

        garch_type = ['sGARCH', 'eGARCH', 'iGARCH']
        dist = ['norm', 'std']

        mse = []
        garch_team = []
        garch_ = []
        limit = []

        for g in itertools.product(garch_type, dist):

            garch_team.append(g)
            garch_model = self.garch_R.ugarchspec(
                mean_model=robjects.r(f'list(armaOrder = c({p},{q}))'),
                variance_model=robjects.r(f'list(model="{f"{g[0]}"}",garchOrder=c(1,1))'),
                distribution_model=g[1]
            )

            garch_fitted = self.garch_R.ugarchfit(
                spec=garch_model,
                data=self.train_data.values,
                out_sample=10
            )

            garch_result = garch_fitted.slots['fit']
            mu = garch_result[9][0]
            resid = garch_result[8]

            garch_coefficients = {'mu': mu, 'p': garch_result[9][1:p + 1], 'q': garch_result[9][p + 1:p + q + 1],
                                       'resid': resid}

            garch_.append(garch_coefficients)
            result = self.val(garch_coefficients)
            mse.append(result[0])
            limit.append(result[1:])

        self.garch_coefficients =garch_[np.argmax(mse)]

        mean = limit[np.argmax(mse)][0]
        std = limit[np.argmax(mse)][1]
        self.limit = [mean-2*std, mean+2*std]

        print(f"the better garch type is {garch_team[np.argmax(mse)]}")
        print(self.limit)

        pandas2ri.deactivate()
        print('arima garch fit done')
        print(f"arima garch coeffecient is{self.garch_coefficients['mu'],self.garch_coefficients['p'], self.garch_coefficients['q']}")

    def garch_test(self) -> None:
        """
        :func: test whether garch fit well
        """

        resid = self.garch_coefficients['resid']
        if self._white_noise(resid):
            if self._white_noise(abs(resid)):
                print('arima garch fit well')

        else:
            print('arima garch fit bad')

        # sm.qqplot(resid, line='q', fit=True)
        # self._plot_acf(resid)
        # self._plot_acf(abs(resid))
        plt.hist(resid)
        plt.show()

    def signal(self,
               data: pd.DataFrame) -> pd.Series:
        """
        :func: output trade signal
        :param data: test data
        :return: signal series each time index
        """
        data = data.basis.rolling(self.time_dim).mean().iloc[np.where(~np.arange(4800) % self.time_dim == 0)]

        test_diff = (data / data.shift(1)).apply(np.log).dropna()
        signal = pd.Series(index=data.index,data=np.zeros(len(data)))

        mu = self.garch_coefficients['mu']
        p = self.garch_coefficients['p']
        q = self.garch_coefficients['q']
        p_len = len(p)
        q_len = len(q)

        ar = np.zeros(p_len)
        resid = np.zeros(q_len)

        for j, _ in enumerate(test_diff):
            pred = mu
            for p_index in range(p_len):
                pred += p[p_index] * (ar[-p_index - 1])
            for q_index in range(q_len):
                pred += q[q_index] * resid[-q_index - 1]

            if pred >= self.limit[1]:
                signal.iloc[j] = -1
            elif pred <= self.limit[0]:
                signal.iloc[j] = 1

            if p_len:
                ar = np.append(np.delete(ar, 0), test_diff.iloc[j] - mu)
            if q_len:
                resid = np.append(np.delete(resid, 0), test_diff.iloc[j] - pred)

        return signal


'''    def predict(self,
                data: pd.DataFrame,
                model_type: str = 'GARCH') -> tuple:
        """
        :func: predict by using arima + garch
               can not directly predict by use the package
               Use formula to calculate,only get the arima part coefficient
               Use MSE and sign direction to judge
        :param data: test data
        :param model_type: arima, garch,etc..
        * can get more useful standard
        """
        if model_type == 'GARCH':

            mu = self.garch_coefficients['mu']
            p = self.garch_coefficients['p']
            q = self.garch_coefficients['q']
            p_len = len(p)
            q_len = len(q)

            mse_list = []
            mse_true_sign_list = []
            sign_error_list = []
            print(f'the test data is {data}')

            for i, _ in enumerate(data.columns):
                resid = np.zeros(q_len)
                test_dat = data.iloc[:, i]

                pred_list = []
                pre_sign_list = []
                pre_true_sign_index = []
                for j, _ in enumerate(test_dat):
                    pred = mu
                    for p_index in range(p_len):
                        pred += p[p_index] * (test_dat.iloc[j - p_index - 1] - mu)
                    for q_index in range(q_len):
                        pred += q[q_index] * resid[-q_index - 1]

                    pred_list.append(pred)
                    resid = np.append(np.delete(resid, 0), data.iloc[j] - pred)

                    if self.get_sign(test_dat.iloc[j],pred):
                        pre_sign_list.append(1)
                        pre_true_sign_index.append(j)

                mse_list.append(self.mse(test_dat, pred_list))
                mse_true_sign_list.append(self.mse(test_dat.iloc[pre_true_sign_index],pd.Series(pred_list).iloc[pre_true_sign_index].values))
                sign_error_list.append(sum(pre_sign_list)/len(test_dat))

            return mse_list,sign_error_list,mse_true_sign_list'''

'''    def result(self) -> None:
        print('the test result is:')
        result = self.predict(self.test_data)
        print(f'the mse list is {result[0]}')
        print(f'the sum mse is {sum(result[0])}')
        print(f'the sign error list is {result[1]}')
        print(f'the mean sign error is {np.mean(result[1])}')
        print(f'the true sign mse is {result[2]}')'''


if __name__ == '__main__':
    # 20 time interval
    dp = DataProcessing(ALL_DATES)
    train, val, test = dp.data_for_arima(20)

    # diff for stationary timeseries
    train = (train / train.shift(1)).apply(np.log).dropna()

    # model fit
    basic_model = Basic_TS_R(train, val, test, 20)
    basic_model.arima_fit()
    basic_model.arima_test()
    basic_model.garch_fit()
    basic_model.garch_test()

    # backtest
    backtest = BackTest(basic_model, test, 20)
    backtest.backtest_real()


    # Parallel
    '''
    mu = basic_ts_r.garch_coefficients['mu']
    p = basic_ts_r.garch_coefficients['p']
    q = basic_ts_r.garch_coefficients['q']
    p_len = len(p)
    q_len = len(q)


    def mse(data_original, data_predict):
        dat = torch.tensor(data_original)
        pre = torch.tensor(data_predict)
        return torch.nn.MSELoss(reduction='sum')(dat, pre)


    def get_sign(dat, pre):
        if dat > 0:
            if pre > 0:
                return 1
        else:
            if pre <= 0:
                return 1
        return 0


    def base_predict(data_columns,data, mu, p, q, p_len, q_len):
        resid = np.zeros(q_len)
        test_dat = data[data_columns]

        pred_list = []
        pre_sign_list = []
        pre_true_sign_index = []

        for j,_ in enumerate(test_dat):
            pred = mu
            for p_index in range(p_len):
                pred += p[p_index] * (test_dat.iloc[j - p_index - 1] - mu)
            for q_index in range(q_len):
                pred += q[q_index] * resid[-q_index - 1]

            pred_list.append(pred)
            resid = np.append(np.delete(resid, 0), data.iloc[j] - pred)

            if get_sign(test_dat.iloc[j], pred):
                pre_sign_list.append(1)
                pre_true_sign_index.append(j)

        return mse(test_dat, pred_list), mse(test_dat.iloc[pre_true_sign_index],
                                                       pd.Series(pred_list).iloc[pre_true_sign_index].values), sum(
            pre_sign_list) / len(test_dat)

    result_df = pd.DataFrame(
            parLapply(basic_ts_r.test_data.columns, base_predict, data=basic_ts_r.test_data, mu=mu, p=p, q=q, p_len=p_len, q_len=q_len),
            columns=['mse_list', 'sign_error', 'true_sign_mse'])

    print(result_df)
    '''



