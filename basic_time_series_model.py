import numpy as np

from helper import *
import statsmodels.tsa.stattools as ts
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from arch import arch_model

import torch

##R
from rpy2.robjects import r, pandas2ri,numpy2ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

##get the data processing from original data
class DataProcessing:
    def __init__(self, all_dates, time_interval=5, validate=10):

        self.time_interval = time_interval
        self.validate = validate

        self.all_dates = all_dates

    def _data_processing_for_original(self):
        train_data = pd.DataFrame(columns=['Time'])
        test_data = pd.DataFrame(columns=['Time'])

        train_date = self.all_dates[:-int(len(self.all_dates) / self.validate)]
        test_date = self.all_dates[-int(len(self.all_dates) / self.validate):]

        dl = DataLoader()

        for i in train_date:
            data = dl.get_second_data(i, 'three')
            data.reset_index(inplace=True)
            data.Time = data.Time.apply(lambda x: time(x.hour, x.minute, int(np.floor(x.second / 3)) * 3))
            data = data.drop_duplicates(subset='Time')

            train_data = pd.merge(train_data, data[['Time', 'basis']], on='Time', how='outer')

        train_data.sort_values(by='Time', inplace=True)
        train_data.fillna(method='ffill', inplace=True)
        train_data.set_index('Time', inplace=True)

        for i in test_date:
            data = dl.get_second_data(i, 'three')
            data.reset_index(inplace=True)
            data.Time = data.Time.apply(lambda x: time(x.hour, x.minute, int(np.floor(x.second / 3)) * 3))
            data = data.drop_duplicates(subset='Time')
            data.rename(columns={'basis': f'basis_{i[0:10]}'}, inplace=True)

            test_data = pd.merge(test_data, data[['Time', f'basis_{i[0:10]}']], on='Time', how='outer')

        test_data.sort_values(by='Time', inplace=True)
        test_data.fillna(method='ffill', inplace=True)
        test_data.set_index('Time', inplace=True)
        return train_data, test_data

    def data_for_arima(self):
        train_data, test_data = self._data_processing_for_original()

        train_data['mean_basis'] = train_data.mean(axis=1)
        train_data = train_data['mean_basis'].diff(5).iloc[np.where(np.arange(len(train_data)) % 5 == 0)].dropna()

        test_data = test_data.diff(5).iloc[np.where(np.arange(len(train_data)) % 5 == 0)].dropna()

        return train_data, test_data


##ARIMA and GARCH by using python
class Basic_TS_Py:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.arima_model = None
        self.garch_model = None

    @staticmethod
    def _get_stationary(data):
        if ts.adfuller(data)[1] <= 0.05:
            if ts.kpss(data)[1] >= 0.05:
                print('data stationary')
                return True

        print('data not stationary')
        return False

    @staticmethod
    def _white_noise(data):
        if acorr_ljungbox(data, lags=[10]).values[0][1] >= 0.05:
            print('data WN')
            return True

        print('data not Wn')
        return False

    @staticmethod
    def _plot_acf(data):
        sm.graphics.tsa.plot_acf(data, lags=20)

    def _select_best_model(self, max_ar, max_ma):
        if self._get_stationary(self.train_data):
            if not self._white_noise(self.train_data):
                p, q = sm.tsa.arma_order_select_ic(self.train_data, max_ar=max_ar, max_ma=max_ma, ic='bic')[
                    'bic_min_order']
                return p, q
        return 0

    def arima_fit(self, max_ar=4, max_ma=4):
        try:
            p, q = self._select_best_model(max_ar, max_ma)

        except TypeError as e:
            print('p,q not define')

        self.arima_model = sm.tsa.arima.ARIMA(self.train_data, order=(p, 0, q)).fit()

        print('arima fit done')
        print(self.arima_model.summary())

    def arima_test(self):
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

    def garch_fit(self, pv=1, qv=1):
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

##ARIMA and GARCH by using R
class Basic_TS_R:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

        self.arima_R = importr('forecast')
        self.garch_R = importr('rugarch')

        self.arima_coefficients = None
        self.garch_coefficients = None

    @staticmethod
    def _get_stationary(data):
        if ts.adfuller(data)[1] <= 0.05:
            if ts.kpss(data)[1] >= 0.05:
                print('data stationary')
                return True

        print('data not stationary')
        return False

    @staticmethod
    def _white_noise(data):
        if acorr_ljungbox(data, lags=[10]).values[0][1] >= 0.05:
            print('data WN')
            return True

        print('data not Wn')
        return False

    @staticmethod
    def _plot_acf(data):
        sm.graphics.tsa.plot_acf(data, lags=20)

    def arima_fit(self, max_ar=8, max_ma=8):
        pandas2ri.activate()

        arima_model = self.arima_R.auto_arima(self.train_data.values, max_p=max_ar, max_q=max_ma, ic="bic",
                                              seasonal=False, trace=False)
        p, q = arima_model[6][0], arima_model[6][1]
        resid = arima_model[7]

        self.arima_coefficients = {'p': arima_model[0][:p], 'q': arima_model[0][p:p + q], 'resid': resid}
        print('arima fit done')
        pandas2ri.deactivate()

    def arima_test(self):
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

    def garch_fit(self):
        pandas2ri.activate()
        p = len(self.arima_coefficients['p'])
        q = len(self.arima_coefficients['q'])
        garch_model = self.garch_R.ugarchspec(
            mean_model=robjects.r(f'list(armaOrder = c({p},{q}))'),
            variance_model=robjects.r('list(model= "eGARCH",garchOrder=c(1,1))'),
            distribution_model='std'
        )

        garch_fitted = self.garch_R.ugarchfit(
            spec=garch_model,
            data=train_data.values,
            out_sample=10
        )

        numpy2ri.deactivate()
        garch_result = garch_fitted.slots['fit']
        mu = garch_result[9][0]
        resid = garch_result[8]

        self.garch_coefficients = {'mu': mu, 'p': garch_result[9][1:p + 1], 'q': garch_result[9][p + 1:p + q + 1],
                                   'resid': resid}

        pandas2ri.deactivate()
        print('arima garch fit done')

    def garch_test(self):
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

    def predict(self, data, model_type='GARCH'):
        if model_type == 'GARCH':

            mu = self.garch_coefficients['mu']
            p = self.garch_coefficients['p']
            q = self.garch_coefficients['q']
            p_len = len(p)
            q_len = len(q)

            mse_list = []
            for i in range(len(data.columns)):
                resid = np.zeros(q_len)
                test_dat = data.iloc[:, i]
                pred_list = []

                for j in range(max(p_len, q_len), len(test_dat)):
                    pred = mu
                    for p_index in range(p_len):
                        pred += p[p_index] * (test_dat.iloc[j - p_index - 1] - mu)
                    for q_index in range(q_len):
                        pred += q[q_index] * resid[-q_index - 1]

                    pred_list.append(pred)
                    resid = np.append(np.delete(resid, 0), data.iloc[j] - pred)

                mse_list.append(self.mse(test_dat.iloc[-len(pred_list):], pred_list))
        return mse_list

    def result(self):
        return self.predict(self.test_data)

    @staticmethod
    def mse(data_original, data_predict):
        dat = torch.tensor(data_original)
        pre = torch.tensor(data_predict)
        return torch.nn.MSELoss(reduction='sum')(dat, pre)


if __name__ == '__main__':
    dp = DataProcessing(ALL_DATES)
    train_data = dp.data_for_arima()[0]
    test_data = dp.data_for_arima()[1]

    basic_ts_r = Basic_TS_R(train_data,test_data)

    basic_ts_r.arima_fit()
    basic_ts_r.arima_test()
    basic_ts_r.garch_fit()
    basic_ts_r.garch_test()
    print(basic_ts_r.result())



