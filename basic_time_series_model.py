from helper import *
import statsmodels.tsa.stattools as ts
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from arch import arch_model

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


##ARIMA
class Basic_TS:
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
        if acorr_ljungbox(data, lags=1).values[0][1] >= 0.05:
            print('data WN')
            return True

        print('data not Wn')
        return False

    @staticmethod
    def _plot_acf(data):
        sm.graphics.tsa.plot_acf(data, lags=20)

    def _select_best_model(self, max_ar=4, max_ma=4):
        if self._get_stationary(self.train_data):
            if not self._white_noise(self.train_data):
                p, q = sm.tsa.arma_order_select_ic(self.train_data, max_ar=max_ar, max_ma=max_ma, ic='bic')[
                    'bic_min_order']
                return p, q
        return 0

    def arima_fit(self):
        try:
            p, q = self._select_best_model()

        except TypeError as e:
            print('p,q not define')

        self.arima_model = sm.tsa.arima.ARIMA(self.train_data, order=(p, 0, q)).fit()

        print('arima fit done')
        print(self.arima_model)

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

        qqplot(resid, line='q', fit=True)
        # self._plot_acf(resid)
        # self._plot_acf(abs(resid))

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
        qqplot(resid, line='q', fit=True)
        plt.show()



















