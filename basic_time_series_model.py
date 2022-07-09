from helper import *
import statsmodels.tsa.stattools as ts
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm


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
class ARIMA:
    def __init__(self, train_data, test_data):
        self.trian_data = train_data
        self.test_data = test_data

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

    def select_best_model(self):
        if self._get_stationary(self.trian_data):
            if not self._white_noise(self.trian_data):
                p, q = sm.tsa.arma_order_select_ic(self.trian_data, max_ar=8, max_ma=8, ic='bic')['bic_min_order']
                return p, q
        return "not stationary or WN"

