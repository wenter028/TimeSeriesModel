from helper import *
import statsmodels.tsa.stattools as ts
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm

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

