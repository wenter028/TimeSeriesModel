from helper import *

##get the data processing from original data
class DataProcessing:
    TIME_INTERVAL: int = 5
    VALIDATE: int = 10

    def __init__(self,
                 all_dates: list,
                 time_interval:int = TIME_INTERVAL,
                 validate: int = VALIDATE):

        self.time_interval = time_interval
        self.validate = validate

        self.all_dates = all_dates

    def _data_processing_for_arima(self):
        train_data = pd.DataFrame(columns=['Time'])
        test_data = pd.DataFrame(columns=['Time'])

        train_date = self.all_dates[:-int(len(self.all_dates) / self.validate)]
        test_date = self.all_dates[-int(len(self.all_dates) / self.validate):]

        dl = DataLoader()
        data_columns = ['wpr_price_etf','wpr_price_future','basis']
        for i in train_date:
            data = dl.get_second_data(i, 'three').loc[:,data_columns]
            data.reset_index(inplace=True)
            data.Time = data.Time.apply(lambda x: time(x.hour, x.minute, int(np.floor(x.second / 3)) * 3))
            data = data.drop_duplicates(subset='Time')

            train_data = pd.merge(train_data, data[['Time', 'basis']], on='Time', how='outer')

        train_data.sort_values(by='Time', inplace=True)
        train_data.fillna(method='ffill', inplace=True)
        train_data.set_index('Time', inplace=True)

        for i in test_date:
            data = dl.get_second_data(i, 'three').loc[:,data_columns]
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
        train_data, test_data = self._data_processing_for_arima()

        train_data['mean_basis'] = train_data.mean(axis=1)
        train_data = train_data['mean_basis'].diff(5).iloc[np.where(np.arange(len(train_data)) % 5 == 0)].dropna()

        test_data = test_data.diff(5).iloc[np.where(np.arange(len(test_data)) % 5 == 0)].dropna()

        return train_data, test_data


    def data_for_lstm(self):
        pass



