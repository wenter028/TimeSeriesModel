# This Code is for the concat future to get the major contract

import os

from helper import *

__all__ = ['get_date_list',
           'choose_open_interest',
           'get_etf']


#get the data I have
def get_date_list(etf,now_fu,next_fu):
    etf_list = os.listdir(etf)
    now_future_list = os.listdir(now_fu)
    next_future_list = os.listdir(next_fu)

    return set(etf_list).intersection(set(now_future_list).union(next_future_list))

#get major contract
def choose_open_interest(date, now_fu, next_fu,PATH):

    '''
    now_fu:当月
    next_fu:下月
    '''

    if date in os.listdir(now_fu) and date in os.listdir(next_fu):
        now_data = load(now_fu + '/' + date)
        next_data = load(next_fu + '/' + date)

        if now_data.iloc[-1].OpenInterest > next_data.iloc[-1].OpenInterest:
            return save_gzip(now_data,(PATH+'/'+date))
        else:
            return save_gzip(next_data,(PATH+'/'+date))

    else:
        if date in os.listdir(now_fu):
            now_data = load(now_fu + '/' + date)
            return save_gzip(now_data,(PATH+'/'+date))
        else:
            next_data = load(next_fu + '/' + date)
            return save_gzip(next_data,(PATH+'/'+date))
#
def get_etf(date,PATH):
    data = load(ETF_PATH+'/'+date)
    return save_gzip(data, (PATH+'/'+date))


if __name__ == '__main__':
    os.makedirs(PROJ_PATH+'/'+'major contract',exist_ok=True)
    os.makedirs(PROJ_PATH + '/' + 'ETF', exist_ok=True)

    date_list = get_date_list(ETF_PATH,FUTURE_PATH_NOW,FUTURE_PATH_NEXT)

    parLapply(date_list,choose_open_interest, now_fu=FUTURE_PATH_NOW, next_fu=FUTURE_PATH_NEXT,PATH=(PROJ_PATH+'/'+'major contract'))

    parLapply(date_list, get_etf,PATH=(PROJ_PATH + '/' + 'ETF'))