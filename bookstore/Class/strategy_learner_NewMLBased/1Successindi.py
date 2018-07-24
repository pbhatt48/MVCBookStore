import pandas as pd
import numpy as np
from scipy import stats
import datetime as dt
from util import get_data, plot_data

def normalize_date(df):
    return df/df.ix[0,:]

def get_prices(start_date, end_date, syms):
    #print("we are in get prices function!!")
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data([syms], dates)  # automatically adds SPY
    print(prices_all)
    for sym in [syms]:
        prices_all[sym].fillna(method='ffill', inplace=True)
        prices_all[sym].fillna(method='backfill', inplace=True)

    # print("PRICES ALL BEGINNING AFTER BACKFILL")
    # print(prices_all)
    prices_all = prices_all.drop(['SPY'], axis=1)
    #print(prices_all)
    return prices_all
pricetest = get_prices(start_date=dt.datetime(2008, 1, 1), end_date=dt.datetime(2009, 12, 31), syms='AAPL')

def get_price_to_sma(df, symbol, lookback):
    #print("We are in get price to SMA!!")
    symbol = [symbol]
    normalize_price = normalize_date(df)
    normalize_price.index.name = "Date"
    rolling_mean = normalize_price[symbol].rolling(window=lookback).mean()
    rolling_mean.fillna(method='ffill', inplace=True)
    rolling_mean.fillna(method='backfill', inplace=True)
    #print("ROlling mean type == ", type(rolling_mean))
    sma = normalize_price / rolling_mean
    print("sma type == ", type(sma))
    print(sma)
    return sma
get_price_to_sma(pricetest, 'AAPL', lookback =10)

def get_bollinger(df, symbol, lookback):
    #print("You are in get Bollinger function!!")
    symbol = [symbol]
    normalize_price = normalize_date(df)
    normalize_price.index.name = "Date"
    rolling_mean = normalize_price[symbol].rolling(window=lookback).mean()
    rolling_mean.fillna(method='ffill', inplace=True)
    rolling_mean.fillna(method='backfill', inplace=True)
    rolling_std = normalize_price[symbol].rolling(window=lookback, min_periods=lookback).std()
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)
    bollinger_indicator = (normalize_price - lower_band)/(upper_band - lower_band)
    bollinger_indicator.fillna(method='ffill', inplace=True)
    bollinger_indicator.fillna(method='backfill', inplace=True)
    print("bollinger_indicator type == ", type(bollinger_indicator))
    print(bollinger_indicator)
    return bollinger_indicator

get_bollinger(pricetest, 'AAPL', lookback=10)

def get_momentum(df, window=5):
    #print("You are in get MOmemntum function!!")
    normalize_price = normalize_date(df)
    normalize_price.index.name = "Date"
    normalize_price[window:] = normalize_price[window:] / normalize_price.values[:-window]
    df[window:] = df[window:] / df.values[:-window] - 1
    #print(df)
    return df

#get_momentum(pricetest, window=10)

def n_day_return(df,symbol, nday):
    print("N DAY RETURN!")
    #daily_ret=((prices[symbol].shift(-1*Nday))/prices[symbol])-1
    nday_return = ((df[symbol].shift(-1 * nday)) / df[symbol]) - 1
    nday_return.fillna(method='ffill', inplace=True)
    print(nday_return)
    return nday_return
n_day_return(pricetest,'AAPL', 3)





if __name__ == "__main__":
    # build_orders()
    # price,price_all,price_SPY = compute_prices()
    # build_orders()

    print 'good job'






