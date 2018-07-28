import pandas as pd
import numpy as np
from scipy import stats
import datetime as dt
from util import get_data, plot_data
import matplotlib.pyplot as plt

def normalize_date(df):
    price_df = df.copy()
    return price_df/price_df.ix[0,:]

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
#get_price_to_sma(pricetest, 'AAPL', lookback =10)

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
    print("upper_band == ",  upper_band)
    print("lower band == " ,lower_band)
    print("bollinger_indicator", bollinger_indicator)
    return bollinger_indicator, upper_band, lower_band

#get_bollinger(pricetest, 'AAPL', lookback=10)


def get_momentum(df, window):
    print("You are in get MOmemntum function!!")
    #normalize_price = normalize_date(df)
    #normalize_price.index.name = "Date"
    #normalize_price[window:] = normalize_price[window:] / normalize_price.values[:-window]
    momentum = df.copy()
    momentum[window:] = momentum[window:] / momentum.values[:-window] - 1
    print(momentum)
    #momentum = df.copy()
    #momentum.index.name = "Date"
    #momentum.ix[window:] = df.ix[window:] / df.values[:-window] - 1

    # plot Code
    # norm_price = normalize_date(df["AAPL"])
    # fig, ax = plt.subplots()
    # ax.plot(df.index, norm_price, label="STOCK PRICE")
    # ax.plot(df.index, momentum, label="MOMENTUM")
    # plt.title('Price Vs NDay Return')
    # # plt.title('RTLearner VS LEAF SIZE ')
    # plt.legend(loc='lower right')
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Normalized Price/ Return")
    # fig.text(0.95, 0.05, 'Prakash Graph',
    #          fontsize=50, color='gray',
    #          ha='right', va='bottom', alpha=0.5)
    # plt.show()

    return momentum

#get_momentum(pricetest, window=10)

def n_day_return(df,symbol, nday):
    print("N DAY RETURN!")
    #daily_ret=((prices[symbol].shift(-1*Nday))/prices[symbol])-1
    nday_return = ((df[symbol].shift(-1 * nday)) / df[symbol]) - 1
    nday_return.fillna(method='ffill', inplace=True)
    print(nday_return)

    #plot Code
    # norm_price = normalize_date(df["AAPL"])
    # fig, ax = plt.subplots()
    # ax.plot(df.index, norm_price, label="STOCK PRICE")
    # ax.plot(df.index, nday_return, label="NDAY Return")
    # plt.title('Price Vs NDay Return')
    # # plt.title('RTLearner VS LEAF SIZE ')
    # plt.legend(loc='lower right')
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Normalized Price/ Return")
    # fig.text(0.95, 0.05, 'Prakash Graph',
    #          fontsize=50, color='gray',
    #          ha='right', va='bottom', alpha=0.5)
    # plt.show()

    return nday_return
#n_day_return(pricetest,'AAPL', 3)



