"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import pandas as pd
import util as ut
import random
from indicators import *
from best_strategy import build_benchmark
import RTLearner as rtl
import BagLearner as bl


class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact

    def get_features_data(self, sd, ed, syms, lookback):
        print("GET FEATURES!!")
        delta_day = 20
        price = get_prices(sd - dt.timedelta(delta_day), ed, syms)
        mom = get_momentum(price, window=5)[sd:]
        print("GET FEATURES MOM == ")
        print(mom)

        #(df, symbol, lookback)
        ptsma10 = get_price_to_sma(price, syms, lookback=10)[sd:]
        print("GET FEATURES ptmsa10 == ")
        print(ptsma10)

        #get_bollinger(df, symbol, lookback):
        bollinger10 = get_bollinger(price, syms, lookback=10)[sd:]
        print("GET FEATURES bollinger10 == ")
        print(bollinger10)

        #n_day_return(df,symbol, nday):
        nday_return3 = n_day_return(price, syms, nday=3)
        print("GET FEATURES nday_return3 == ")
        print(nday_return3)

        #buysell decisions
        buy_sell_decision = nday_return3.copy()
        print("GET FEATURES buy_sell_decision == ", type(buy_sell_decision))
        for i in range(0, nday_return3.shape[0]):
            if nday_return3.iloc[i,0] >= 3.0:
                buy_sell_decision[i:] = 1
            elif nday_return3.iloc[i,0] < -3.0:
                buy_sell_decision[i:] = -1
            else:
                buy_sell_decision.iloc[i:] = 0

        print(buy_sell_decision)




        #create a train X train Y dataframe out of the indicators
        indicators_df = pd.concat((mom, ptsma10, bollinger10, nday_return3), axis=1)
        indicators_df.columns = ["MOM", "PTSMA", "BOLLINGER", "DailyReturns"]
        print("Main indicators")
        #indicators_df= indicators_df.dropna()
        print(type(indicators_df))
        print(indicators_df.tail(10))
        return indicators_df[sd:]


    #get_features_data(self, sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 12, 31), syms = ['AAPL'], lookback = 21)

        # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol="IBM", \
                        sd=dt.datetime(2008, 1, 1), \
                        ed=dt.datetime(2009, 1, 1), \
                        sv=10000):
        train_data = self.get_features_data(sd, ed, symbol, lookback=10)
        print("TRAINING DATA BEFORE")
        print(train_data)
        trainX = train_data.values[:, 0:-1]
        trainY = train_data.values[:, -1]
        print("Train X \n", trainX)
        print("Train Y \n", trainY)
        self.bag_learner = bl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size": 5}, bags=20, boost=False,
                                         verbose=False)
        self.bag_learner.addEvidence(trainX, trainY)



            # add your code to do learning here


            # # example usage of the old backward compatible util function
            # syms = [symbol]
            # dates = pd.date_range(sd, ed)
            # prices_all = ut.get_data(syms, dates)  # automatically adds SPY
            # prices = prices_all[syms]  # only portfolio symbols
            # prices_SPY = prices_all['SPY']  # only SPY, for comparison later
            # if self.verbose:
            #     print prices
            #
            # # example use with new colname
            # volume_all = ut.get_data(syms, dates, colname="Volume")  # automatically adds SPY
            # volume = volume_all[syms]  # only portfolio symbols
            # volume_SPY = volume_all['SPY']  # only SPY, for comparison later
            # if self.verbose:
            #     print volume

    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):
        train_data = self.get_features_data(sd, ed, symbol, lookback=10)
        print("TRAINING DATA in TEST POLICY")
        print(train_data)
        trainX = train_data.values[:, 0:-1]
        trainY = train_data.values[:, -1]

        predictYValue = self.bag_learner.query(trainX)
        #self.bag_learner.query(trainX)
        predictYValue = pd.DataFrame(predictYValue)
        print("PREDICTION Y VALUE == ", type(predictYValue))
        print(predictYValue)
        train_data.index.name = "Date"


        newData = train_data.copy()
        print("NEW DATA ", predictYValue.loc[:,:].values)
        print("TESTTEST")
        newData['DailyReturns'] = predictYValue.loc[:,:].values
        print(newData)
        #newData = pd.DataFrame(newData['DailyReturns'])

        #creating trades and holdings df
        # trades_df = newData['DailyReturns'].copy() *0
        # holdings_df = newData['DailyReturns'].copy() *0
        # print(trades_df, holdings_df)

        newData['Trades'] = 0
        newData['Holdings'] = 0
        print(newData)
        #newData.index.name = "Date"
        newData.reset_index(level=["Date"], inplace=True)
        print("THE INDEX == ", newData.index)
        print(type(newData))

        for i in newData.index[1:]:
            if newData.get_value(i, 'DailyReturns') >= 1.0:
                newData.set_value(i, "Trades", 1000)
                newData.set_value(i, 'Holdings', newData.get_value(i-1, 'Holdings') + 1000)
            elif newData.get_value(i, 'DailyReturns') < -2.0 and newData.get_value(i-1, 'Holdings') >= 1000:
                newData.set_value(i, "Trades", -1000)
                newData.set_value(i, 'Holdings', newData.get_value(i - 1, 'Holdings') - 1000)
            else:
                newData.set_value(i, 'Holdings', newData.get_value(i - 1, 'Holdings') )


        # for index, row in newData.iterrows():
        #     #print(row['DailyReturns'])
        #     print(newData.at[index, 'Trades'])
        #     print(row['Trades'])
        #     if row['DailyReturns'] >= 1.0:
        #     #     #trades_df.loc[row['Date'], row['Symbol']] += row['Shares']
        #         newData.at[index, 'Trades'] = 1000
        #         newData.at[index, 'Holdings'] = newData.at[index-dt.timedelta(1), 'Holdings'] + 1000
        #
        #         #newData.set_value(index,'Trades', 1000)
        #         #newData.set_value(index, 'Holdings', 1000)
        #         #print(index)
        #         #print(row)
        #         #newData.set_value(index, 'Holdings', row['Holdings'].cumsum(axis=0))
        #     elif row['DailyReturns'] < -1.0 and row['Holdings'] >= 1000:
        #         newData.at[index, 'Trades'] = -1000
        #         #newData.set_value(index, 'Trades', - 1000)
        #     else:
        #         #newData.at[index, 'Holdings'] = newData.at[index-dt.timedelta(1), 'Holdings']
        #         pass
        #     #
        #     #     newData.loc[row['Date'], row['Trades']]= "SELL"

        print(newData)






        # here we build a fake set of trades
        # your code should return the same sort of data
        # dates = pd.date_range(sd, ed)
        # prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        # trades = prices_all[[symbol,]]  # only portfolio symbols
        # trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        # trades.values[:,:] = 0 # set them all to nothing
        # trades.values[0,:] = 1000 # add a BUY at the start
        # trades.values[40,:] = -1000 # add a SELL
        # trades.values[41,:] = 1000 # add a BUY
        # trades.values[60,:] = -2000 # go short from long
        # trades.values[61,:] = 2000 # go long from short
        # trades.values[-1,:] = -1000 #exit on the last day
        # if self.verbose: print type(trades) # it better be a DataFrame!
        # if self.verbose: print trades
        # if self.verbose: print prices_all
        # return trades


    def build_RL_data( sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), syms=['AAPL'], lookback=21):
        delta_day = 20
        price = get_prices(sd-dt.timedelta(delta_day), ed,syms)
        # print price

        sma = get_price_to_sma(price, syms, lookback=10)
        print("SMA == \n", sma.head(20))
        bbp = get_bollinger(price, syms, lookback=10)
        print("bbp == \n", bbp.head(20))
        rsi = rsi_indicator(price)
        stoch, stochd = stochastic_indicator(price)

        # print bbp
        # rsi_SPY = rsi_indicator(price_SPY)

        # Orders starts as a NaN array of the same shape/index as price
        Ydata = price.copy()
        Ydata.values[:-lookback, :] = price.values[lookback:, :] / price.values[:-lookback, :] - 1
        Ydata.values[-lookback:, :] = np.NaN
        Ydata = Ydata.fillna(method='ffill')

        YBUY = 0.01
        YSELL = -0.01

        for i in range(Ydata.shape[0]):
            if Ydata.values[i, :] > YBUY:
                Ydata.values[i, :] = 1.0
            elif Ydata.values[i, :] < YSELL:
                Ydata.values[i, :] = -1.0
            else:
                Ydata.values[i, :] = 0.0

        # print Ydata
        # k = Ydata[Ydata < 0]
        # print k
        sma_sd = standardization_indicator(sma)
        bbp_sd = standardization_indicator(bbp)
        rsi_sd = standardization_indicator(rsi)
        stoch_sd = standardization_indicator(stoch)

        sma_sd = sma_sd.rename(columns={'AAPL': 'sma'})
        bbp_sd = bbp_sd.rename(columns={'AAPL': 'bbp'})
        rsi_sd = rsi_sd.rename(columns={'AAPL': 'rsi'})
        stoch_sd = stoch_sd.rename(columns={'AAPL': 'stoch'})
        Ydata = Ydata.rename(columns={'AAPL': 'Ydata'})

        train_data = pd.concat([sma_sd, bbp_sd, rsi_sd, stoch_sd, Ydata], axis=1)

        print("TRAIN DATA == \n")
        train_data = train_data.drop(['stoch'], axis=1)
        print(train_data)

        return train_data
    build_RL_data(sd=dt.datetime(2012, 1, 1), ed=dt.datetime(2012, 12, 31), syms=['AAPL'], lookback=10)

    # this method should create a QLearner, and train it for trading
    def addEvidence2(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):
        train_data = self.build_RL_data(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), syms=['AAPL'], lookback=21)
        print("TRAIN DATA22 == \n")
        print(train_data)
        test_data = self. build_RL_data(sd = dt.datetime(2010,1,1),ed = dt.datetime(2011,12,31))

        dataX = train_data.values[:, 0:-1]
        dataY = train_data.values[:, -1]
        testX = test_data.values[:, 0:-1]
        testY = test_data.values[:, -1]
        self.bag_learner = bl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size": 5}, bags=20, boost=False,
                                         verbose=False)

        self.bag_learner.addEvidence(dataX, dataY)

        # # add your code to do learning here
        #
        # # example usage of the old backward compatible util function
        # syms=[symbol]
        # dates = pd.date_range(sd, ed)
        # prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        # prices = prices_all[syms]  # only portfolio symbols
        # prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        # if self.verbose: print prices
        #
        # # example use with new colname
        # volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        # volume = volume_all[syms]  # only portfolio symbols
        # volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        # if self.verbose: print volume


    # this method should use the existing policy and test it against new data
    def testPolicy2(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        trades.values[:,:] = 0 # set them all to nothing
        trades.values[0,:] = 1000 # add a BUY at the start
        trades.values[40,:] = -1000 # add a SELL 
        trades.values[41,:] = 1000 # add a BUY 
        trades.values[60,:] = -2000 # go short from long
        trades.values[61,:] = 2000 # go long from short
        trades.values[-1,:] = -1000 #exit on the last day
        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        return trades

def test_code():
    x= StrategyLearner(verbose = False, impact=0.0)
    #x.build_RL_data(sd = dt.datetime(2010,1,1),ed = dt.datetime(2011,12,31))
    #x.get_features_data(sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 12, 31), syms = ['AAPL'], lookback = 21)
    x.addEvidence(symbol = ['AAPL'], sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 12, 31),  sv=10000)
    x.testPolicy(symbol=['AAPL'], sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=10000)
if __name__=="__main__":
    print "One does not simply think up a strategy"
    test_code()
