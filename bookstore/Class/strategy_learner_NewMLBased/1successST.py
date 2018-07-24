"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import pandas as pd
import util as ut
import random
from indicators_submission import *
import RTLearner as rtl
import BagLearner as bl


# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)


class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact

    def author(self):
        return 'pbhatta3'

    def get_features_data(self, sd, ed, syms, lookback):
        #print("GET FEATURES!!")
        delta_day = 20
        price = get_prices(sd - dt.timedelta(delta_day), ed, syms)
        mom = get_momentum(price, window=5)[sd:]
        print("GET FEATURES MOM == ", mom.shape)
        print(mom)

        #(df, symbol, lookback)
        ptsma10 = get_price_to_sma(price, syms, lookback=10)[sd:]
        print("GET FEATURES ptmsa10 == ", ptsma10.shape)
        print(ptsma10)

        #get_bollinger(df, symbol, lookback):
        bollinger10 = get_bollinger(price, syms, lookback=10)[sd:]
        print("GET FEATURES bollinger10 == ", bollinger10.shape)
        print(bollinger10)

        #n_day_return(df,symbol, nday):
        nday_return3 = n_day_return(price, syms, nday=3)
        print("GET FEATURES nday_return3 == ", nday_return3.shape)
        print(nday_return3)

        #buysell decisions
        buy_sell_decision = nday_return3.copy()
        #print("GET FEATURES buy_sell_decision == ", type(buy_sell_decision))
        # for i in range(0, nday_return3.shape[0]):
        #     if nday_return3.iloc[i,0] >= 3.0:
        #         buy_sell_decision[i:] = 1
        #     elif nday_return3.iloc[i,0] < -3.0:
        #         buy_sell_decision[i:] = -1
        #     else:
        #         buy_sell_decision.iloc[i:] = 0

        #print(buy_sell_decision)




        #create a train X train Y dataframe out of the indicators
        indicators_df = pd.concat((mom, ptsma10, bollinger10, nday_return3), axis=1)
        #indicators_df = pd.concat((mom, ptsma10, bollinger10, buy_sell_decision), axis=1)
        indicators_df.columns = ["MOM", "PTSMA", "BOLLINGER", "DailyReturns"]
        #indicators_df.columns = ["MOM", "PTSMA", "BOLLINGER", "BuySell"]
        #print("Main indicators")
        #indicators_df= indicators_df.dropna()
        #print(type(indicators_df))
        #print(indicators_df.tail(10))
        return indicators_df[sd:]


    #get_features_data(self, sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 12, 31), syms = ['AAPL'], lookback = 21)

        # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol="IBM", \
                        sd=dt.datetime(2008, 1, 1), \
                        ed=dt.datetime(2009, 1, 1), \
                        sv=10000):
        train_data = self.get_features_data(sd, ed, symbol, lookback=10)
        #print("TRAINING DATA BEFORE")
        #print(train_data)
        trainX = train_data.values[:, 0:-1]
        trainY = train_data.values[:, -1]
        #print("Train X \n", trainX)
        #print("Train Y \n", trainY)
        self.bag_learner = bl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size": 5}, bags=20, boost=False,
                                         verbose=False)
        self.bag_learner.addEvidence(trainX, trainY)



    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):
        train_data = self.get_features_data(sd, ed, symbol, lookback=10)
        #print("TRAINING DATA in TEST POLICY")
        #print(train_data)
        trainX = train_data.values[:, 0:-1]
        trainY = train_data.values[:, -1]

        predictYValue = self.bag_learner.query(trainX)
        #self.bag_learner.query(trainX)
        predictYValue = pd.DataFrame(predictYValue)
        #print("PREDICTION Y VALUE == ", type(predictYValue))
        #print(predictYValue)
        train_data.index.name = "Date"


        newData = train_data.copy()
        #print("NEW DATA ", predictYValue.loc[:,:].values)
        #print("TESTTEST")
        newData['DailyReturns'] = predictYValue.loc[:,:].values
        #print(newData)
        #newData = pd.DataFrame(newData['DailyReturns'])

        #creating trades and holdings df
        # trades_df = newData['DailyReturns'].copy() *0
        # holdings_df = newData['DailyReturns'].copy() *0
        # print(trades_df, holdings_df)

        newData["BSC"] = 0
        newData['Trades'] = 0
        newData['Holdings'] = 0

        #print("BEFORE")
        #print(newData)
        preserved_index = newData.index
        #print("THE INDEX == ", preserved_index)
        #newData.index.name = "Date"
        newData.reset_index(level=["Date"], inplace=True)
        #print("THE INDEX == ", newData.index)
        #print(type(newData))

        for i in newData.index[1:]:
            if newData.get_value(i, 'DailyReturns') >= 1.0 and newData.get_value(i - 1, 'Holdings') == 0:
                newData.set_value(i, "Trades", 1000)
                newData.set_value(i, "BSC", 1)
                newData.set_value(i, 'Holdings', newData.get_value(i - 1, 'Holdings') + newData.get_value(i, 'Trades'))
            elif newData.get_value(i, 'DailyReturns') >= 1.0 and newData.get_value(i - 1, 'Holdings') == 1000:
                newData.set_value(i, "Trades", 0)
                newData.set_value(i, "BSC", 1)
                newData.set_value(i, 'Holdings', newData.get_value(i - 1, 'Holdings') + newData.get_value(i, 'Trades'))
            elif newData.get_value(i, 'DailyReturns') < -2.0 and newData.get_value(i - 1, 'Holdings') == 1000:
                newData.set_value(i, "Trades", -2000)
                newData.set_value(i, 'Holdings', newData.get_value(i - 1, 'Holdings') + newData.get_value(i, 'Trades'))
                newData.set_value(i, "BSC", -1)
            elif newData.get_value(i, 'DailyReturns') >= 1.0 and newData.get_value(i - 1, 'Holdings') == -1000:
                newData.set_value(i, "Trades", 2000)
                newData.set_value(i, 'Holdings', newData.get_value(i - 1, 'Holdings') + newData.get_value(i, 'Trades'))
                newData.set_value(i, "BSC", 1)
            else:
                newData.set_value(i, 'Holdings', newData.get_value(i - 1, 'Holdings'))
                newData.set_value(i, "BSC", 0)

        print(newData)

        final_trades = pd.DataFrame(newData['Trades'])
        #print("PRESERVED INDEX ++ ", preserved_index)
        final_trades.index = preserved_index
        print("Type of final trades == ", type(final_trades))
        print(final_trades)
        return final_trades

def test_code():
    x= StrategyLearner(verbose = False, impact=0.0)
    #x.build_RL_data(sd = dt.datetime(2010,1,1),ed = dt.datetime(2011,12,31))
    #x.get_features_data(sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 12, 31), syms = ['AAPL'], lookback = 21)
    symbs='AAPL'
    x.addEvidence(symbol = symbs, sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 12, 31),  sv=10000)
    x.testPolicy(symbol=symbs, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=10000)
if __name__=="__main__":
    print "One does not simply think up a strategy"
    test_code()
