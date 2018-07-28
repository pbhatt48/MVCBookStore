import marketsimcode
import datetime as dt
import pandas as pd
import util as ut
import random
import StrategyLearner
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


"""

Experiment 2: Provide an hypothesis regarding how changing the value of impact should affect in sample trading behavior and results (provide at least two metrics). Conduct an experiment with JPM on the in sample period to test that hypothesis. Provide charts, graphs or tables that illustrate the results of your experiment. The code that implements this experiment and generates the relevant charts and data should be submitted as experiment2.py
Training / in sample: January 1, 2008 to December 31 2009

"""

def author():
    return("pbhatta3")

# please use getplot value to True to get the plotting diagram
gen_plot = True

def my_strategy_vs_impact(impact_val):

    x= StrategyLearner.StrategyLearner(verbose = False, impact=impact_val)
    #x.build_RL_data(sd = dt.datetime(2010,1,1),ed = dt.datetime(2011,12,31))
    #x.get_features_data(sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 12, 31), syms = ['AAPL'], lookback = 21)
    symbs='JPM'
    x.addEvidence(symbol = symbs, sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 12, 31),  sv=10000)
    strat_trades = x.testPolicy(symbol=symbs, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=10000)

    print("My trades from STLearner== ", strat_trades)

    order_list= []
    strat_trades.index.name = "Date"
    for index, row in strat_trades.iterrows():
        #print(trades_df.index, row)
        print("STRAT Index == ", index.strftime('%Y-%m-%d'))
        #print("STRAT ROw == ", row[0])
        #print(strat_trades.loc[row['Date'], row['Trades']])
        #loc[row['Date'], row['Symbol']]
        if row[0] == 1000 or  row[0] == 2000:
            print("Trading Happened")
            print(row[0])
            #[trading_start_date,symbol,'BUY', 1000]
            order_list.append([index.strftime('%Y-%m-%d'),'JPM','BUY',row[0]])
        elif row[0] == -1000 or row[0] == -2000:
            order_list.append([index.strftime('%Y-%m-%d'), 'JPM', 'SELL', -1*row[0]])

    print("ORDER LIST == ", order_list)
    stlearner_orders_df = pd.DataFrame(order_list, columns=['Date', 'Symbol', 'Order', 'Shares'])
    print(stlearner_orders_df)
    sv = 1000000
    portvals = marketsimcode.compute_portvals(orders_df=stlearner_orders_df, start_val=sv)

    return (portvals.ix[-1][0]), portvals

#my_strategy_vs_impact()

def generate_graphs():

    #impacts = np.arange(0.0, 0.12, 0.02)
    impacts = [0.005, 0.01, 0.02, 0.1, 0.12]
    print(impacts)
    final_port_values= []
    port_df = pd.DataFrame()
    fv = []
    #get DF at impact == 0 :
    port_end_val, port_val_df = my_strategy_vs_impact(0)
    print("FIRST PORT VAL ==", type(port_val_df),port_val_df)

    for i in impacts:
        print(i)
        port_val_df.index.name = "Date"
        port_end_val, port_val = my_strategy_vs_impact(i)
        final_port_values.append(port_end_val)
        print("PORT 0000 VALUES == ")
        # #data = pd.DataFrame({i: port_val})
        print(port_val)
        port_val_df[i] = port_val
        # #fv.append(port_val.values.tolist())
        # #print("FV === ", fv)
    print(final_port_values)
    print("FINAL PORT DF BEFORE== ")
    print(port_val_df)
    port_val_df.fillna(method='ffill', inplace=True)
    port_val_df.fillna(method='bfill', inplace=True)
    print("AFTER")
    print(port_val_df)


    x_axis = port_val_df.index
    print("INDEX OF combined == ", x_axis[0], x_axis[-1])
    y_axis = port_val_df['VALUE']
    y2_axis = port_val_df[0.005]
    y3_axis = port_val_df[0.01]
    y4_axis = port_val_df[0.02]
    y5_axis = port_val_df[0.1]
    y6_axis = port_val_df[0.12]

    plt.plot(x_axis, y_axis, label="IMPACT @ 0.0")
    plt.plot(x_axis, y2_axis, label="IMPACT @ 0.005")
    plt.plot(x_axis, y3_axis, label="IMPACT @ 0.01")
    plt.plot(x_axis, y4_axis, label="IMPACT @ 0.02")
    plt.plot(x_axis, y5_axis, label="IMPACT @ 0.1")
    plt.plot(x_axis, y6_axis, label="IMPACT @ 0.12")
    plt.xlabel("DATE")
    plt.ylabel("PORTFOLIO VALUE")
    print("X axis value = ", x_axis)
    plt.title('IMPACT VS PORTFOLIO')
    plt.legend(loc='lower right')
    plt.xlim(x_axis[0], x_axis[-1])
    plt.show()



if gen_plot == True:
    generate_graphs()
else :
    print("PLEASE CHANGE THE gen_plot variable to True to generate plots")
