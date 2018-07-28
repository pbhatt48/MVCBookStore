import marketsimcode
import datetime as dt
import pandas as pd
import util as ut
import random
import StrategyLearner
import matplotlib.pyplot as plt


"""

Benchmark: The performance of a portfolio starting with $100,000 cash,
investing in 1000 shares of the symbol in use and holding that position.
Include transaction costs.

Benchmark: Buy 1000 shares on the first trading day, Sell 1000 shares on the last day.

Experiment 1: Using the benchmark described above, plot the performance of the benchmark versus your strategy in sample. Trade only the symbol JPM for this evaluation. The code that implements this experiment and generates the relevant charts and data should be submitted as experiment1.py
Describe your experiment in detail: Assumptions, parameter values and so on.
Describe the outcome of your experiment.
Would you expect this relative result every time with in-sample data? Explain why or why not.
Training / in sample: January 1, 2008 to December 31 2009

"""

# please use getplot value to True to get the plotting diagram
gen_plot = True
def author():
    return("pbhatta3")

def get_benchmark(symbol):
    start_date = dt.date(2008,01,01)
    end_date = dt.date(2009,12,31)
    dates = pd.date_range(start_date, end_date)
    prices_all = ut.get_data([symbol], dates)
    print("PRICES == ", prices_all)
    print("Here index", prices_all.index[0], "and", prices_all.index[-1])
    trading_start_date = prices_all.index[0]
    trading_end_date = prices_all.index[-1]
    # benchmark_orders = [{'Date': start_date, 'Symbol': symbol, 'Order': 'BUY', 'Shares': 1000},
    #          {'Date': end_date, 'Symbol': symbol, 'Order': 'SELL', 'Shares': 1000},]
    # benchmark_orders_df = pd.DataFrame(benchmark_orders)
    benchmark_orders = [[trading_start_date,symbol,'BUY', 1000],[trading_end_date,symbol,'SELL', 1000]]
    benchmark_orders_df = pd.DataFrame(benchmark_orders, columns=['Date', 'Symbol', 'Order', 'Shares'])
    benchmark_orders_df.index.name = "Date"
    #benchmark_orders_df['Date'] = pd.DatetimeIndex(benchmark_orders_df.Date).normalize()
    benchmark_orders_df.reset_index(drop=True, inplace=True)
    print(benchmark_orders_df)
    sv = 100000
    portvals = marketsimcode.compute_portvals(orders_df = benchmark_orders_df, start_val = sv)
    return portvals

#get_benchmark('JPM')

def my_strategy(symbol):

    x= StrategyLearner.StrategyLearner(verbose = False, impact=0.0)
    #x.build_RL_data(sd = dt.datetime(2010,1,1),ed = dt.datetime(2011,12,31))
    #x.get_features_data(sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 12, 31), syms = ['AAPL'], lookback = 21)
    symbs=symbol
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
    sv = 100000
    portvals = marketsimcode.compute_portvals(orders_df=stlearner_orders_df, start_val=sv)
    return portvals



def create_graph():
    sv = 100000
    benchmark = get_benchmark('JPM')
    stlearner_port = my_strategy('JPM')
    print("BENCHMARK PORTFOLI VAL == ", benchmark.shape, benchmark)
    print("MY STRATEGY PORTFOLI VAL == ", stlearner_port.shape, stlearner_port)

    #combined_df = benchmark.join(stlearner_port, on=benchmark.index)
    print("Combined DF", type(benchmark), benchmark.index, type(stlearner_port),stlearner_port.index )
    combined_df = pd.concat((benchmark, stlearner_port), axis=1)
    # indicators_df = pd.concat((mom, ptsma10, bollinger10, buy_sell_decision), axis=1)
    # indicators_df.columns = ["MOM", "PTSMA", "BOLLINGER", "DailyReturns"]
    combined_df.columns = ["BENCHMARK", "STLEARNER PORT"]
    combined_df['STLEARNER PORT'].fillna(method='ffill', inplace=True)
    combined_df['STLEARNER PORT'].fillna(method='bfill', inplace=True)
    print("INDEX OF combined == ", combined_df.index)
    #combined_df.fillna(sv, inplace=True)
    print(combined_df)



    # plot Impact Vs Portfolio

    x_axis = combined_df.index
    print("INDEX OF combined == ", x_axis[0], x_axis[-1])
    y_axis = combined_df['BENCHMARK']
    y2_axis = combined_df['STLEARNER PORT']
    plt.plot(x_axis, y_axis, label="BENCHMARK PORTFOLIO VALUE")
    plt.plot(x_axis, y2_axis, label="STLEARNER PORTFOLIO VALUE")
    plt.xlabel("DATE")
    plt.ylabel("PORTFOLIO VALUE")
    print("X axis value = ", x_axis)
    plt.title('BENCHMARK VS STLEARNER')
    plt.legend(loc='lower right')
    plt.xlim(x_axis[0]-dt.timedelta(1), x_axis[-1]+dt.timedelta(1))
    plt.show()


    # fig2, ax2 = plt.subplots()
    # print("LIMITTT == ", benchmark.index.min)
    # #ax2.set_xlim(benchmark.index)
    #
    # ax2.plot(benchmark.index, combined_df['BENCHMARK'], label="BENCHMARK PORTFOLIO VALUE")
    # ax2.plot(benchmark.index, combined_df['STLEARNER PORT'], label="STRATEGY LEARNER PORT VALUE")
    #
    # plt.title('BENCHMARK VS STLEARNER')
    # plt.legend(loc='lower right')
    # ax2.set_xlabel("Date")
    # ax2.set_ylabel("PORTFOLIO VALUE")
    # fig2.text(0.95, 0.05, 'Prakash Graph',
    #          fontsize=50, color='gray',
    #          ha='right', va='bottom', alpha=0.5)
    #
    # plt.show()

if gen_plot == True:
    create_graph()
else :
    print("PLEASE CHANGE THE gen_plot variable to True to generate plots")
