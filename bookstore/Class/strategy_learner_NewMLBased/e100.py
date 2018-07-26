import marketsimcode
import datetime as dt
import pandas as pd
import util as ut
import random


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

def author():
    return("pbhatta3")

def get_benchmark(symbol):
    start_date = dt.date(2008,01,01)
    end_date = dt.date(2009,12,31)
    # benchmark_orders = [{'Date': start_date, 'Symbol': symbol, 'Order': 'BUY', 'Shares': 1000},
    #          {'Date': end_date, 'Symbol': symbol, 'Order': 'SELL', 'Shares': 1000},]
    # benchmark_orders_df = pd.DataFrame(benchmark_orders)
    benchmark_orders = [[start_date,symbol,'BUY', 1000],[start_date,symbol,'SELL', 1000]]
    benchmark_orders_df = pd.DataFrame(benchmark_orders, columns=['Date', 'Symbol', 'Order', 'Shares'])
    benchmark_orders_df.index.name = "Date"
    #benchmark_orders_df['Date'] = pd.DatetimeIndex(benchmark_orders_df.Date).normalize()
    benchmark_orders_df.reset_index(drop=True, inplace=True)
    print(benchmark_orders_df)
    sv = 1000000
    portvals = marketsimcode.compute_portvals(orders_df = benchmark_orders_df, start_val = sv)

get_benchmark('JPM')

def create_df_benchmark(symbol, start_date, end_date, num_shares):
    """Create a dataframe of benchmark data. Benchmark is a portfolio consisting of
    num_shares of the symbol in use and holding them until end_date.
    """
    # Get adjusted close price data
    benchmark_prices = ut.get_data([symbol], pd.date_range(start_date, end_date),
                                addSPY=False).dropna()
    # Create benchmark df: buy num_shares and hold them till the last date
    df_benchmark_trades = pd.DataFrame(
        data=[(benchmark_prices.index.min(), num_shares),
        (benchmark_prices.index.max(), -num_shares)],
        columns=["Date", "Shares"])
    df_benchmark_trades.set_index("Date", inplace=True)
    print(df_benchmark_trades)
    return df_benchmark_trades

start_date = dt.datetime(2010, 1, 1)
end_date = dt.datetime(2011, 12, 31)
#create_df_benchmark("JPM",start_date, end_date, 1000 )
