#
# Python Module with Class
# for Vectorized Backtesting
# of EMA-based Strategies
# Adapted from:
# Python for Algorithmic Trading
# Yves J. Hilpisch
#

import numpy as np
import pandas as pd
from scipy.optimize import brute
import yfinance as yf

class EMA(object):
    ''' Class for the vectorized backtesting of EMA-based trading strategies.

    Attributes
    ==========
    symbol: str
        RIC symbol with which to work with
    EMA1: int
        time window in days for shorter EMA
    EMA2: int
        time window in days for longer EMA
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval

    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    set_parameters:
        sets one or two new EMA parameters
    run_strategy:
        runs the backtest for the EMA-based strategy
    plot_results:
        plots the performance of the strategy compared to the symbol
    update_and_run:
        updates EMA parameters and returns the (negative) absolute performance
    optimize_parameters:
        implements a brute force optimizeation for the two EMA parameters
    '''
   
    def __init__(self, symbol, EMA1, EMA2, start, end):
        self.symbol = symbol
        self.EMA1 = EMA1
        self.EMA2 = EMA2
        self.start = start
        self.end = end
        self.results = None
        self.get_data()

    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        mu = yf.Ticker(self.symbol)
        raw = mu.history(start=self.start, end=self.end).dropna()
        raw = pd.DataFrame(raw['Close'])
        raw.rename(columns={'Close': 'price'}, inplace=True)
        raw['return'] = np.log(raw / raw.shift(1))
        raw['EMA1'] = raw['price'].ewm(span = self.EMA1, adjust = False).mean()
        raw['EMA2'] = raw['price'].ewm(span = self.EMA2, adjust = False).mean()
        self.data = raw

    def set_parameters(self, EMA1=None, EMA2=None):
        ''' Updates EMA parameters and resp. time series.
        '''
        if EMA1 is not None:
            self.EMA1 = EMA1
            self.data['EMA1'] = self.data['price'].ewm(span = self.EMA1, adjust = False).mean()
        if EMA2 is not None:
            self.EMA2 = EMA2
            self.data['EMA2'] = self.data['price'].ewm(span = self.EMA2, adjust = False).mean()

    def run_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()
        data['position'] = np.where(data['EMA1'] > data['EMA2'], 1, 0)
        data['strategy'] = data['position'].shift(1) * data['return']
        data.dropna(inplace=True)
        data['creturns'] = data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        self.results = data
        # gross performance of the strategy
        aperf = data['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - data['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)

    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = '%s | EMA1=%d, EMA2=%d' % (self.symbol,
                                               self.EMA1, self.EMA2)
        self.results[['creturns', 'cstrategy']].plot(title=title,
                                                     figsize=(10, 6))

    def update_and_run(self, EMA):
        ''' Updates EMA parameters and returns negative absolute performance
        (for minimazation algorithm).

        Parameters
        ==========
        EMA: tuple
            EMA parameter tuple
        '''
        self.set_parameters(int(EMA[0]), int(EMA[1]))
        return -self.run_strategy()[0]

    def optimize_parameters(self, EMA1_range, EMA2_range):
        ''' Finds global maximum given the EMA parameter ranges.

        Parameters
        ==========
        EMA1_range, EMA2_range: tuple
            tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (EMA1_range, EMA2_range), finish=None)
        return opt, -self.update_and_run(opt)