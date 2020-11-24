import os
import matplotlib.pyplot as plt
import pandas  as pd
import numpy as np
import datetime as dt
import pyEX as p
from matplotlib import style
import pandas_datareader.data as web


class Data:
    '''
    This class is responsible for loading the data for the VIX and Stocks
    '''

    def __init__(self, ticker: str, vol_file: str):
        '''

        :param ticker: Stock ticker to analyze
        :param vol_file: corresponding volitility file
        '''

        self.ticker:str = ticker

        self.end = dt.datetime.now()
        # Start Year, Start Month, Start Day
        self.start = dt.datetime(2019, 1, 1)

        self.ticker_df = self.define_ticker_df(ticker)
        # VXX, VXZ, VIXM
        self.vix_df = self.define_ticker_df(vol_file)
        self.plot_ticker()

        print('Loaded Stock Prices:', ticker, vol_file)

        self.examples_list = self.make_data_set()
        print('Created Example Set')

        self.training_data, self.testing_data = self.get_test_train_split()

    def define_ticker_df(self, ticker):
        '''
        Grabs data for a given ticker from the IEX  API
        :param ticker:
        :return:
        '''
        ticker_df = web.DataReader(ticker, 'yahoo', self.start, self.end)
        return ticker_df

    def plot_ticker(self):
        df_to_plot = self.ticker_df['Adj Close']
        plt.plot(df_to_plot.index, df_to_plot.values)
        plt.savefig("spy.png")

        plt.figure()
        df_to_plot = self.vix_df['Adj Close']
        plt.plot(df_to_plot.index, df_to_plot.values)
        plt.savefig("vix.png")

    def make_data_set(self):
        '''
        Creates a list of tuples, where tuple[0] is the data point and tuple[1] is the target
        :return:
        '''

        ticker_arr = np.asarray(self.ticker_df)
        vix_arr = np.asarray(self.vix_df)
        examples_list = []

        for i in range(len(ticker_arr[:-10])):
            ticker_data = ticker_arr[i:i+10][:, 3]
            vix_data = vix_arr[i:i+10][:, 3]
            examples_list.append((ticker_data, vix_data))

        return examples_list

    def get_test_train_split(self, n=5):
        data_points = len(self.examples_list)
        fold_length = int(data_points/n)
        testing_data = self.examples_list[-fold_length:]
        training_data = self.examples_list[(n-1) * fold_length:]

        return training_data, testing_data

