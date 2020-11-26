import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader.data as web
import pickle as pkl

class Data:
    '''
    This class is responsible for loading the data for the VIX and Stocks
    '''

    def __init__(self, ticker: str, vol_file: str, read_vol_csv: bool=True, skip_init=False):
        '''

        :param ticker: Stock ticker to analyze
        :param vol_file: corresponding volitility file
        '''
        self.ticker: str = ticker
        self.vol_file: str = vol_file

        if not skip_init:

            self.end = dt.datetime.now()

            # Start Year, Start Month, Start Day
            self.start = dt.datetime(1990, 1, 1)

            self.ticker_df = self.define_ticker_df(ticker)
            self.ticker_close_array = np.asarray(self.ticker_df['Adj Close'])
            self.ticker_time_array = np.asarray(self.ticker_df['Adj Close'].index)

            t_arr = []
            for i in self.ticker_time_array:
                t_arr.append(pd.to_datetime(i, format='%Y-%m-%dT'))

            self.ticker_time_array = t_arr

            if read_vol_csv:
                self.vix_df = self.read_vol_csv(vol_file)

                close_name = ''
                for col in self.vix_df.columns:
                    if col != 'DATE':
                        close_name = col
                print(close_name)

                self.vix_close_array = np.asarray(self.vix_df[close_name])
                self.vix_time_array = np.asarray(self.vix_df['DATE'])

                t_arr = []
                for i in self.vix_time_array:
                    t_arr.append(pd.to_datetime(i,format='%Y-%m-%d'))
                self.vix_time_array = t_arr
                self.vix_close_array, self.vix_time_array = self.clean_vol_data()

            else:
                # VXX, VXZ, VIXM
                self.vix_df = self.define_ticker_df(vol_file)
                self.vix_close_array = np.asarray(self.vix_df['Adj Close'])
                self.vix_time_array = np.asarray(self.vix_df['Adj Close'].index)

            # self.plot_ticker()

            print('Loaded Stock Prices:', ticker, vol_file)

            self.check_length()
            print("Aligned Data")

            self.examples_list = self.make_data_set()
            print('Created Example Set')

            self.training_data, self.testing_data = self.get_test_train_split()

            self.training_vix_data_split, \
            self.training_spy_data_split, \
            self.testing_vix_data_split,\
            self.testing_spy_data_split = self.partition_data()


    def partition_data(self):
        '''
        Creates an array of testing and training data for easy plotting
        In other words, we split the data into overlapping examples, but never have all the data in one array
        :return:
        '''
        training_vix_data_split = self.training_data[0][1][:]
        training_spy_data_split = self.training_data[0][0][:]
        for i in self.training_data[1:]:
            training_vix_data_split.append(i[1][-1])
            training_spy_data_split.append(i[0][-1])

        testing_vix_data_split = self.testing_data[0][1][:]
        testing_spy_data_split = self.testing_data[0][0][:]
        for i in self.testing_data[1:]:
            testing_vix_data_split.append(i[1][-1])
            testing_spy_data_split.append(i[0][-1])

        return training_vix_data_split, training_spy_data_split, testing_vix_data_split, testing_spy_data_split

    def clean_vol_data(self):
        '''
        The CSV files return strings, and are missing a lot of values.
        Replace missing with linear interpolations and convert to double
        :return:
        '''

        vals_list = []
        time_data = []

        for ind, vix_val in enumerate(self.vix_close_array):
            if vix_val != '.':
                converted_val = np.float(vix_val)
                vals_list.append(converted_val)
                time_data.append(self.vix_time_array[ind])

        return np.asarray(vals_list), np.asarray(time_data)

    def check_length(self):
        '''
        Since the volitility and price data is coming from two different places, check to make sure
        the same time is being compared
        :return:
        '''
        ticker_len = len(self.ticker_close_array)
        vix_len = len(self.vix_close_array)

        if ticker_len > vix_len:
            start_ind = self.get_start_index(self.ticker_time_array, self.vix_time_array)
            self.ticker_time_array, \
            self.vix_time_array,    \
            self.ticker_close_array,\
            self.vix_close_array = self.align_timeseries(self.ticker_time_array,
                                                         self.vix_time_array,
                                                         self.ticker_close_array,
                                                         self.vix_close_array,
                                                         start_ind)
        else:
            start_ind = self.get_start_index(self.vix_time_array, self.ticker_time_array)
            self.vix_time_array,    \
            self.ticker_time_array, \
            self.vix_close_array,   \
            self.ticker_close_array = self.align_timeseries(self.vix_time_array,
                                                            self.ticker_time_array,
                                                            self.vix_close_array,
                                                            self.ticker_close_array,
                                                            start_ind)

    @staticmethod
    def get_start_index(longer_time_array, shorter_time_array):
        '''
        Finds the first data point for which the two datasets are on the same day
        :param longer_time_array:
        :param shorter_time_array:
        :return:
        '''
        start_index = -1

        for ind, time in enumerate(longer_time_array):
            if time == shorter_time_array[0]:
                print(time, shorter_time_array[0], ind)
                start_index = ind
                break

        return start_index

    @staticmethod
    def align_timeseries(longer_time_array, shorter_time_array, longer_vals_arr, shorter_vals_arr, start_index):
        '''
        Since the data is coming from different sources, values may be missing
        To avoid comparing different dates, we use a bit of dynamic time warping to make sure
        that we are only including data points if they are in both data sets
        '''

        longer_time = []
        longer_vals = []

        shorter_time = []
        shorter_vals = []

        ind_diff = 0
        for ind, time in enumerate(longer_time_array[start_index:]):
            if len(shorter_time_array) > ind + ind_diff:
                # One of the dates is missing so we need to shift indexes
                if time != shorter_time_array[ind + ind_diff]:
                    print(time, shorter_time_array[ind + ind_diff])
                    if time > shorter_time_array[ind + ind_diff]:
                        ind_diff += 1
                    else:
                        ind_diff -= 1
                else:
                    shorter_time.append(shorter_time_array[ind + ind_diff])
                    longer_time.append(time)

                    longer_vals.append(longer_vals_arr[ind + start_index])
                    shorter_vals.append(shorter_vals_arr[ind + ind_diff])

        return longer_time, shorter_time, longer_vals, shorter_vals

    @staticmethod
    def replace_missing_val(arr, ind):
        '''
        The vol data is missing data marked by '.'
        Replace with linear interpolation
        '''

        val = '.'
        while val == '.':
            val = arr[ind - 1]
            ind = ind - 1

        lower_val = np.float(val)

        val = '.'
        while val == '.':
            val = arr[ind + 1]
            ind = ind + 1

        upper_val = np.float(val)
        return (lower_val + upper_val)/2

    def define_ticker_df(self, ticker):
        '''
        Grabs data for a given ticker from the IEX  API
        :param ticker:
        :return:
        '''
        ticker_df = web.DataReader(ticker, 'yahoo', self.start, self.end)
        return ticker_df

    def read_vol_csv(self, filename):
        df = pd.read_csv(filename)
        return df

    def plot_ticker(self):
        df_to_plot = self.ticker_df['Adj Close']
        plt.plot(df_to_plot.index, df_to_plot.values)
        plt.savefig("spy.png")

        plt.figure()
        plt.plot(self.vix_time_array, self.vix_close_array)
        plt.savefig("vix.png")

    def make_data_set(self):
        '''
        Creates a list of tuples, where tuple[0] is the data point and tuple[1] is the target
        :return:
        '''
        examples_list = []

        for i in range(len(self.ticker_close_array[:-10])):
            ticker_data = self.ticker_close_array[i:i+10]
            vix_data = self.vix_close_array[i:i+10]
            examples_list.append((ticker_data, vix_data))

        return examples_list

    def get_test_train_split(self, n=5):
        '''
        Splits data into fifths and creates a testing and training set.
        Since we have time series data (correlated!) we cannot shuffle the examples
        :param n:
        :return:
        '''

        data_points = len(self.examples_list)
        fold_length = int(data_points/n)
        testing_data = self.examples_list[-fold_length:]
        training_data = self.examples_list[0:(n-1) * fold_length]

        return training_data, testing_data

    def save_data(self):
        '''
        Saves Data object as pickle
        '''

        ticker = self.ticker
        filename = 'vol_prediction' + '/' + ticker

        with open(filename, 'wb') as f:
            pkl.dump(self.__dict__, f)

        print('Saved:', filename)

    def load_data(self):
        '''
        Loads Data object from pickle
        '''

        ticker = self.ticker
        filename = 'vol_prediction' + '/' + ticker

        with open(filename, 'rb') as f:
            temp_dict = pkl.load(f)
            self.__dict__.update(temp_dict)

        print('Loaded:', filename)
