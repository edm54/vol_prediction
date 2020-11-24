import os
import matplotlib.pyplot as plt
import pandas  as pd
import numpy as np
import datetime as dt
import pyEX as p
from matplotlib import style
import pandas_datareader.data as web
from PIL import Image
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg


class Data:
    '''
    This class is responsible for loading the data for the VIX and Stocks
    '''

    def __init__(self, ticker: str, vol_file: str, read_vol_csv: bool=True):
        '''

        :param ticker: Stock ticker to analyze
        :param vol_file: corresponding volitility file
        '''

        self.ticker:str = ticker

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
            self.vix_df = self.read_vol_csv('vol_prediction/vix_data_1990.csv')
            self.vix_close_array = np.asarray(self.vix_df['VIXCLS'])
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
        # plt.figure()
        # plt.plot(self.vix_time_array, self.vix_close_array)
        # plt.show()
        # canvas = plt.get_current_fig_manager().canvas
        # agg = canvas.switch_backends(FigureCanvasAgg)
        # agg.draw()
        # s, (width, height) = agg.print_to_buffer()
        # # Convert to a NumPy array.
        # X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        # # Pass off to PIL.
        # im = Image.frombytes("RGBA", (width, height), s)
        # im.show()

        # self.plot_ticker()

        print('Loaded Stock Prices:', ticker, vol_file)

        self.check_length()
        print("Aligned Data")

        self.examples_list = self.make_data_set()
        print('Created Example Set')

        self.training_data, self.testing_data = self.get_test_train_split()

    def clean_vol_data(self):
        '''
        The CSV files return strings, and are missing a lot of values.
        Replace missing with linear interpolations and convert to double
        :return:
        '''

        vals_list = []
        time_data = []

        for ind, vix_val in enumerate(self.vix_close_array):
            if vix_val == '.':
                vix_val = self.replace_missing_val(self.vix_close_array, ind)
                # Pass! --> Leave these values out
            else:
                converted_val = np.float(vix_val)
                vals_list.append(converted_val)
                time_data.append(self.vix_time_array[ind])

        return np.asarray(vals_list), np.asarray(time_data)

    def check_length(self):
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
        start_index = -1

        for ind, time in enumerate(longer_time_array):
            if time == shorter_time_array[0]:
                print(time, shorter_time_array[0], ind)
                start_index = ind
                break

        return start_index

    @staticmethod
    def align_timeseries(longer_time_array, shorter_time_array, longer_vals_arr, shorter_vals_arr, start_index):
        longer_time = []
        longer_vals = []

        shorter_time = []
        shorter_vals = []

        ind_diff = 0
        for ind, time in enumerate(longer_time_array[start_index:]):

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
        data_points = len(self.examples_list)
        fold_length = int(data_points/n)
        testing_data = self.examples_list[-fold_length:]
        training_data = self.examples_list[0:(n-1) * fold_length]

        return training_data, testing_data

