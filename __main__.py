from argparse import ArgumentParser
from data import Data
import pickle as pkl


arg_parser = ArgumentParser(description='Control the project')

spy_data = Data('SPY', 'VXX')

ticker = 'SPY'
vol = 'VXX'
filename = 'vol_prediction' + '/' + ticker+ '_' + vol

with open(filename, 'wb') as f:
    pkl.dump(spy_data, f)

print('Saved:', filename)
